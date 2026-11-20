"""SM90a launch-partition and terminal address work for paired native FA."""
from dataclasses import dataclass, replace
import re
from h100_paired_mainloop import NativeLongMemoryLoopPair, _sass_instructions, _kv_bounds

# H100 unified L1/shared capacity and selectable shared carveouts (KiB).
_CARVEOUTS = (0, 8, 16, 32, 64, 100, 132, 164, 196, 228)
_ADDRESS_OPS = frozenset(('IMAD', 'UIMAD', 'IADD3', 'UIADD3', 'LEA', 'ULEA',
                          'LOP3', 'ULOP3', 'SHF', 'USHF', 'SGXT', 'PRMT'))

@dataclass(frozen=True)
class SegmentWork:
    carveout_bytes: int
    register_count: int
    single_cta: bool
    loop_local_bytes: int | None
    tail_address_ops: int | None

@dataclass(frozen=True)
class WriterWork(SegmentWork):
    tail_live_bytes: int | None
    tail_load_words: int


def _local_words(inst):
    words = set()
    loads = 0
    for _, op, args in inst:
        root, *modifiers = op.split('.')
        if root not in ('LDL', 'STL'):
            continue
        offset = re.search(r'\[R1(?:\+(0x[0-9a-fA-F]+))?\]', args)
        if not offset or int(offset[1] or '0',16)%4:
            return None, 0
        base = int(offset[1] or '0',16)//4
        width = 4 if '128' in modifiers else 2 if '64' in modifiers else 1
        words.update(range(base,base+width))
        if root=='LDL':
            loads += width
    return words, loads


def segment_work(sass, resource_usage, dynamic_shared, threads, *, writer=False):
    resource = re.search(r'Function\s+frisk_execution:\s*REG:(\d+)\s+STACK:\d+\s+SHARED:(\d+)', resource_usage)
    if not resource:
        raise ValueError("missing SM90 native resource usage")
    regs, shared = map(int, resource.groups())
    # cuobjdump includes the CUDA-reserved bytes in SHARED on this path.
    needed = shared + dynamic_shared
    carveout = next((k*1024 for k in _CARVEOUTS if k*1024 >= needed), None)
    if carveout is None:
        raise ValueError("launch exceeds the H100 shared carveout")
    single = 2*threads*regs > 65536 or 2*needed > 228*1024
    inst = _sass_instructions(sass)
    lo, hi = _kv_bounds(inst)[0]
    words, _ = _local_words(x for x in inst if lo <= x[0] <= hi)
    # A local word is private to each thread. Count distinct offsets only.
    footprint = None if words is None else 4*threads*len(words)
    exits = {int(pc,16) for pc in re.findall(r'^\s*/\*([0-9a-fA-F]+)\*/\s+EXIT\s*;',sass,re.M)}
    tail = []
    for pc, op, args in inst:
        if pc <= hi:continue
        if pc in exits:break
        tail.append((pc,op,args.strip()))
    else:
        raise ValueError("missing native exit after the KV loop")
    # Exclude the repeated normalization slow paths before the final call.
    start = max((i+1 for i,(_,op,_) in enumerate(tail) if op.startswith('CALL')),default=0)
    suffix = tail[start:]
    # A backward branch in the suffix requires an execution multiplicity.
    repeated = any(op=='BRA' and re.fullmatch(r'0x[0-9a-fA-F]+',args)
                   and suffix[0][0] <= int(args,16) <= pc for pc,op,args in suffix)
    address = None if repeated else sum(op.split('.')[0] in _ADDRESS_OPS for _,op,_ in suffix)
    base = SegmentWork(carveout,regs,single,footprint,address)
    if not writer:
        return base
    # Spill slots surviving the KV backedge can compete with the normalizer
    # and writeback for L1. Tail loads execute once, outside the KV trip count.
    # Restrict to straight-line/forward-control tails. Out-of-line subroutine
    # bodies after EXIT are excluded; static predication is an approximation.
    backward = any(op=='BRA' and re.search(r'0x[0-9a-fA-F]+$',args) and
                   int(args.rsplit(' ',1)[-1],16)<=pc for pc,op,args in tail)
    tail_words, loads = _local_words(tail)
    live = None if backward or words is None or tail_words is None else 4*threads*len(words|tail_words)
    return WriterWork(**vars(base),tail_live_bytes=live,tail_load_words=loads)

def _extra_sectors(work, common_carveout, load_words, warp_iterations):
    footprint = work.loop_local_bytes
    if not footprint:return 0
    l1 = 256*1024 - work.carveout_bytes
    common_l1 = 256*1024 - common_carveout
    uncovered = max(0, footprint-l1) - max(0, footprint-common_l1)
    # Four 32-byte sectors for a warp of contiguous per-thread 32-bit words.
    return (4*load_words*warp_iterations*uncovered + footprint//2)//footprint

@dataclass(frozen=True)
class NativeSegmentPair(NativeLongMemoryLoopPair):
    cache_miss_sectors_delta: int
    tail_address_issues_delta: int

    def mlir_field(self):
        if max(abs(self.cache_miss_sectors_delta),abs(self.tail_address_issues_delta)) > 2**30:
            raise ValueError("segment feedback exceeds supported range")
        return (super().mlir_field()[:-1] +
                f", cache_miss_sectors_delta = {self.cache_miss_sectors_delta} : i64, "
                f"tail_address_issues_delta = {self.tail_address_issues_delta} : i64}}")

def attach_segments(pair, direct, striped, warps):
    common = min(direct.carveout_bytes, striped.carveout_bytes)
    miss = 0
    if (direct.single_cta and striped.single_cta and
            direct.loop_local_bytes is not None and striped.loop_local_bytes is not None):
        miss = (_extra_sectors(striped,common,pair.striped.local_load_words,pair.warp_iterations)
                - _extra_sectors(direct,common,pair.direct.local_load_words,pair.warp_iterations))
    addr = 0
    if direct.tail_address_ops is not None and striped.tail_address_ops is not None:
        addr = warps*(striped.tail_address_ops-direct.tail_address_ops)
    return NativeSegmentPair(**vars(pair),cache_miss_sectors_delta=miss,tail_address_issues_delta=addr)

@dataclass(frozen=True)
class NativeWriterPair(NativeSegmentPair):
    tail_cache_miss_sectors_delta: int

    def mlir_field(self):
        if abs(self.tail_cache_miss_sectors_delta)>2**30:
            raise ValueError("writer feedback exceeds supported range")
        return (super().mlir_field()[:-1] +
                f", tail_cache_miss_sectors_delta = {self.tail_cache_miss_sectors_delta} : i64}}")


def attach_writer(pair, direct, striped, warps):
    common = min(direct.carveout_bytes, striped.carveout_bytes)
    delta = 0
    if (direct.single_cta and striped.single_cta and direct.tail_live_bytes is not None
            and striped.tail_live_bytes is not None):
        def tail_sectors(arm):
            live = replace(arm,loop_local_bytes=arm.tail_live_bytes)
            return _extra_sectors(live,common,arm.tail_load_words,warps)
        delta = tail_sectors(striped)-tail_sectors(direct)
    return NativeWriterPair(**vars(pair),tail_cache_miss_sectors_delta=delta)


@dataclass(frozen=True)
class GuardedDrain:
    address: int
    interstore: int


def guarded_drain(sass):
    """Straight-line scalar writer after normalization, with actual row guards."""
    inst = _sass_instructions(sass)
    _, backedge = _kv_bounds(inst)[0]
    exits = {int(pc,16) for pc in re.findall(r'^\s*/\*([0-9a-fA-F]+)\*/\s+EXIT\s*;',sass,re.M)}
    end = min((pc for pc in exits if pc>backedge),default=None)
    if end is None:
        return None
    tail = [(pc,op,args) for pc,op,args in inst if backedge<pc<end]
    calls = [i for i,(_,op,_) in enumerate(tail) if op.startswith('CALL')]
    if not calls:
        return None
    merge = next((i for i in range(calls[-1]+1,len(tail)) if tail[i][1]=='BSYNC'),None)
    if merge is None:
        return None
    body = tail[merge+1:]
    stores = [(pc,op) for pc,op,_ in body if op.split('.')[0]=='STG']
    if len(stores)<2 or any(op!='STG.E' for _,op in stores):
        return None
    first,last = stores[0][0],stores[-1][0]
    masks = {int(pc,16) for pc in re.findall(r'^\s*/\*([0-9a-fA-F]+)\*/\s+@!?P[0-6]\s+STG\.E\s',sass,re.M)}
    if not any(pc in masks for pc,_ in stores):
        return None
    if any(op.split('.')[0] in {'BRA','BRX','JMP','JMX','CALL','RET','WARPGROUP','HGMMA'} or
           (op=='EXIT' and pc!=last-16) for pc,op,_ in body):
        return None
    issued = [pc for pc,op,_ in body if op.split('.')[0] in _ADDRESS_OPS]
    return GuardedDrain(len(issued),sum(first<pc<last for pc in issued))


@dataclass(frozen=True)
class NativeDrainPair(NativeWriterPair):
    drain_address_issues_delta: int
    drain_interstore_issues_delta: int

    def mlir_field(self):
        if max(abs(self.drain_address_issues_delta),abs(self.drain_interstore_issues_delta))>2**30:
            raise ValueError("guarded drain feedback exceeds supported range")
        return (super().mlir_field()[:-1] +
                f", drain_address_issues_delta = {self.drain_address_issues_delta} : i64, "
                f"drain_interstore_issues_delta = {self.drain_interstore_issues_delta} : i64}}")


def attach_drain(pair,direct,striped,warps):
    address = interstore = 0
    if direct is not None and striped is not None:
        address = warps*(striped.address-direct.address)
        interstore = warps*(striped.interstore-direct.interstore)
    return NativeDrainPair(**vars(pair),drain_address_issues_delta=address,
                           drain_interstore_issues_delta=interstore)


@dataclass(frozen=True)
class NativeWavePair(NativeDrainPair):
    def mlir_field(self):
        return super().mlir_field()[:-1] + ", guarded_wave = true}"
