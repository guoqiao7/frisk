"""Extract fixed-pipeline SM90a loop work from two cuobjdump -sass listings.

Supports a uniform outer KV loop containing one repeated QK loop and unrolled
PV. Trip counts come from the frozen pipeline, including padded key tiles.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ReplayLoopPair:
    source: str
    warp_issues_delta: int
    warp_imads_delta: int

    def __post_init__(self):
        if not self.source.startswith("#frisk.distributed<") or not self.source.endswith(">"):
            raise ValueError("expected a materialized distributed source encoding")
        if any(type(n) is not int or abs(n) > 2**30
               for n in (self.warp_issues_delta, self.warp_imads_delta)):
            raise ValueError("invalid per-CTA warp instruction delta")

    def mlir_field(self):
        return (f"paired_mainloop = {{source = {self.source}, "
                f"warp_issues_delta = {self.warp_issues_delta} : i64, "
                f"warp_imads_delta = {self.warp_imads_delta} : i64}}")


def _sass_instructions(sass):
    return [(int(pc, 16), op, args) for pc, op, args in re.findall(
        r'^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?\w+\s+)?([\w.]+)\s*([^;\n]*)\s*;',
        sass, re.M)]


def _kv_bounds(inst):
    mma = [pc for pc, op, _ in inst if op.startswith("HGMMA.")]
    branches = [(int(args.strip(), 16), pc) for pc, op, args in inst
                if op == "BRA" and re.fullmatch(r'0x[0-9a-fA-F]+', args.strip())]
    outer = [(lo, hi) for lo, hi in branches if mma and lo <= min(mma) <= max(mma) <= hi]
    if len(outer) != 1:
        raise ValueError("no unique structured KV loop enclosing the tensor operations")
    return outer[0], branches


def _loop_work(sass, qk_trips, entry="replay", mma_group_size=1, stages=((64,8),(64,8)), memory=False, long_loads=False):
    if re.findall(r'Function\s*:\s*(\S+)', sass) != [entry] or "code for sm_90a" not in sass:
        raise ValueError("expected a single matching SM90a entry")
    inst = _sass_instructions(sass)
    (lo, hi), branches = _kv_bounds(inst)
    inner = [(s, e) for s, e in branches if lo < s <= e < hi]
    if len(inner) != (1 if entry=="replay" else 0):
        raise ValueError("no unique nested QK loop")
    start, end = inner[0] if inner else (-1,-1)
    issues = imads = 0
    pipeline = []
    for pc, op, args in inst:
        if not lo <= pc <= hi:
            continue
        root, *modifiers = op.split(".")
        if root in {"EXIT", "CALL", "JMP", "JMX", "BRX", "RET", "BSSY", "BSYNC"} or (
                root == "BRA" and pc not in (hi, end)):
            raise ValueError("unsupported control flow inside the replay main loop")
        nested = start <= pc <= end
        weight = qk_trips if nested else 1
        issues += weight
        # The CUDA disassembler also spells moves, shifts and adds as IMAD
        # aliases. These already contribute to total issues.
        if root == "IMAD" and not {"MOV", "SHL", "IADD"}.intersection(modifiers):
            imads += weight
        if root in {"HGMMA", "WARPGROUP", "BAR"}:
            # Operand class distinguishes register/shared A. Register numbers
            # and descriptor registers can change during allocation.
            pipeline.append((nested, op, (args.split(",")[1].strip().startswith("gdesc") if root == "HGMMA" else args.strip())))
    if entry=="frisk_execution":
        # The position of each wait matters as well as its count. Preserve
        # the SS/RS boundary and reject a different atom or wait depth.
        actual = [(op, arg) for _, op, arg in pipeline
                  if op.startswith(("HGMMA.", "WARPGROUP.DEPBAR"))]
        expected = []
        for shared_a, (atom_n, stage_issues) in zip((True,False),stages,strict=True):
            if atom_n not in (32,64,128) or stage_issues < 1 or stage_issues % mma_group_size:
                raise ValueError("unsupported native issue group")
            for _ in range(stage_issues // mma_group_size):
                expected.extend([(f"HGMMA.64x{atom_n}x16.F32", shared_a)] * mma_group_size)
                expected.append(("WARPGROUP.DEPBAR.LE", "gsb0, 0x0"))
        if actual != expected:
            raise ValueError("native SS/RS instructions or waits disagree with the requested group size")
    elif not any(n and op.startswith("HGMMA.") for n, op, _ in pipeline):
        raise ValueError("missing repeated QK tensor group")
    detail = None
    if memory:
        body = [(op,args) for pc,op,args in inst if lo <= pc <= hi]
        detail = _long_memory_work(body) if long_loads else _memory_work(body)
    return issues, imads, pipeline, detail


def paired_replay_mainloop(direct_sass, striped_sass, *, source, kv_tiles, qk_trips, warps):
    """Estimate the signed routed-minus-direct warp work of one CTA, without running it."""
    if any(type(n) is not int or not 0 < n <= 2**30 for n in (kv_tiles, qk_trips, warps)):
        raise ValueError("invalid frozen loop or warp count")
    direct, routed = [_loop_work(s, qk_trips) for s in (direct_sass, striped_sass)]
    if direct[2] != routed[2]:
        raise ValueError("paired candidates have different tensor or synchronization schedules")
    scale = kv_tiles * warps
    return ReplayLoopPair(source, (routed[0] - direct[0]) * scale,
                          (routed[1] - direct[1]) * scale)


@dataclass(frozen=True)
class NativeLoopPair(ReplayLoopPair):
    def mlir_field(self):
        return super().mlir_field().replace("paired_mainloop = {", 'paired_mainloop = {backend = "native", ', 1)


@dataclass(frozen=True)
class LoopMemory:
    global_wait: int
    local_wait: int
    local_load_words: int
    local_store_words: int


def _registers(operands):
    words = set()
    for reg, size in re.findall(r'\bR(\d+)(?:\.(64|128))?', operands):
        base = int(reg)
        words.update(range(base, base + int(size or 32)//32))
    return words


# Latency uncertainty in warp issue opportunities. All native SM90 kernels
# use the same geometric grid and equal weights, independent of their shape.
RAW_HORIZONS = (64, 128, 256)


@dataclass(frozen=True)
class LongLoopMemory(LoopMemory):
    # (exposed global, exposed local) for each horizon beyond the base one.
    longer_waits: tuple


def _memory_events(instructions):
    for op, operands in instructions:
        root, *mods = op.split('.')
        first, sep, remaining = operands.partition(',')
        first = first.strip()
        dest = re.fullmatch(r'R(\d+)', first) if sep else None
        if root.startswith(('ST','RED')) or root in ('BAR','WARPGROUP'):
            dest = None
        width = 4 if '128' in mods else 2 if ('64' in mods or 'WIDE' in mods) else 1
        used = _registers(remaining if dest else operands)
        if root in ('STL','STS','STG') and width > 1:
            value = re.match(r'\s*R(\d+)\b',remaining)
            if value:
                base = int(value[1])
                used.update(range(base,base+width))
        outputs = range(int(dest[1]),int(dest[1])+width) if dest else ()
        yield root, used, outputs, width


def _score_memory(events, horizon):
    """RAW-only issue-slot simulation; predicates are treated as active."""
    ready = {}
    clock = global_wait = local_wait = loads = stores = 0
    for root, used, outputs, width in events:
        latest, kind = max((ready[r] for r in used if r in ready), default=(0,''))
        delay = max(0, latest-clock)
        if kind == 'LDG':
            global_wait += delay
        elif kind == 'LDL':
            local_wait += delay
        clock += delay
        if root=='LDL':
            loads += width
        elif root=='STL':
            stores += width
        for word in outputs:
            if root in ('LDG','LDL'):
                ready[word] = (clock+horizon,root)
            else:
                ready.pop(word,None)
        clock += 1
    return LoopMemory(global_wait,local_wait,loads,stores)


def _memory_work(instructions, horizon=64):
    return _score_memory(_memory_events(instructions),horizon)


def _long_memory_work(instructions):
    events = tuple(_memory_events(instructions))
    base,*longer = [_score_memory(events,horizon) for horizon in RAW_HORIZONS]
    return LongLoopMemory(base.global_wait, base.local_wait, base.local_load_words,
                          base.local_store_words, tuple((v.global_wait,v.local_wait) for v in longer))


@dataclass(frozen=True)
class NativeMemoryLoopPair(NativeLoopPair):
    direct: LoopMemory
    striped: LoopMemory
    warp_iterations: int

    @property
    def warp_load_wait_delta(self):
        a,b=self.direct,self.striped
        return self.warp_iterations*(b.global_wait+b.local_wait-a.global_wait-a.local_wait)

    @property
    def warp_local_word_delta(self):
        a,b=self.direct,self.striped
        return self.warp_iterations*(b.local_load_words+b.local_store_words-a.local_load_words-a.local_store_words)

    def mlir_field(self):
        values=(self.warp_load_wait_delta,self.warp_local_word_delta)
        if any(abs(x)>2**30 for x in values):
            raise ValueError("native memory feedback exceeds supported range")
        return (super().mlir_field()[:-1] +
                f", warp_load_wait_delta = {values[0]} : i64, warp_local_word_delta = {values[1]} : i64}}")


@dataclass(frozen=True)
class NativeLongMemoryLoopPair(NativeMemoryLoopPair):
    direct: LongLoopMemory
    striped: LongLoopMemory

    @property
    def warp_load_wait_delta(self):
        # Average signed per-CTA bills and round once. The traffic counts are
        # common to all scenarios and the inherited local-word bill is paid once.
        delta = super().warp_load_wait_delta
        for a,b in zip(self.direct.longer_waits,self.striped.longer_waits,strict=True):
            delta += self.warp_iterations*(sum(b)-sum(a))
        count = 1+len(self.direct.longer_waits)
        magnitude = (abs(delta)+count//2)//count
        return -magnitude if delta<0 else magnitude


def paired_native_mainloop(direct_sass, striped_sass, *, source, kv_tiles, warps, mma_group_size=1, stages=((64,8),(64,8)), memory=False, long_loads=False):
    """Both QK and PV are unrolled inside a single native KV backedge."""
    if any(type(v) is not int or v <= 0 for v in (kv_tiles,warps)):
        raise ValueError("invalid native trip or warp count")
    if type(mma_group_size) is not int or mma_group_size not in (1, 4):
        raise ValueError("native WGMMA group size must be 1 or 4")
    if long_loads and not memory:
        raise ValueError("long-load scoring requires native memory feedback")
    a,b = [_loop_work(text, 1, "frisk_execution", mma_group_size, stages, memory, long_loads) for text in (direct_sass,striped_sass)]
    if a[2] != b[2]:
        raise ValueError("native tensor or synchronization schedule changed between candidates")
    scale = kv_tiles*warps
    work = (source,(b[0]-a[0])*scale,(b[1]-a[1])*scale)
    if long_loads:
        return NativeLongMemoryLoopPair(*work,a[3],b[3],scale)
    if memory:
        return NativeMemoryLoopPair(*work,a[3],b[3],scale)
    return NativeLoopPair(*work)
