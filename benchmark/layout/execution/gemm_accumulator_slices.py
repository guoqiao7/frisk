"""SM90 CUDA replay of a solved, zero-initialized SS GEMM with optional column sum.

This entry takes *layouts* from frisk-opt and an explicit dense GEMM problem.
It does not interpret arbitrary surrounding MLIR. Each CTA owns an independent
row-major f16 A/B (16-byte aligned), f32 C and optional f32 R. Launch one block
per independent problem with 256 threads and the returned dynamic shared byte
count. By default both residency modes stage adjacent K16 steps in distinct slots; the
wait policy controls how many WGMMA instructions precede the drain. Static
selection ranks all contiguous resident atom counts and compiles one image.
An explicit atom count or the legacy full/slice controls override selection.
Use --output kernel.cu to inspect CUDA, or --output kernel.cubin to compile.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


def _one(pattern, text):
    hits = re.findall(pattern, text)
    if len(hits) != 1:
        raise ValueError(f"expected one occurrence of {pattern}, got {len(hits)}")
    return hits[0]


def _numbers(text):
    return tuple(map(int, text.split(",")))


def _owners(line):
    """Expand the selected F2 map in register-fast PTX carrier order."""
    enc = line.rsplit("#frisk.distributed<", 1)[-1]
    names = re.findall(r'"(\w+)"', _one(r"inputs = \[([^\]]+)\]", enc))
    widths = _numbers(_one(r"input_bits = \[([^\]]+)\]", enc))
    output = _numbers(_one(r"output_bits = \[([^\]]+)\]", enc))
    topology = _numbers(_one(r"topology = \[([^\]]+)\]", enc))
    if topology[1:] != (32, 4, 2, 1) or len(names) != len(set(names)):
        raise ValueError("expected two consumer warpgroups and unique carriers")
    data, nrows, ncols = _one(r"matrix = dense<(.*?)> : tensor<(\d+)x(\d+)xi1>", enc)
    nrows, ncols = int(nrows), int(ncols)
    if (nrows, ncols) != (sum(output), sum(widths)):
        raise ValueError("inconsistent distributed matrix")
    if data.startswith('"0x'):
        raw = np.frombuffer(bytes.fromhex(data[3:-1]), dtype=np.uint8)
        bits = np.unpackbits(raw, bitorder="little")[:nrows*ncols]
    else:
        bits = np.array([s == "true" for s in re.findall(r"true|false", data)])
    rows = bits.reshape(nrows, ncols)
    regs = topology[0]
    idx = np.arange(256*regs)
    cols = []
    for name, width in zip(names, widths, strict=True):
        i = ("register", "lane", "warp", "warp_group", "cta").index(name)
        if 1 << width != topology[i]:
            raise ValueError("carrier width does not match topology")
        carrier = idx // int(np.prod(topology[:i])) % topology[i]
        cols.extend((carrier >> j) & 1 for j in range(width))
    logical = rows.astype(np.int32) @ np.asarray(cols, np.int32) & 1
    address = np.zeros(idx.size, dtype=np.int32)
    begin = 0
    for width in output:
        address <<= width
        for i in range(width):
            address |= logical[begin+i] << i
        begin += width
    return address.reshape(256, regs)


@dataclass(frozen=True)
class GemmLayouts:
    m: int
    n: int
    atom: int
    origins: tuple  # In original register order: (row0, col0, row1, col1).
    reduced_writers: tuple | None  # thread and register column basis


def _check_row_major_writer(ir, store, shape):
    target = _one(r"frisk\.tile_store %[^,]+, (%[\w]+)", store)
    view = _one(rf"(?m)^\s*{re.escape(target)} = (frisk\.layout_view .*)$", ir)
    ref = _one(r"map = (#map\d*)>", view)
    index = _one(rf"(?m)^{ref} = affine_map<(.*)>$", ir)
    extent = _numbers(_one(r"input_extents = \[([^\]]+)\]", view))
    expected = (f"(d0, d1) -> (d0 * {shape[1]*4} + d1 * 4, 0)"
                if len(shape)==2 else "(d0) -> (d0 * 4, 0)")
    if extent != shape or index != expected or "memory_space Global" not in view:
        raise ValueError("expected dense row-major Global FP32 writeback")


def read_layouts(ir):
    mma = _one(r"(?m)^.* = frisk\.mma .*?$", ir)
    stores = re.findall(r"(?m)^\s*frisk\.tile_store .*?$", ir)
    m, n = (int(_one(rf"\b{axis} = (\d+) : i64", mma)) for axis in ("m", "n"))
    atom = int(_one(r"atom = array<i64: 64, (\d+), 16>", mma))
    if (atom not in (32, 64, 128, 256) or m % 64 or n % atom or
            'form = "ss"' not in mma or 'input_type = f16' not in mma or
            'accumulator_type = f32' not in mma or 'target = "sm_90a"' not in mma or
            not re.search(r"\bk = 16 : i64", mma) or len(stores) not in (1, 2)):
        raise ValueError("expected f16 SS K16 GEMM with C and optional dim0 sum")
    for role, leading, major, stride, swizzle in (("a", m*16, "k", 128, 0),
                                                  ("b", 256, "mn", n*16, 32)):
        d = _one(rf"{role}_descriptor = #frisk.mma_descriptor<\{{([^}}]+)\}}>", mma)
        fields = (int(_one(r"leading = (\d+)", d)), _one(r'major = "(\w+)"', d),
                  int(_one(r"stride = (\d+)", d)), int(_one(r"swizzle = (\d+)", d)))
        if fields != (leading, major, stride, swizzle):
            raise ValueError(f"unsupported {role.upper()} descriptor")
        expected = []
        for start in range(0, m if role == "a" else n, 64 if role == "a" else atom):
            addr = start*16
            expected.extend((start, 0, addr, (addr>>7)&7 if swizzle else 0))
        if _numbers(_one(r"entries = array<i64: ([^>]+)>", d)) != tuple(expected):
            raise ValueError(f"unsupported {role.upper()} descriptor starts")
    _check_row_major_writer(ir, stores[0], (m,n))
    if len(stores)==2:
        _check_row_major_writer(ir, stores[1], (n,))
    acc = _owners(mma)
    if not np.array_equal(acc, _owners(stores[0])):
        raise ValueError("sliced C delivery requires the certified in-place writer")
    if acc.size != m*n or not np.array_equal(np.sort(acc.ravel()), np.arange(m*n)):
        raise ValueError("accumulator is not a bijection on C")
    tid, reg = np.indices((256, atom//2))
    rr = tid//32%4*16 + tid%32//4 + reg//2%2*8
    cc = tid%4*2 + reg%2 + reg//4*8
    origins = []
    for begin in range(0, acc.shape[1], atom//2):
        part = acc[:, begin:begin+atom//2]
        row, col = np.divmod(part[[0, 128], 0], n)
        if (np.any(row%64) or np.any(col%atom) or
                not np.array_equal(part, (row[tid//128]+rr)*n + col[tid//128]+cc)):
            raise ValueError("slice does not match the selected WGMMA register order")
        origins.append((int(row[0]), int(col[0]), int(row[1]), int(col[1])))
    gm,gn = _numbers(_one(r"grid = array<i64: ([^>]+)>", mma))
    repeats = _numbers(_one(r"repeats = array<i64: ([^>]+)>", mma))
    if gm*gn != 2 or repeats != (m//(64*gm),n//(atom*gn),1):
        raise ValueError("inconsistent MMA grid and atom repeats")
    interleaved = 'row_partition = "interleaved64"' in mma
    for i,origin in enumerate(origins):
        mr,nr = divmod(i,repeats[1])
        expected = []
        for wg in range(2):
            row = (mr*gm+wg//gn) if interleaved else (wg//gn*repeats[0]+mr)
            expected.extend((row*64, (wg%gn*repeats[1]+nr)*atom))
        if tuple(expected) != origin:
            raise ValueError("output partition disagrees with certified grid")
    writers = None
    reductions = re.findall(r"(?m)^.* = frisk\.reduce_tensor .*?$", ir)
    if len(reductions) != len(stores)-1:
        raise ValueError("expected one R writer per reduction")
    if reductions:
        reduce, = reductions
        if 'dim = 0 : i64' not in reduce or 'kind = "sum"' not in reduce:
            raise ValueError("expected a column sum")
        if not np.array_equal(_owners(reduce.split(" -> ")[0]), acc):
            raise ValueError("reduction source differs from the accumulator")
        table = _owners(stores[1])
        if not np.array_equal(_owners(reduce), table):
            raise ValueError("R writer differs from the reduction layout")
        flat = table.ravel()
        basis = tuple(int(flat[1<<b]) for b in range(len(flat).bit_length()-1))
        nonzero = [x for x in basis if x]
        if (len(nonzero) != n.bit_length()-1 or set(nonzero) !=
                {1<<i for i in range(n.bit_length()-1)} or flat[0] != 0):
            raise ValueError("expected a separable first-owner R map")
        rbits = table.shape[1].bit_length()-1
        writers = (basis[rbits:], basis[:rbits])
    return GemmLayouts(m, n, atom, tuple(origins), writers)


def _intervals(starts, width):
    result = []
    for start in sorted(set(starts)):
        if result and result[-1][1] == start:
            result[-1] = (result[-1][0], start+width)
        else:
            result.append((start, start+width))
    return result


def _atom_groups(plan, width):
    for start in range(0, len(plan.origins), width):
        yield plan.origins[start:start + width]


def _copy_regions(group, atom):
    rows = _intervals([v for r, c, s, d in group for v in (r, s)], 64)
    cols = _intervals([v for r, c, s, d in group for v in (c, d)], atom)
    return rows, cols


def rank_residencies(plan, k, *, staged_k_tiles=2, grid_ctas=None, sm_count=114):
    """Pre-ptxas relative byte-cost estimates per CTA, with a fixed atom order.

    Each group retains its accumulators for all K. Charge the actual copy
    intervals once per group, even when the two warpgroups share an operand.
    Estimate one local store/load per excess live FP32 word per K16 update.
    The temporary register allowance is a proxy, not a ptxas allocation.
    """
    if type(k) is not int or k <= 0 or k % 16 or staged_k_tiles not in (1, 2):
        raise ValueError("positive K multiple of 16 and one or two K slots required")
    if type(sm_count) is not int or sm_count < 1 or (
            grid_ctas is not None and (type(grid_ctas) is not int or grid_ctas < 1)):
        raise ValueError("positive SM and CTA counts required")
    # A standalone compile targets throughput unless a smaller grid is known.
    target_ctas = 2 if grid_ctas is None or grid_ctas > sm_count else 1
    # H100 has a 228 KiB shared arena, including 1 KiB reserved per CTA.
    shared_limit = (228 * 1024) // (shared_bytes(plan) + 1024)
    if not shared_limit:
        raise ValueError("operand slots and reduction scratch exceed the SM90 CTA shared budget")
    target_ctas = min(target_ctas, shared_limit)
    depth = min(k // 16, staged_k_tiles)
    # CTA addressing, vector copy temporaries, and two 64-bit descriptors/slot.
    loop_temps = 16 + 8 + 4 + 4 * depth  # Four 64-bit base pointers.
    # Reduction values/shuffles coexist with the undelivered C registers.
    delivery_temps = 16 + (plan.atom // 4 if plan.reduced_writers is not None else 0)
    overhead = max(loop_temps, delivery_temps)
    base_bytes = 2 * k * (plan.m + plan.n)
    candidates = []
    for width in range(1, len(plan.origins) + 1):
        a_bytes = b_bytes = spill_bytes = groups = peak = 0
        for group in _atom_groups(plan, width):
            rows, cols = _copy_regions(group, plan.atom)
            a_bytes += 2 * k * sum(end - start for start, end in rows)
            b_bytes += 2 * k * sum(end - start for start, end in cols)
            live = len(group) * (plan.atom // 2)
            peak = max(peak, live + overhead)
            # Delivery executes once; avoid multiplying epilogue-only pressure by K.
            loop_excess = max(0, live + loop_temps - 255)
            delivery_excess = max(0, live + delivery_temps - 255)
            spill_bytes += 2 * 4 * 256 * (loop_excess * (k // 16) + delivery_excess)
            groups += 1
        candidates.append(dict(resident_atoms=width, slices=groups,
                               peak_register_estimate=peak, a_bytes=a_bytes, b_bytes=b_bytes,
                               repeated_ab_bytes=a_bytes + b_bytes - base_bytes,
                               estimated_spill_bytes=spill_bytes,
                               score_bytes=a_bytes + b_bytes + spill_bytes))
    # Estimate the latency-sensitive fraction using the larger of the two
    # compulsory workloads (operand staging and C/R delivery). Extra copy
    # and local traffic are always additive. The maximum is a scoring proxy;
    # it does not imply that the epilogue physically overlaps the K loop.
    output_bytes = 4 * (plan.m * plan.n + (plan.n if plan.reduced_writers is not None else 0))
    compulsory = max(base_bytes, output_bytes)
    for candidate in candidates:
        registers = min(255, candidate["peak_register_estimate"])
        allocated = ((registers + 7) // 8) * 8  # 256 register words per warp.
        resident = min(shared_limit, 65536 // (256 * allocated))
        lost = max(0, target_ctas - resident)
        charge = max(output_bytes, base_bytes * (1 + lost)) - compulsory
        candidate.update(resident_ctas_estimate=resident, target_ctas=target_ctas,
                         compulsory_units=compulsory, residency_cost_units=charge,
                         score_units=compulsory + charge + candidate["repeated_ab_bytes"]
                                     + candidate["estimated_spill_bytes"])
    return sorted(candidates, key=lambda c: (c["score_units"], c["slices"],
                                             c["peak_register_estimate"], c["resident_atoms"]))


def _residency(plan, k, staged_k_tiles, sliced, resident_atoms, grid_ctas, sm_count):
    if sliced is not None:
        if type(sliced) is not bool or resident_atoms is not None:
            raise ValueError("choose either sliced or resident_atoms")
        resident_atoms = 1 if sliced else len(plan.origins)
    if resident_atoms is None:
        return rank_residencies(plan, k, staged_k_tiles=staged_k_tiles, grid_ctas=grid_ctas, sm_count=sm_count)[0]["resident_atoms"]
    if type(resident_atoms) is not int or not 1 <= resident_atoms <= len(plan.origins):
        raise ValueError("resident_atoms outside the certified atom count")
    return resident_atoms



def _mma(atom, count):
    regs = atom//2
    dest = ", ".join(f"%{i}" for i in range(regs))
    asm = ["wgmma.fence.sync.aligned;"]
    for j in range(count):
        asm.append(f"wgmma.mma_async.sync.aligned.m64n{atom}k16.f32.f16.f16 "
                   f"{{{dest}}}, %{regs+2*j}, %{regs+2*j+1}, 1, 1, 1, 0, 1;")
    asm.extend(["wgmma.commit_group.sync.aligned;", "wgmma.wait_group.sync.aligned 0;"])
    operands = ", ".join(f'"+f"(d[{i}])' for i in range(regs))
    descriptors = [f"{v}{j}" for j in range(count) for v in ("a","b")]
    args = ", ".join("uint64_t "+v for v in descriptors)
    inputs = ", ".join(f'"l"({v})' for v in descriptors)
    return (f"__device__ __forceinline__ void issue{count}(float* d, {args}) {{\n"
            + '  asm volatile("' + "\\n\\t".join(asm) + f'" : {operands} : {inputs} : "memory");\n}}')


def shared_bytes(plan):
    tiles = 2*(plan.m+plan.n)*32
    scratch = max(tiles, 8*plan.atom*4)
    partials = plan.m//64*plan.n*4 if plan.reduced_writers is not None else 0
    return max(65536, scratch+partials)


def _write_columns(plan):
    if plan.reduced_writers is None:
        return []
    threads, regs = plan.reduced_writers
    tid_mask = sum(1<<i for i,x in enumerate(threads) if not x)
    reg_mask = sum(1<<i for i,x in enumerate(regs) if not x)
    terms = [f"(({var}>>{i}&1)*{v})" for var,bits in (("tid",threads),("s",regs))
             for i,v in enumerate(bits) if v]
    index = " | ".join(terms) or "0"
    return [f"if (!(tid & {tid_mask})) {{", "#pragma unroll",
            f"for (int s=0;s<{1<<len(regs)};++s) if (!(s & {reg_mask})) {{",
            f"int j = {index};", f"R[j]=finish_column<{plan.m//64},{plan.n}>(partial,j);", "}}"]


def emit(plan, k, *, sliced=None, resident_atoms=None, wait_group=1, staged_k_tiles=2, grid_ctas=None, sm_count=114):
    if k<=0 or k%16 or wait_group not in (1,2):
        raise ValueError("positive K multiple of 16 and wait group 1 or 2 required")
    if staged_k_tiles not in (1,2) or wait_group > staged_k_tiles:
        raise ValueError("wait group must fit the staged K tiles")
    m,n,atom = plan.m,plan.n,plan.atom
    if max(m*n,m*k,k*n) >= 1<<31:
        raise ValueError("per-CTA addressing requires signed 32-bit offsets")
    pitch = (m+n)*32
    scratch = max(pitch*2,8*atom*4)
    source = [Path(__file__).with_suffix(".cuh").read_text(), _mma(atom,1), _mma(atom,2),
              'extern "C" __global__ __launch_bounds__(256) void replay('
              'const uint16_t* A,const uint16_t* B,float* C,float* R) {',
              f"A+=(size_t)blockIdx.x*{m*k}; B+=(size_t)blockIdx.x*{n*k}; C+=(size_t)blockIdx.x*{m*n};",
              "extern __shared__ __align__(1024) unsigned char sm[];",
              f"float* partial=(float*)(sm+{scratch}); int tid=threadIdx.x,wg=tid/128;"]
    if plan.reduced_writers is not None:
        source.append(f"R+=(size_t)blockIdx.x*{n};")
    width = _residency(plan, k, staged_k_tiles, sliced, resident_atoms, grid_ctas, sm_count)
    for group in _atom_groups(plan, width):
        rows, cols = _copy_regions(group, atom)
        source += ["{", f"float acc[{len(group)}][{atom//2}]={{}};"]
        # The odd final K16 step gets a single-buffer tail in all four arms.
        steps = ((1,0,k),) if staged_k_tiles==1 else ((2,0,k//32*32),(1,k//32*32,k))
        for depth, start, stop in steps:
            if start == stop:
                continue
            source += ["#pragma unroll 1", f"for(int kt={start}; kt<{stop}; kt+={16*depth}) {{",
                       "__syncthreads();"]
            for step in range(depth):
                for lo,hi in rows:
                    source.append(f"copy_a<{m},{k},{lo},{hi}>(sm+{step*pitch},A,kt+{step*16});")
                for lo,hi in cols:
                    source.append(f"copy_b<{n},{lo},{hi}>(sm+{step*pitch+m*32},B,kt+{step*16});")
            # Every lane completes its own copies before the CTA publishes the slots.
            source += ['asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");',
                       "__syncthreads();", 'asm volatile("fence.proxy.async.shared::cta;" ::: "memory");']
            for i,(r,c,s,d) in enumerate(group):
                source.append(f"{{int row={r}+wg*({s-r}), col={c}+wg*({d-c});")
                args = []
                for j in range(depth):
                    source += [f"auto a{j}=descriptor(sm+{j*pitch}+row*16,{(m<<16)|(8<<32)}ull);",
                               f"auto b{j}=descriptor(sm+{j*pitch+m*32}+col*16,{(16<<16)|(n<<32)|(3<<62)}ull);"]
                    args.extend([f"a{j}",f"b{j}"])
                if depth==2 and wait_group==2:
                    source.append(f"issue2(acc[{i}],"+",".join(args)+");")
                else:
                    for j in range(depth):
                        source.append(f"issue1(acc[{i}],a{j},b{j});")
                source.append("}")
            source.append("}")
        source.append("__syncthreads();")  # operand slots may now serve as reduction scratch
        for i,(r,c,s,d) in enumerate(group):
            fan = "true" if plan.reduced_writers is not None else "false"
            source.append(f"deliver_atom<{n},{atom},{fan}>(acc[{i}],C,{r}+wg*({s-r}),"
                          f"{c}+wg*({d-c}),(float*)sm,partial);")
        source.append("}")
    return "\n".join(source+_write_columns(plan)+["}",""])


def compile_accumulator_gemm(layout_ir, k, *, sliced=None, resident_atoms=None, wait_group=1, staged_k_tiles=2, grid_ctas=None, sm_count=114, nvcc="nvcc"):
    """Return a CUDA driver-loadable cubin and its launch contract."""
    plan = read_layouts(layout_ir)
    ranked = rank_residencies(plan, k, staged_k_tiles=staged_k_tiles, grid_ctas=grid_ctas, sm_count=sm_count)
    width = (ranked[0]["resident_atoms"] if sliced is None and resident_atoms is None else
             _residency(plan, k, staged_k_tiles, sliced, resident_atoms, grid_ctas, sm_count))
    source = emit(plan, k, resident_atoms=width, wait_group=wait_group, staged_k_tiles=staged_k_tiles)
    with tempfile.TemporaryDirectory(prefix="frisk-accumulator-") as tmp:
        path = Path(tmp)
        (path/"kernel.cu").write_text(source)
        result = subprocess.run([nvcc,"-O3","-std=c++17","-arch=sm_90a","--cubin","-Xptxas=-v",
                                 "kernel.cu","-o","kernel.cubin"],cwd=path,capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        image = (path/"kernel.cubin").read_bytes()
    return image, {"problem_shape":(plan.m,plan.n,k),"column_sum":plan.reduced_writers is not None,
                   "symbol":"replay","threads":256,"dynamic_shared_bytes":shared_bytes(plan),
                   "resident_atoms":width,"slices":(len(plan.origins)+width-1)//width,
                   "ranked_residencies":ranked,"ptxas":result.stderr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solved_layouts",type=Path)
    parser.add_argument("--k",type=int,required=True)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--resident", choices=("auto", "full", "slice"), default="auto")
    choice.add_argument("--resident-atoms", type=int)
    parser.add_argument("--grid-ctas", type=int, help="known number of independent output tiles")
    parser.add_argument("--sm-count", type=int, default=114, help="available SMs (default H100 PCIe)")
    parser.add_argument("--explain", type=Path, help="write the pre-compile ranking as JSON")
    parser.add_argument("--wait-group",type=int,choices=(1,2),default=1)
    parser.add_argument("--staged-k-tiles",type=int,choices=(1,2),default=2)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    ir = args.solved_layouts.read_text()
    sliced = {"auto":None, "slice":True, "full":False}[args.resident]
    options = dict(sliced=sliced, resident_atoms=args.resident_atoms, wait_group=args.wait_group,
                   staged_k_tiles=args.staged_k_tiles, grid_ctas=args.grid_ctas, sm_count=args.sm_count)
    if args.output.suffix == ".cubin":
        image,info = compile_accumulator_gemm(ir,args.k,**options)
        args.output.write_bytes(image)
        smem = info["dynamic_shared_bytes"]
        width = info["resident_atoms"]
        ranked = info["ranked_residencies"]
    else:
        layouts = read_layouts(ir)
        args.output.write_text(emit(layouts,args.k,**options))
        smem = shared_bytes(layouts)
        width = _residency(layouts,args.k,args.staged_k_tiles,sliced,args.resident_atoms,args.grid_ctas,args.sm_count)
        ranked = rank_residencies(layouts,args.k,staged_k_tiles=args.staged_k_tiles, grid_ctas=args.grid_ctas, sm_count=args.sm_count)
    if args.explain:
        args.explain.write_text(json.dumps(dict(selected_atoms=width,candidates=ranked),indent=2)+"\n")
    print(f"threads=256 dynamic_shared_bytes={smem} resident_atoms={width}")
