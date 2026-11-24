"""SM90 CUDA replay of a solved, zero-initialized SS GEMM with optional column sum.

This entry takes *layouts* from frisk-opt and an explicit dense GEMM problem.
It does not interpret arbitrary surrounding MLIR. Each CTA owns an independent
row-major f16 A/B (16-byte aligned), f32 C and optional f32 R. Launch one block
per independent problem with 256 threads and the returned dynamic shared byte
count. By default both residency modes stage adjacent K16 steps in distinct slots; the
wait policy controls how many WGMMA instructions precede the drain. A sliced
kernel completes all K steps and delivers each original atom before the next.
Use --output kernel.cu to inspect CUDA, or --output kernel.cubin to compile.
"""
import argparse
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


def emit(plan, k, *, sliced=True, wait_group=1, staged_k_tiles=2):
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
    groups = [(origin,) for origin in plan.origins] if sliced else [plan.origins]
    for group in groups:
        source += ["{", f"float acc[{len(group)}][{atom//2}]={{}};"]
        # The odd final K16 step gets a single-buffer tail in all four arms.
        steps = ((1,0,k),) if staged_k_tiles==1 else ((2,0,k//32*32),(1,k//32*32,k))
        for depth, start, stop in steps:
            if start == stop:
                continue
            source += ["#pragma unroll 1", f"for(int kt={start}; kt<{stop}; kt+={16*depth}) {{",
                       "__syncthreads();"]
            for step in range(depth):
                for lo,hi in _intervals([v for r,c,s,d in group for v in (r,s)],64):
                    source.append(f"copy_a<{m},{k},{lo},{hi}>(sm+{step*pitch},A,kt+{step*16});")
                for lo,hi in _intervals([v for r,c,s,d in group for v in (c,d)],atom):
                    source.append(f"copy_b<{n},{lo},{hi}>(sm+{step*pitch+m*32},B,kt+{step*16});")
            source += ["__syncthreads();", 'asm volatile("fence.proxy.async.shared::cta;" ::: "memory");']
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


def compile_accumulator_gemm(layout_ir, k, *, sliced=True, wait_group=1, staged_k_tiles=2, nvcc="nvcc"):
    """Return a CUDA driver-loadable cubin and its launch contract."""
    plan = read_layouts(layout_ir)
    source = emit(plan,k,sliced=sliced,wait_group=wait_group,staged_k_tiles=staged_k_tiles)
    with tempfile.TemporaryDirectory(prefix="frisk-accumulator-") as tmp:
        path = Path(tmp)
        (path/"kernel.cu").write_text(source)
        result = subprocess.run([nvcc,"-O3","-std=c++17","-arch=sm_90a","--cubin","-Xptxas=-v",
                                 "kernel.cu","-o","kernel.cubin"],cwd=path,capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        image = (path/"kernel.cubin").read_bytes()
    return image, {"symbol":"replay","threads":256,"dynamic_shared_bytes":shared_bytes(plan),
                   "slices":len(plan.origins) if sliced else 1,"ptxas":result.stderr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solved_layouts",type=Path)
    parser.add_argument("--k",type=int,required=True)
    parser.add_argument("--resident",choices=("full","slice"),default="slice")
    parser.add_argument("--wait-group",type=int,choices=(1,2),default=1)
    parser.add_argument("--staged-k-tiles",type=int,choices=(1,2),default=2)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    ir = args.solved_layouts.read_text()
    options = dict(sliced=args.resident=="slice",wait_group=args.wait_group,staged_k_tiles=args.staged_k_tiles)
    if args.output.suffix == ".cubin":
        image,info = compile_accumulator_gemm(ir,args.k,**options)
        args.output.write_bytes(image)
        smem = info["dynamic_shared_bytes"]
    else:
        layouts = read_layouts(ir)
        args.output.write_text(emit(layouts,args.k,**options))
        smem = shared_bytes(layouts)
    print(f"threads=256 dynamic_shared_bytes={smem}")
