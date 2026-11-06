
"""Build a fixed one-slot D64 aligned/tail qualification on the pinned SM90 path."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns

from async_host import physical_abi
from resource_prediction import compiler_exact_resources
from tiled_attention import emit_tiled_attention


PTXAS_SHA256 = 'c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca'
LIBTRITON_SHA256 = '401cc8a47b481c4b74eb55ddba8ce03ba1cc92db0652af541e7c54ee1d17a995'
TARGET = dict(backend='cuda', arch=90, warp_size=32)
OPTIONS = dict(num_warps=4, num_ctas=1, enable_fp_fusion=False)
SOURCE = Path(__file__).resolve().parent
TILE_IR = SOURCE.parents[2] / 'test/Transforms/attention-layout-tile-d64.mlir'


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _write(path, payload):
    with path.open('x') as out:
        json.dump(payload, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


def _toolchain():
    import triton
    from triton._C import libtriton
    from triton.backends.nvidia.compiler import get_ptxas

    library = Path(libtriton.__file__).resolve()
    assembler = Path(get_ptxas().path).resolve()
    hashes = {str(library): _sha(library), str(assembler): _sha(assembler)}
    if (triton.__version__, hashes[str(library)], hashes[str(assembler)]) != (
            '3.4.0', LIBTRITON_SHA256, PTXAS_SHA256):
        raise ValueError('tiled Attention requires the pinned Triton and ptxas')
    return triton, hashes


def _build_case(triton, out, batch, heads, length, packet):
    from triton.backends.compiler import GPUTarget

    case = f'attention-b{batch}-h{heads}-n{length}-d64'
    spec = dict(B=batch, H=heads, N=length, D=64, q_block=64, kv_block=64)
    dest = out / case
    dest.mkdir()
    start = perf_counter_ns()
    ttgir, proof = emit_tiled_attention(spec, tile_packet=packet)
    emission_ns = perf_counter_ns() - start
    input_path = dest / 'kernel.ttgir'
    input_path.write_text(ttgir)
    proof_path = dest / 'proof.json'
    _write(proof_path, proof)
    start = perf_counter_ns()
    result = triton.compile(str(input_path), target=GPUTarget(**TARGET), options=OPTIONS)
    compile_ns = perf_counter_ns() - start
    meta = result.metadata
    if (meta.name != proof['entry'] or meta.shared != 0 or meta.global_scratch_size != 0
            or meta.num_warps != 4 or meta.num_ctas != 1 or tuple(meta.cluster_dims) != (1, 1, 1)):
        raise ValueError('native artifact changed its checked launch contract')
    ptx = result.asm['ptx']
    slots = physical_abi(ptx, 0)
    if ptx.count(slots[-1]) != 1:
        raise ValueError('the physical scratch pointer must be unused')
    paths = {}
    for field, payload in (('ptx', ptx.encode()), ('cubin', result.asm['cubin'])):
        path = dest / f'kernel.{field}'
        path.write_bytes(payload)
        paths.update({field: str(path), field+'_sha256': _sha(path)})
    exact = compiler_exact_resources(paths['cubin'], tool_path='/usr/local/cuda-12.8/bin/cuobjdump')
    if exact['resources']['raw_shared_bytes'] not in (proof['static_shared_bytes'], proof['static_shared_bytes'] + 1024):
        raise ValueError('native artifact changed its checked static Shared allocation')
    _write(dest / 'compiler-resources.json', exact)
    image = {k: proof[k] for k in ('entry', 'dynamic_shared_bytes', 'grid', 'block', 'alignment_bytes')}
    return dict(case=case, spec=spec, image=dict(image, **paths), charged_slot=0,
                ttgir=str(input_path), ttgir_sha256=_sha(input_path), proof=str(proof_path),
                proof_sha256=_sha(proof_path), physical_params=slots,
                native_resources=exact['resources'], emission_wall_ns=emission_ns,
                native_wall_ns=compile_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frisk-opt', required=True, type=Path)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--heads', type=int, default=8)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('native build must run CPU-only with CUDA_VISIBLE_DEVICES empty')
    out = args.output.resolve()
    out.mkdir()
    os.environ['TRITON_ALWAYS_COMPILE'] = '1'
    os.environ['TRITON_CACHE_DIR'] = str(out / 'native-cache')
    start = perf_counter_ns()
    triton, tools = _toolchain()
    opt = args.frisk_opt.resolve(strict=True)
    selected, packet_path = out/'tile-selected.mlir', out/'tile-execution.json'
    tools[str(opt)] = _sha(opt)
    inference = [str(opt), str(TILE_IR), '--frisk-layout-pipeline', '-verify-each', '-o', str(selected)]
    export = [str(opt), str(selected), f'--frisk-export-execution=output={packet_path}', '-o', '/dev/null']
    for command, phase in ((inference, 'tile-inference'), (export, 'tile-export')):
        phase_start = perf_counter_ns()
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        _write(out/f'{phase}.json', dict(command=command, stderr=result.stderr,
                                       wall_ns=perf_counter_ns()-phase_start))
    packet = json.loads(packet_path.read_bytes())
    witness = dict(source=str(TILE_IR), source_sha256=_sha(TILE_IR),
                   selected=str(selected), selected_sha256=_sha(selected),
                   execution=str(packet_path), execution_sha256=_sha(packet_path))
    sources = {name: _sha(SOURCE / name) for name in
               ('access_plan.py', 'attention_tile_mma.py', 'backend.py', 'build_tiled_attention.py',
                'native_static_support.py', 'tiled_attention.py', 'valid_domain.py')}
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    _write(out / 'registration.json', dict(source_commit=commit, sources=sources, tool_hashes=tools,
        target=TARGET, options=OPTIONS, B=args.batch, H=args.heads, N=[512, 513], D=64,
        slots_per_shape=1, compiler_invocations_per_shape=1, tile_witness=witness,
        numeric='global max, fixed ordered f32 sum/div, RN f16 P before ascending K16 PV'))
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(_build_case, triton, out, args.batch, args.heads, n, packet) for n in (512, 513)]
        rows = [task.result() for task in futures]
    if sources != {name: _sha(SOURCE / name) for name in sources}:
        raise ValueError('checked Attention source changed during compilation')
    _write(out / 'programs.json', dict(cases=rows, registration_sha256=_sha(out / 'registration.json'),
                                      tile_witness=witness, native_compiler_invocations=len(rows), gpu_launches=0,
                                      complete_wall_ns=perf_counter_ns()-start))
    print(out / 'programs.json')


if __name__ == '__main__':
    main()
