"""Build the paid queues of checked, staged Attention images."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns

from attention_layout_search import K4, MODES, digest, select_attention
from build_tiled_attention import SOURCE, TILE_IR, _sha, _toolchain, _write, compile_emitted_image
from tiled_attention import checked_geometry


def _spec(shape):
    batch, heads, length = shape
    return dict(B=batch, H=heads, N=length, D=64, q_block=64, kv_block=64)


def _case(spec):
    return 'attention-b{B}-h{H}-n{N}-d{D}'.format(**spec)


def _sources():
    filenames = sorted(SOURCE.glob('attention_*.py')) + [
        SOURCE / name for name in ('access_plan.py', 'backend.py', 'valid_domain.py',
                                   'native_static_support.py', 'async_host.py',
                                   'resource_prediction.py', 'tiled_attention.py',
                                   'build_tiled_attention.py', 'build_attention_layout_search.py')]
    return {str(path): _sha(path) for path in filenames}


def _prepare(opt, out):
    selected = out / 'selected.mlir'
    packet = out / 'execution.json'
    start = perf_counter_ns()
    source_hash = _sha(TILE_IR)
    commands = ([str(opt), str(TILE_IR), '--frisk-layout-pipeline', '-verify-each',
                 '-o', str(selected)],
                [str(opt), str(selected), f'--frisk-export-execution=output={packet}',
                 '-o', '/dev/null'])
    traces = [subprocess.run(cmd, capture_output=True, text=True, check=True).stderr
              for cmd in commands]
    if source_hash != _sha(TILE_IR):
        raise ValueError('Attention source changed during layout inference')
    paths = (TILE_IR, selected, packet)
    return json.loads(packet.read_bytes()), dict(wall_ns=perf_counter_ns()-start,
        commands=commands, diagnostics=traces, paths={str(path): _sha(path) for path in paths})


def _compile(triton, out, case, spec, item):
    # The caller charges the slot whether compilation succeeds or fails.
    invoked = False

    class PaidCompiler:
        def compile(self, *args, **kwargs):
            nonlocal invoked
            invoked = True
            return triton.compile(*args, **kwargs)

    try:
        built = compile_emitted_image(PaidCompiler(), out, case, spec, item.ttgir, item.proof)
    except Exception as error:
        return dict(status='compile-failed', physical_compile_invoked=invoked,
                    reason=f'{type(error).__name__}: {error}')
    return dict(built, status='compiled', physical_compile_invoked=invoked)


def build_search(output, frisk_opt, shapes, *, mode):
    modes = MODES if mode == 'all' else (mode,)
    specs = tuple(map(_spec, shapes))
    if not specs or len({_case(spec) for spec in specs}) != len(specs):
        raise ValueError('each checked Attention shape must be supplied once')
    for spec in specs:
        checked_geometry(spec)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Attention image builds must be CPU-only')
    start = perf_counter_ns()
    output = Path(output).resolve()
    output.mkdir()
    os.environ['TRITON_ALWAYS_COMPILE'] = '1'
    os.environ['TRITON_CACHE_DIR'] = str(output/'cache')
    sources = _sources()
    triton, tools = _toolchain()
    opt = Path(frisk_opt).resolve(strict=True)
    tools[str(opt)] = _sha(opt)
    packet, prepared = _prepare(opt, output)
    problem_sha = digest(list(prepared['paths'].values()))
    source_commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'],
                                            text=True).strip()
    registration = output/'registration.json'
    _write(registration, dict(specs=specs, modes=modes, native_budget=K4, tool_hashes=tools,
        prepared=prepared, problem_sha256=problem_sha, source_commit=source_commit, sources=sources,
        numeric='fixed global max, ordered FP32 sum/div, RN FP16 P, ascending PV K16, FP32 output'))
    compiled = {}
    pending = []
    searches = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for spec in specs:
            case = _case(spec)
            selected, audit = select_attention(spec, packet, problem_sha256=problem_sha)
            searches[case] = audit
            case_path = output/case
            case_path.mkdir()
            _write(case_path/'search.json', audit)
            for engine in modes:
                for slot, item in enumerate(selected[engine]):
                    row = dict(case=case, spec=spec, engine=engine, slot=slot,
                               logical_slot_charged=True, native_compiler_invoked=False)
                    if item is None:
                        schedule = audit['legacy_schedule_order'][slot]
                        row.update(status='rejected', schedule=schedule,
                                   reason='paid legacy schedule outside checked domain',
                                   execution_kv_block=next(p['tuning']['kv_width'] for p in audit['attempts']
                                       if p['layout'] == 'legacy' and p['schedule'] == schedule))
                        pending.append((row, None, None))
                        continue
                    identity = item.proof['ttgir_sha256']
                    plan = dict(layout=item.layout, schedule=item.schedule, tuning=item.tuning,
                                features=item.features, checked_ttgir_sha256=item.proof['ttgir_sha256'])
                    plan_path = case_path/f'{engine}-c{slot}-plan.json'
                    _write(plan_path, plan)
                    proof_path = case_path/f'{engine}-c{slot}-checked.json'
                    _write(proof_path, item.proof)
                    row.update(candidate_proof=str(proof_path), candidate_proof_sha256=_sha(proof_path))
                    row.update(execution_kv_block=item.proof['execution_kv_width'],
                               tuning=item.tuning, layout=item.layout, schedule=item.schedule,
                               plan=str(plan_path), plan_sha256=_sha(plan_path), ttgir_sha256=identity)
                    key = (case, identity)
                    if key not in compiled:
                        dest = case_path/f'{engine}-c{slot}'
                        compiled[key] = (engine, slot, pool.submit(_compile, triton, dest, case, spec, item))
                    origin_engine, origin_slot, future = compiled[key]
                    pending.append((row, future, (origin_engine, origin_slot)))
            print(case, audit['local_layout'], {e: [(p.layout, p.schedule) if p else None
                  for p in selected[e]] for e in modes}, flush=True)
        rows = []
        for row, future, origin in pending:
            if future is not None:
                built = future.result()
                if built['status'] == 'compiled' and built['ttgir_sha256'] != row['ttgir_sha256']:
                    raise RuntimeError('compiled TTGIR does not bind the paid candidate')
                row.update(built)
                is_origin = (row['engine'], row['slot']) == origin
                row['native_compiler_invoked'] = row.pop('physical_compile_invoked') and is_origin
                if not is_origin:
                    row['compile_alias_of'] = dict(engine=origin[0], slot=origin[1])
                if row['status'] == 'compiled':
                    row['charged_slot'] = row['slot']
                    # The candidate-specific proof is the sealed plan, even
                    # when another paid slot supplied the native image.
                    row['proof'] = row['candidate_proof']
                    row['proof_sha256'] = row['candidate_proof_sha256']
            _write(output/row['case']/f"{row['engine']}-c{row['slot']}-receipt.json", row)
            rows.append(row)
    if sources != _sources():
        raise ValueError('Attention sources changed while creating native images')
    result = output/'programs.json'
    _write(result, dict(cases=rows, fixed_choices={_case(spec): {engine: 0 for engine in modes}
         for spec in specs}, registration_sha256=_sha(registration), native_budget_per_mode=K4,
         common_preview_attempts={case: audit['common_preview_attempts'] for case, audit in searches.items()},
         native_compiler_invocations=sum(row['native_compiler_invoked'] for row in rows),
         charged_slots=len(rows), gpu_launches=0, complete_wall_ns=perf_counter_ns()-start))
    print(result, flush=True)
    return result
