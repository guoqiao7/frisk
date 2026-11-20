"""Known-program K4 backend validation using the existing GPU execution engine.

This is a development regression, not a fresh quality qualification. No copied
numerical/sanitizer loop, production cache-identity relaxation or old-root write.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import time
import partial_cost_qualification as engine
import network_selection_fixture as numerical
import selection_retention_archive as archive
import selection_refinement_decision as adapter

STAGE = 'selection-refinement-known-backend-validation'


@contextmanager
def epoch():
    saved = engine.suite, engine.BASE
    try:
        engine.suite = SimpleNamespace(STAGE=STAGE)
        engine.BASE = adapter.BASE
        yield
    finally:
        engine.suite, engine.BASE = saved


def _case_mode(case):
    mode = case.get('mode', 'J')
    if mode not in ('J', 'LI'):
        raise ValueError('matrix mode must be J or LI')
    return mode


def _checked_archive(case):
    mode = _case_mode(case); search = Path(case['search'])
    record = adapter.read(search / 'record.json')
    pool = adapter.read(search / 'candidate-archive.json')
    if record.get('selection') != mode:
        raise ValueError('matrix/source archive selection mode differs')
    requested = (archive.ROUTING_POLICY if 'retention_resolution' in pool.get('search', {})
                 else archive.POLICY)
    archive.check_archive(record, pool, mode, 4, expected_policy=requested)
    return record, pool, mode


def _arm(mode, entry):
    return ('joint-c' if mode == 'J' else 'local-c') + str(entry['index'])


def init(run, matrix):
    from cost_calibration import source_identity
    matrix = Path(matrix).resolve()
    value = adapter.read(matrix)
    if value.get('schema_version') != 1 or not isinstance(value.get('cases'), list) or not value['cases']:
        raise ValueError('complete known K4 matrix required')
    seal = adapter.SealedInputs()
    immutable = {str(matrix): adapter.sha(matrix), str(seal.path): adapter.KNOWN_INDEX_SHA256}
    known_run = Path('/data0/baopeihua-frisk-network-20261105/run-network-selection-r3')
    known_manifest = seal.read(known_run / 'manifest.json')
    immutable[str(known_run / 'manifest.json')] = adapter.sha(known_run / 'manifest.json')
    known_pairs = {(numerical.digest(row['spec']), row['prepared']) for row in known_manifest['rows'] if row['model'] == 'surviving'}
    cases = []; profile = None; seen = set()
    for case in value['cases']:
        if set(case) not in ({'spec', 'prepared', 'search'}, {'spec', 'prepared', 'search', 'mode'}):
            raise ValueError('matrix case schema differs')
        spec = case['spec']; search = Path(case['search']).resolve(); mode = _case_mode(case)
        if (spec['key'], mode) in seen or spec['threads'] != 128 or spec['dtype'] != 'float32' or spec['case_id'] == 'attention':
            raise ValueError('unique scalar 128-thread known programs required')
        seen.add((spec['key'], mode))
        prepared = Path(case['prepared']).resolve()
        if (numerical.digest(spec), str(prepared)) not in known_pairs:
            raise ValueError('matrix is not a known sealed program/preparation pair')
        meta = seal.read(prepared / 'prepared.json')
        for name in ('prepared.json', 'input.mlir', 'domains.json'):
            immutable[str(prepared / name)] = seal.check(prepared / name)
        record, pool, mode = _checked_archive(dict(case, search=str(search)))
        if record['prepared_digest'] != meta['package_digest']:
            raise ValueError('archive/preparation identity differs')
        for name in ('record.json', 'candidate-archive.json'):
            immutable[str(search / name)] = adapter.sha(search / name)
        static = adapter.KNOWN_INDEX.parent / 'static-network-proxy-profile.json'
        seal.check(static); immutable[str(static)] = adapter.sha(static)
        if record['cost_profile'] != adapter.read(static):
            raise ValueError('registered static profile differs')
        if profile is not None and profile != str(static):
            raise ValueError('common static profile required')
        profile = str(static)
        development_spec = dict(spec, split='development', role='known-development-regression')
        cases.append(dict(spec=development_spec, original_spec=spec, prepared=str(prepared), search=str(search), mode=mode))
    run = adapter._fresh_output(run)
    protocol = dict(schema_version=1, stage=STAGE, role='known-development-regression-not-fresh',
        specifications=[c['spec'] for c in cases], matrix=cases,
        configuration=dict(static=dict(profile=profile)),
        safety=dict(tools=['memcheck', 'racecheck', 'synccheck', 'initcheck']),
        native_budget=dict(capacity_per_pool=4, pools=len(cases), requests=sum(len(adapter.read(Path(c['search'])/'candidate-archive.json')['entries']) for c in cases)),
        compilation_execution=dict(contract='fixed-ttgir-four-worker-batch-v1', cpus=[8,9,10,11], compiler_input_reuse=True),
        qualification_pass_claim=False)
    engine.write(run / 'protocol.json', protocol)
    tools = engine.closure.frozen_tools(engine.OPT)
    for path in (engine.BENCH, Path('/usr/local/cuda/bin/nvdisasm')):
        tools['hashes'][str(path.resolve())] = adapter.sha(path)
    registration = dict(created_ns=time.time_ns(), source=source_identity(), tools=tools,
        optimizer=str(engine.OPT), protocol_sha256=adapter.sha(run / 'protocol.json'),
        protocol_digest=numerical.digest(protocol), immutable_inputs=immutable)
    engine.write(run / 'registration.json', registration)
    engine.write(run / 'manifest.json', dict(status='registered', registration_sha256=adapter.sha(run / 'registration.json'), rows=[]))
    return run


def generate(run):
    run = Path(run)
    with epoch():
        protocol = engine.verify(run); manifest = engine.read(run / 'manifest.json')
        if manifest['status'] != 'registered':
            raise ValueError('fresh registered matrix required')
        manifest['status'] = 'generating'; engine.write(run / 'manifest.json', manifest)
        (run / 'generation-logs').mkdir()
        for case in protocol['matrix']:
            spec = case['spec']; prepared = Path(case['prepared']); meta = engine.read(prepared / 'prepared.json')
            record, pool, mode = _checked_archive(case)
            for entry in pool['entries']:
                arm = _arm(mode, entry); assignment = run / (spec['key'] + '-' + arm + '-assignment.json')
                engine.write(assignment, entry)
                row = engine.add_case(run, manifest, spec, arm, 'surviving', prepared, meta,
                    run / 'cases' / (spec['key'] + '-' + arm), assignment=assignment, entry=entry,
                    replay_arguments=archive.replay_args(entry), replay_validator=archive.check_replay)
                # The shared engine intentionally performs every fixed-assignment
                # replay as J. Preserve its selection field and bind the pool's
                # actual search mode separately to the registered inventory.
                row['source_archive_mode'] = mode
                engine.write(run / 'manifest.json', manifest)
        expected = protocol['native_budget']['requests']
        if len(manifest['rows']) != expected:
            raise ValueError('native request budget differs')
        manifest['status'] = 'generated'; engine.write(run / 'manifest.json', manifest)
        engine.verify(run)
    return manifest


def expected_inventory(run, protocol):
    expected = {}
    for case in protocol['matrix']:
        prepared = Path(case['prepared']); meta = engine.read(prepared / 'prepared.json')
        search_record, pool, mode = _checked_archive(case)
        for entry in pool['entries']:
            arm = _arm(mode, entry); key = numerical.digest(case['spec']), arm
            if key in expected:
                raise ValueError('duplicate registered K4 identity')
            expected[key] = dict(spec=case['spec'], arm=arm, entry=entry, prepared=str(prepared),
                source_archive_mode=mode, source_mode_metadata_required='mode' in case,
                path=str(Path(run) / 'cases' / (case['spec']['key'] + '-' + arm)),
                prepared_sha256=adapter.sha(prepared / 'prepared.json'),
                domain={name: meta[name] for name in ('common_digest', 'domain_digest', 'coverage_digest')})
    return expected


def check_inventory_rows(rows, expected):
    actual = [(numerical.digest(row['spec']), row['arm']) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError('manifest differs from registered complete K4 inventory')
    for row, key in zip(rows, actual):
        witness = expected[key]
        for name in ('spec', 'arm', 'prepared', 'path', 'prepared_sha256', 'domain'):
            if row.get(name) != witness[name]:
                raise ValueError('manifest registered ' + name + ' differs')
        source_mode = witness.get('source_archive_mode', 'J')
        legacy_default = 'J' if not witness.get('source_mode_metadata_required', False) else None
        if row.get('source_archive_mode', legacy_default) != source_mode:
            raise ValueError('manifest registered source archive mode differs')
        entry = witness['entry']
        if (row.get('archive_entry') != entry or row.get('assignment_sha256') != entry['complete_assignment_sha256']
                or row.get('complete_program_sha256') != entry['complete_program_sha256']
                or row.get('cost') != entry['cost'] or row.get('model_id') != entry['cost_model']
                or row.get('selection') != 'J' or row.get('conversion') != 'C'
                or row.get('model') != 'surviving' or row.get('scope') != 'natural'
                or row.get('status') not in ('generated', 'validated')):
            raise ValueError('manifest complete-program witness differs')
    return True


def verify_inventory(run, *, check_artifacts=True, sanitizer=False):
    run = Path(run)
    with epoch():
        if sanitizer:
            manifest, registration = engine.closure.verify_run(run, sanitizer=True)
            protocol = engine.read(run / 'protocol.json')
            if protocol.get('stage') != STAGE or numerical.digest(protocol) != registration.get('protocol_digest'):
                raise ValueError('sanitizer registered protocol differs')
        else:
            protocol = engine.verify(run); manifest = engine.read(run / 'manifest.json')
        expected = expected_inventory(run, protocol)
        if len(expected) != protocol['native_budget']['requests']:
            raise ValueError('registered native request count differs')
        check_inventory_rows(manifest['rows'], expected)
        for row in manifest['rows']:
            directory = Path(row['path'])
            generation = directory / 'generation-inputs.json'
            if adapter.sha(generation) != row['generation_inputs_sha256']:
                raise ValueError('row generation input identity differs')
            if check_artifacts:
                engine.closure.verify_files(directory, engine.read(generation))
                record = engine.read(directory / 'record.json')
                archive.check_replay(record, row['archive_entry'])
                configuration = engine.read(directory / 'configuration.json')
                if configuration != dict(spec=row['spec'], scope='natural', arm=row['arm'], model='surviving', entry='frisk_execution', grid=[1,1,1], threads=128):
                    raise ValueError('registered numerical/execution configuration differs')
                if row['status'] == 'validated':
                    engine.closure.verify_accepted_row(directory, row, sanitizer=sanitizer)
        return protocol


def compile_batch(run):
    """Reuse the fixed four-worker compiler; request budget remains the K4 sum."""
    from parallel_compile_batch import compile_many_parallel
    run = Path(run)
    with epoch():
        protocol = verify_inventory(run); manifest = engine.read(run / 'manifest.json')
        registration = engine.read(run / 'registration.json')
        if manifest['status'] != 'generated' or manifest.get('compiled'):
            raise ValueError('fresh generated matrix required')
        output = run / 'parallel-native-compilation'
        if output.exists() or (run / 'batch-compilation.json').exists():
            raise FileExistsError('native compilation refuses previous attempt')
        start = time.perf_counter_ns()
        result = compile_many_parallel([Path(row['path']) for row in manifest['rows']],
            run / 'native-artifact-cache', output=output,
            execution_modules=Path(__file__).resolve().parent, cpus=(8, 9, 10, 11),
            compiler_input_reuse=True, source=registration['source'], tools=registration['tools'])
        count = protocol['native_budget']['requests']
        if not (result.get('status') == 'complete' and result.get('attempted_requests') == count
                and result.get('request_cap') == count and result.get('succeeded_requests') == count
                and result.get('failed_requests') == 0 and result.get('not_attempted_requests') == 0
                and result.get('descendants_reaped') is True):
            engine.write(run / 'batch-compilation.json', dict(status='failed', result=result, wall_ns=time.perf_counter_ns()-start))
            raise ValueError('complete native request budget not satisfied')
        engine.write(run / 'batch-compilation.json', dict(status='complete', result=result, wall_ns=time.perf_counter_ns()-start))
        manifest['compiled'] = True
        manifest['batch_compilation_sha256'] = adapter.sha(run / 'batch-compilation.json')
        engine.write(run / 'manifest.json', manifest)
        engine.verify(run)
        return result


def execute(directory, load_only=False, safety=False, safety_tool=None):
    from native_compile_cache import verify_precompiled
    verify_inventory(Path(directory).parents[1], check_artifacts=False, sanitizer=safety)
    return engine.closure.execute(directory, load_only=load_only, safety=safety,
        safety_tool=safety_tool, numerical_fixture=numerical, compiler_provider=verify_precompiled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'generate', 'compile', 'validate', 'execute', 'verify'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--matrix', type=Path)
    parser.add_argument('--load-only', action='store_true')
    parser.add_argument('--safety', action='store_true')
    parser.add_argument('--safety-tool')
    args = parser.parse_args(); directory = args.directory.resolve()
    if args.command == 'init':
        init(directory, args.matrix)
    elif args.command == 'generate':
        generate(directory)
    elif args.command == 'compile':
        compile_batch(directory)
    elif args.command == 'validate':
        verify_inventory(directory)
        with epoch(): engine.quality.validate(directory, worker_script=__file__)
        verify_inventory(directory)
    elif args.command == 'execute':
        execute(directory, args.load_only, args.safety, args.safety_tool)
    else:
        verify_inventory(directory)


if __name__ == '__main__':
    main()
