"""Timing-free joint model adapter and sealed, known-program decision replay.

The runtime adapter accepts only complete validated native candidate pools. The
CLI is an evidence replay of already observed R3 programs, not a qualification
run or a compiler-cache compatibility bypass. It never reads performance labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import joint_cost_model as model

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'guoqiao/optimize_plan/selection_refinement_artifacts'
DATA = Path('/data0/baopeihua-frisk-selection-refinement-20261105')
KNOWN_INDEX = ROOT / 'guoqiao/optimize_plan/network_selection_execution_artifacts/evidence-index.json'
KNOWN_INDEX_SHA256 = 'a666ac9cc5a987eebc44d431154dd1c0e46528966a5c2e0e8e7c131beefbc423'
CONTRACT = 'selection-refinement-known-decision-replay-v1'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


class SealedInputs:
    """Verify each consumed file against the original cross-filesystem seal."""
    def __init__(self, path=KNOWN_INDEX, expected=KNOWN_INDEX_SHA256):
        self.path = Path(path).resolve()
        if sha(self.path) != expected:
            raise ValueError('sealed index hash differs')
        index = read(self.path)
        self.files = {}
        roots = {}
        for key, value in index['roots'].items():
            root = Path(value['real_absolute_path'])
            if root.is_symlink() or root.resolve() != root:
                raise ValueError('sealed root changed through a symlink')
            actual = root.stat()
            identity = value['identity']
            for field, got in [('device', actual.st_dev), ('inode', actual.st_ino), ('uid', actual.st_uid)]:
                if identity[field] != got:
                    raise ValueError('sealed root identity differs: ' + field)
            roots[key] = root
        for item in index['files']:
            relative = Path(item['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe indexed relative path')
            target = roots[item['root_id']] / relative
            if target in self.files:
                raise ValueError('duplicate sealed file')
            self.files[target] = item
        self.checked = {}

    def check(self, path):
        path = Path(path)
        if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
            raise ValueError('absolute unaliased sealed input required')
        item = self.files.get(path)
        if item is None:
            raise ValueError('input is outside the sealed inventory: ' + str(path))
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('sealed input changed: ' + str(path))
        self.checked[str(path)] = item['sha256']
        return item['sha256']

    def read(self, path):
        self.check(path)
        return read(path)


def select_validated_pool(rows, profiles, fallback_arm, priority_order=None, *,
                          summaries=None, localities=None, interventions=None,
                          allow_empirical_extrapolation=False):
    """Select one K4 pool, with old timing-free policy as a protected fallback.

    Families have independent units and coefficients. Unknown locality or native
    coverage is handled by model dispatch, not silently replaced by zeros.
    Caller registration must bind profiles, archive membership and fallback rule.
    """
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4:
        raise ValueError('complete nonempty K4 validated pool required')
    names = [row['arm'] for row in rows]
    if len(names) != len(set(names)) or fallback_arm not in names:
        raise ValueError('pool identity or fallback differs')
    if any(row.get('status') != 'validated' for row in rows):
        raise ValueError('all native programs must be validated before selection')
    if any(row.get('spec') != rows[0].get('spec') for row in rows):
        raise ValueError('mixed program pool')
    order = list(priority_order or [fallback_arm] + sorted(n for n in names if n != fallback_arm))
    if len(order) != len(names) or set(order) != set(names):
        raise ValueError('complete unique priority order required')
    evidence = {row['arm']: model.candidate_evidence(row['path'], row) for row in rows}
    natives = {row['arm']: read(Path(row['path']) / 'native-analysis.json') for row in rows}
    extra = {} if interventions is None else dict(interventions=interventions)
    decision = model.dispatch(natives, evidence, profiles, fallback_arm, order,
        summaries=summaries, localities=localities,
        allow_empirical_extrapolation=allow_empirical_extrapolation, **extra)
    model.verify_dispatch(decision, natives, evidence, profiles, fallback_arm, order,
        summaries=summaries, localities=localities,
        allow_empirical_extrapolation=allow_empirical_extrapolation, **extra)
    return decision, evidence


def verify_manifest_inventory(rows, pools):
    """Reject unregistered specs, duplicate rows and hidden extra candidate pools."""
    def key(spec, arm):
        return model.digest(spec), arm
    expected = set()
    for pool in pools.values():
        if set(pool['modes']) != {'joint', 'local'}:
            raise ValueError('exact J/LI pool inventory required')
        arms = [pool['legacy']] + [arm for mode in ('joint', 'local') for arm in pool['modes'][mode]['arms']]
        if len(arms) != len(set(arms)):
            raise ValueError('duplicate registered arm')
        additions = {key(pool['spec'], arm) for arm in arms}
        if expected.intersection(additions):
            raise ValueError('duplicate complete program group')
        expected.update(additions)
    actual = [key(row['spec'], row['arm']) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('manifest differs from complete registered pool inventory')
    return True


def _fresh_output(output):
    raw = Path(output)
    resolved = raw.resolve()
    allowed = [BASE.resolve(), DATA.resolve()]
    if raw.absolute() != resolved or raw.is_symlink() or not any(resolved.is_relative_to(root) and resolved != root for root in allowed):
        raise ValueError('fresh output inside new-stage roots required')
    if resolved.exists():
        raise FileExistsError('replay refuses to overwrite an earlier attempt')
    resolved.mkdir(parents=True)
    return resolved


def replay_known(run, bundle, summary_directory, locality_directory, output, *,
                 allow_empirical_extrapolation=False, intervention_directory=None):
    """Re-extract decisions only; measured timings are deliberately unavailable."""
    start = time.perf_counter_ns()
    seal = SealedInputs()
    run = Path(run).resolve()
    manifest = seal.read(run / 'manifest.json')
    pools = seal.read(run / 'candidate-pools.json')
    old = seal.read(run / 'partial-decisions.json')
    registration = seal.read(run / 'registration.json')
    protocol = seal.read(run / 'protocol.json')
    if manifest.get('status') != 'validated' or not old.get('decided_before_measurement') or old.get('uses_gpu_timing') is not False:
        raise ValueError('sealed validated, premeasurement decisions required')
    if manifest.get('registration_sha256') != sha(run / 'registration.json') or registration.get('protocol_sha256') != sha(run / 'protocol.json'):
        raise ValueError('sealed source/protocol registration differs')
    if manifest.get('candidate_pools_sha256') != sha(run / 'candidate-pools.json') or old.get('candidate_pools_sha256') != sha(run / 'candidate-pools.json'):
        raise ValueError('candidate pool identity differs')
    if set(pools) != set(old['groups']):
        raise ValueError('complete observed group inventory required')
    verify_manifest_inventory(manifest['rows'], pools)
    bundle = Path(bundle).resolve()
    profiles = model.load_bundle(bundle)
    out = _fresh_output(output)
    groups = {}; inputs = {str(bundle): sha(bundle)}; evidence_count = 0
    for record in read(bundle)['profiles'].values():
        inputs[record['path']] = record['sha256']
    try:
        for key, pool in pools.items():
            observed = old['groups'][key]
            if pool['spec'] != observed['spec']:
                raise ValueError('observed specification differs')
            matching = [row for row in manifest['rows'] if row['spec'] == pool['spec']]
            expected = {pool['legacy']} | {a for info in pool['modes'].values() for a in info['arms']}
            if len(matching) != len(expected) or {row['arm'] for row in matching} != expected:
                raise ValueError('candidate omitted or added')
            result = {}; group_out = out / key; group_out.mkdir()
            for mode in ('joint', 'local'):
                info = pool['modes'][mode]
                search = Path(info['search'])
                record = seal.read(search / 'record.json')
                archive = seal.read(search / 'candidate-archive.json')
                if info['record_sha256'] != sha(search / 'record.json') or info['archive_sha256'] != sha(search / 'candidate-archive.json'):
                    raise ValueError('sealed archive receipt differs')
                if archive['search'] != record['candidate_archive_search'] or archive['capacity'] != 4:
                    raise ValueError('archive receipt or capacity differs')
                arms = [mode + '-c' + str(entry['index']) for entry in archive['entries']]
                if info['arms'] != arms or len(arms) != len(set(arms)) or not 1 <= len(arms) <= 4:
                    raise ValueError('archive membership differs')
                rows = [{row['arm']: row for row in matching}[arm] for arm in arms]
                for row, entry in zip(rows, archive['entries']):
                    if row.get('archive_entry') != entry or row['assignment_sha256'] != entry['complete_assignment_sha256']:
                        raise ValueError('compiled row/archive witness differs')
                    for name, digest in row['artifact_hashes'].items():
                        if seal.check(Path(row['path']) / name) != digest:
                            raise ValueError('manifest proof hash differs')
                summaries = {}; localities = {}; interventions = {}
                for row in rows:
                    name = Path(row['path']).name + '.json'
                    documents = [(summary_directory, summaries), (locality_directory, localities)]
                    if intervention_directory is not None:
                        documents.append((intervention_directory, interventions))
                    for directory, target in documents:
                        path = Path(directory).resolve() / name
                        target[row['arm']] = read(path)
                        inputs[str(path)] = sha(path)
                fallback = observed['choices'][mode + '-affine-frame']['selected_source']
                order = [fallback] + [a for a in info['static_order'] if a != fallback]
                decision, evidence = select_validated_pool(rows, profiles, fallback, order,
                    summaries=summaries, localities=localities, interventions=interventions if intervention_directory else None,
                    allow_empirical_extrapolation=allow_empirical_extrapolation)
                for arm, value in evidence.items():
                    write(group_out / (arm + '-evidence.json'), value)
                write(group_out / (mode + '-decision.json'), decision)
                evidence_count += len(rows)
                result[mode] = dict(decision=decision, old_policy_source=fallback,
                    archive_sha256=info['archive_sha256'],
                    priority=order, native_requests=len(rows), uses_gpu_timing=False)
            groups[key] = dict(spec=pool['spec'], role='known-development-regression', modes=result)
        result = dict(schema_version=1, contract=CONTRACT, status='complete',
            role='known-development-regression-not-fresh', input_run=str(run),
            input_seal=dict(path=str(seal.path), sha256=KNOWN_INDEX_SHA256),
            verified_sealed_inputs=seal.checked, additional_inputs=inputs,
            implementation_sha256={str(Path(__file__).resolve()): sha(__file__), **model.implementation()},
            groups=groups, candidate_occurrences=evidence_count,
            group_count=len(groups), decision_count=2 * len(groups),
            empirical_extrapolation=allow_empirical_extrapolation,
            uses_gpu_timing=False, launches=0, compiler_calls=0,
            qualification_pass_claim=False, wall_ns=time.perf_counter_ns() - start)
        write(out / 'replay.json', result)
        write(out / 'completion.json', dict(status='complete', replay_sha256=sha(out / 'replay.json'),
            qualification_pass_claim=False, no_labels_read_by_algorithm=True))
        return result
    except Exception as error:
        write(out / 'failure.json', dict(status='failed', error=type(error).__name__ + ': ' + str(error)))
        raise


def install_label_read_guard():
    """Optional process-wide audit proving the runtime path cannot read labels."""
    observed = set()
    forbidden = {'measurement', 'group-quality.json', 'training-ledger.json',
                 'selection-margin-calibration.json', 'fit.json', 'development-regression.json'}
    def audit(event, arguments):
        if event != 'open' or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(arguments[0]))
        mode = arguments[1]
        if isinstance(mode, str) and '+' not in mode and any(x in mode for x in 'wax'):
            return
        paths = (path, path.resolve())
        if any(any(part in forbidden for part in candidate.parts) or candidate.name.endswith('-raw.json') for candidate in paths):
            raise PermissionError('runtime attempted to read a timing/training label: ' + str(path))
        observed.add(str(path))
    sys.addaudithook(audit)
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--summaries', required=True, type=Path)
    parser.add_argument('--localities', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--interventions', type=Path)
    parser.add_argument('--audit-no-labels', action='store_true')
    parser.add_argument('--empirical-exploration', action='store_true', help='known-data diagnostic only; never a calibrated safety claim')
    args = parser.parse_args()
    reads = install_label_read_guard() if args.audit_no_labels else None
    result = replay_known(args.run, args.bundle, args.summaries, args.localities, args.output,
        allow_empirical_extrapolation=args.empirical_exploration, intervention_directory=args.interventions)
    if reads is not None:
        write(args.output / 'runtime-read-audit.json', dict(status='passed', opened_paths=sorted(reads), label_read_attempts=0))
    print(json.dumps(dict(status=result['status'], groups=result['group_count'], decisions=result['decision_count'],
        occurrences=result['candidate_occurrences'], wall_seconds=result['wall_ns'] / 1e9)))


if __name__ == '__main__':
    main()
