"""Read existing native search evidence; never launches a search or GPU work."""
import itertools
import json
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text())


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError('invalid nonnegative integer ' + name)
    return value


def _ledger(record):
    total = record.get('total'); contract = record.get('budget_contract')
    if not isinstance(total, dict) or not isinstance(contract, dict):
        raise ValueError('missing shared search ledger')
    for name in ('probes', 'refinement', 'relationship', 'complete_scores', 'refinement_probes'):
        _count(total.get(name), name)
    if total['probes'] != sum(total[k] for k in ('relationship', 'complete_scores', 'refinement_probes')):
        raise ValueError('search probe decomposition disagrees')
    if total['refinement'] != total['refinement_probes']:
        raise ValueError('refinement probe decomposition disagrees')
    if contract.get('kind') != 'shared-total-search-probes-v1' or type(contract.get('search_performed')) is not bool:
        raise ValueError('missing shared search contract')
    for name, maximum in (('limit', 65536), ('refinement_limit', 256)):
        if _count(contract.get(name), name) > maximum:
            raise ValueError('search cap exceeds protocol maximum')
    if total['probes'] > contract['limit'] or total['refinement'] > contract['refinement_limit']:
        raise ValueError('search ledger exceeds cap')
    if contract.get('used') != total['probes'] or contract.get('refinement_used') != total['refinement']:
        raise ValueError('search contract disagrees with actual ledger')
    return total, contract


def validate_failure_sidecars(directory):
    """Validate actual failed-search evidence while keeping old envelope exact."""
    directory = Path(directory)
    envelope = _read(directory / 'failure.json')
    if set(envelope) != {'status', 'failure_kind', 'fatal', 'reason', 'limits'} or envelope['status'] != 'failed':
        raise ValueError('old failure envelope changed')
    budget = _read(directory / 'failure-budget.json')
    record = _read(directory / 'failure-search-record.json')
    if budget.get('schema_version') != 1 or type(budget.get('budget_context_available')) is not bool:
        raise ValueError('invalid failure budget sidecar schema')
    total, contract = _ledger(budget)
    if record.get('total') != total or record.get('budget_contract') != contract:
        raise ValueError('failure snapshot and budget sidecar disagree')
    if record.get('budget_context_available') != budget['budget_context_available']:
        raise ValueError('failure budget context disagrees')
    for name in ('status', 'failure_kind', 'fatal', 'reason'):
        if record.get(name) != envelope[name]:
            raise ValueError('failure snapshot and old envelope disagree')
    if not budget['budget_context_available'] and (any(total.values()) or contract['search_performed']):
        raise ValueError('presearch failure invents search work')
    if not contract['search_performed'] and any(total.values()):
        raise ValueError('nonsearch failure has search probes')
    return dict(schema_version=1, validated=True, budget_context_available=budget['budget_context_available'],
                total=total, budget_contract=contract, failure_kind=envelope['failure_kind'])


def _difference(candidate, reference):
    if candidate is None or reference is None:
        return dict(delta=None, ratio=None)
    return dict(delta=candidate-reference, ratio=candidate/reference if reference else None)


def summarize_search(rows, checker=None):
    """One CPU command observation per arm; no new gate or timing samples.

    checker, when supplied, is called as checker(record, treatment) for legal
    returned selections. Keeping it injectable avoids a driver import cycle.
    """
    observations = []
    seen = set()
    for row in rows:
        spec = row['spec']; treatment = row['treatment']; path = Path(row['path'])
        key = (row['scope'], spec['key'], row['arm'])
        if key in seen:
            raise ValueError('duplicate search observation: ' + repr(key))
        seen.add(key)
        phases = [p for p in row['phases'] if p['stage'] == 'native-selection']
        if len(phases) != 1:
            raise ValueError('expected exactly one native-selection command')
        command = _read(phases[0]['command_metadata'])
        elapsed = _count(command.get('elapsed_ns'), 'native command elapsed_ns')
        if (path / 'failure.json').exists():
            proof = validate_failure_sidecars(path)
            record = _read(path / 'failure-search-record.json')
            status = 'failed'
        else:
            record = _read(path / 'record.json'); status = 'success'
            if checker is not None:
                checker(record, treatment)
            proof = None
        total, contract = _ledger(record)
        timing = record.get('solve_timing_ns', {})
        for name, value in timing.items():
            _count(value, 'solve_timing_ns.' + name)
        components = record.get('search_components', [])
        seed = sum(_count(c.get(k), k) for c in components for k in ('seed_probes', 'seed_scores'))
        if seed > 8192 or seed > total['probes']:
            raise ValueError('seed work exceeds shared cap')
        observations.append(dict(scope=row['scope'], key=spec['key'], split=spec['split'],
            arm=row['arm'], model=row['model'], selection=row['selection'],
            search=treatment['search'], status=status, path=str(path),
            model_id=record.get('cost_model'), profile_sha256=record.get('cost_profile_sha256'),
            native_command_elapsed_ns=elapsed, native_command_exit_code=command.get('exit_code'),
            native_command_timeout_expired=command.get('timeout_expired'),
            command_metadata=str(phases[0]['command_metadata']), solve_timing_ns=timing,
            solve_total_ns=timing.get('total'), total_probes=total['probes'],
            refinement_probes=total['refinement'], seed_work=seed,
            seed_elapsed_ns=None, budget_contract=contract,
            objective_cost=record.get('cost') if status == 'success' else None,
            failure_proof=proof))
    comparisons = []
    groups = {}
    for observation in observations:
        groups.setdefault((observation['scope'], observation['key']), []).append(observation)
    for (scope, key), group in sorted(groups.items()):
        valid = [x for x in group if x['status'] == 'success']
        for reference, candidate in itertools.combinations(sorted(valid, key=lambda x:x['arm']), 2):
            same_model = (reference['model'], reference['model_id'], reference['profile_sha256']) == (candidate['model'], candidate['model_id'], candidate['profile_sha256'])
            ref_cost = reference['objective_cost']['cost']; cur_cost = candidate['objective_cost']['cost']
            comparisons.append(dict(scope=scope, key=key, split=candidate['split'],
                reference_arm=reference['arm'], candidate_arm=candidate['arm'],
                same_objective=same_model,
                cost_vector_delta=[x-y for x,y in zip(cur_cost, ref_cost)] if same_model else None,
                first_cost_component=_difference(cur_cost[0], ref_cost[0]) if same_model else dict(delta=None, ratio=None),
                cross_model_cost_note=None if same_model else 'Different objective coefficients/semantics: raw cost ratios are not comparable.',
                metrics={name:_difference(candidate[name],reference[name]) for name in
                    ('native_command_elapsed_ns','solve_total_ns','total_probes','refinement_probes','seed_work')}))
    by_arm = []
    arm_groups = {}
    for observation in observations:
        arm_groups.setdefault((observation['arm'], observation['scope'], observation['split']), []).append(observation)
    for (arm, scope, split), group in sorted(arm_groups.items()):
        by_arm.append(dict(arm=arm, scope=scope, split=split, observations=len(group),
            successful_observations=sum(x['status']=='success' for x in group),
            native_command_elapsed_ns=sum(x['native_command_elapsed_ns'] for x in group),
            solve_total_ns=sum(x['solve_total_ns'] for x in group) if all(x['solve_total_ns'] is not None for x in group) else None,
            total_probes=sum(x['total_probes'] for x in group),
            refinement_probes=sum(x['refinement_probes'] for x in group), seed_work=sum(x['seed_work'] for x in group)))
    return dict(schema_version=1, status='descriptive-existing-evidence', measurement_scope=dict(native_command='end-to-end native process: restore prepared proof/domain, selection, materialization, export, optional demand ledger/oracle and diagnostics', solver_clock='record.solve_timing_ns is instrumentation inside solveAblation; excludes native restore/materialize/export and is not a seed-only clock', seed_clock='not instrumented'), observations=observations,
        by_arm_scope_split=by_arm, pairwise_case_comparisons=comparisons,
        limitations=[
            'One observed CPU command per selected arm; no repeated CPU timing or confidence interval.',
            'Native command duration includes setup, scoring, optional oracle/materialization/export; it is not solver time.',
            'J-old means seed-off ablation in the current repaired binary, not an old historical executable.',
            'Seed-specific time was not instrumented; seed_elapsed_ns is unavailable, not zero.',
            'Probe equality is a search work contract, not equal CPU duration or GPU performance.',
            'Model cost differences/ratios are reported only within the same model identity/profile.',
            'Pairwise observations introduce no new acceptance gate and do not choose a faster arm.'])
