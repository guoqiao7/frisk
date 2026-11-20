"""Audited tie preferences and nonnegative affine model completion.

These are conditional model-family certificates and resource preferences. They
are neither hardware latency bounds nor a fitted replacement for native14.
No compiler, CUDA execution, measurement read or cache write occurs here.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import native_cost_calibration as calibration
import partial_cost_selection as partial

CONTRACT = 'frisk-gf2-compiled-decision-priority-v1'
AFFINE_CONTRACT = 'frisk-shared-nonnegative-affine-completion-v1'
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
GF2_DIMENSION = 'gf2_global_sectors32'
BASIS = 'conditional independent nonnegative shared coefficients; not hardware latency'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def integer(value):
    return type(value) is int and value >= 0


def _seal(payload):
    return {'payload': payload, 'payload_sha256': digest(payload)}


def _open(receipt, contract):
    if not isinstance(receipt, dict) or set(receipt) != {'payload', 'payload_sha256'}:
        raise ValueError('receipt envelope differs')
    p = receipt['payload']
    if type(p.get('schema_version')) is not int or p['schema_version'] != 1:
        raise ValueError('strict receipt schema version required')
    if digest(p) != receipt['payload_sha256'] or p.get('contract') != contract or p.get('implementation_sha256') != IMPLEMENTATION_SHA256:
        raise ValueError('receipt implementation or payload changed')
    if p.get('uses_gpu_timing') is not False:
        raise ValueError('timing-free receipt required')
    return p


def _order(names, order):
    order = list(order)
    if len(order) != len(names) or len(set(order)) != len(order) or set(order) != set(names):
        raise ValueError('priority must contain every pool member exactly once')
    return order


def common_static_contract(value):
    """Verify comparable score units while retaining actual objective identity."""
    fields={'contract','grammar','frequency_mode','weight_provenance','units','feature_order','coefficients','sha256'}
    if not isinstance(value,dict) or set(value)!=fields:
        raise ValueError('strict common static contract required')
    if (value['contract']!='sm90-network-static-common-v1'
        or value['grammar']!='ordered-surviving-network-source-issues-v1'
        or value['weight_provenance']!='migrated-engineering-proxy-not-refit'
        or value['units']!='nonnegative-score-ticks-per-feature-unit-not-cycles'
        or not isinstance(value['frequency_mode'],str) or not value['frequency_mode']):
        raise ValueError('static score units/grammar differ')
    order=value['feature_order'];weights=value['coefficients']
    if (not isinstance(order,list) or len(order)!=20 or len(set(order))!=20
        or any(not isinstance(n,str) or not n or '\n' in n or '=' in n for n in order)
        or not isinstance(weights,dict) or set(weights)!=set(order)
        or any(not integer(v) for v in weights.values())):
        raise ValueError('invalid common static coefficient domain')
    lines=[value[n] for n in ('contract','grammar','frequency_mode','weight_provenance','units')]
    lines += [n+'='+str(weights[n]) for n in order]
    actual=hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()
    if actual!=value['sha256']:raise ValueError('common static contract digest differs')
    return actual


def _static_comparison_domain(p):
    common=p.get('common_static_contract')
    if common is None:
        return ('exact-model',p['static_model_id'],p['static_profile_sha256'])
    digest=common_static_contract(common)
    if common['frequency_mode']!=p['frequency_mode']:raise ValueError('common static frequency differs')
    if (p.get('static_effective_model_id')!=p['static_model_id']
        or p.get('static_effective_profile_sha256')!=p['static_profile_sha256']
        or not isinstance(p.get('static_base_model_id'),str)
        or not isinstance(p.get('static_base_profile_sha256'),str)
        or len(p['static_base_profile_sha256'])!=64):
        raise ValueError('effective/base objective identity differs')
    return ('common-network-units',digest,p['static_base_model_id'],p['static_base_profile_sha256'])


def _evidence(receipts):
    result = {}
    for arm, receipt in receipts.items():
        p = _open(receipt, CONTRACT)
        if p.get('kind') != 'candidate-evidence' or p.get('arm') != arm:
            raise ValueError('candidate evidence identity differs')
        cost = p.get('static_cost_tuple')
        if not isinstance(cost, list) or len(cost) != 9 or not all(integer(x) for x in cost):
            raise ValueError('invalid unsaturated static cost tuple')
        for field, metric in [('gf2', 'global_sectors32'), ('local', 'bytes_per_thread')]:
            value = p.get(field)
            if not isinstance(value, dict) or value.get('status') not in ('available', 'unavailable'):
                raise ValueError('typed metric evidence required')
            if value['status'] == 'available' and not integer(value.get(metric)):
                raise ValueError('invalid nonnegative evidence metric')
            if value['status'] == 'unavailable' and value.get(metric) is not None:
                raise ValueError('unavailable metric must not supply a number')
        if not isinstance(arm, str) or not arm or not isinstance(p.get('native_sha256'), str) or len(p['native_sha256']) != 64:
            raise ValueError('candidate name/native digest missing')
        result[arm] = p
    if not result:
        raise ValueError('nonempty candidate pool required')
    if len({(_static_comparison_domain(p), p['frequency_mode'], p['prepared_digest']) for p in result.values()}) != 1:
        raise ValueError('mixed static model/domain/frequency evidence')
    return result


def candidate_evidence(directory, row):
    """Bind verified GF2 address evidence and loaded-image LOCAL footprint.

    Caller supplies the complete validated row. Only this explicit allowlist is
    read; measurement files are never consulted. Present but invalid evidence is
    an integrity error, while a declared carrier fallback is typed unavailable.
    """
    d = Path(directory)
    if row.get('status') != 'validated' or str(d) != row.get('path'):
        raise ValueError('validated exact candidate required')
    names = ('record.json', 'execution.json', 'kernel.ttgir', 'kernel.cubin',
             'native-analysis.json', 'generation-inputs.json', 'image-identity.json',
             'correctness.json', 'resolved-compiler-resources.json',
             'compile-cache-receipt.json', 'surviving-coverage-audit.json',
             'demand-coverage-audit.json')
    bindings = {}
    for name in names:
        path = d / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != d.resolve():
            raise ValueError('unsafe or missing candidate evidence')
        bindings[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get('artifact_hashes', {}).get(name) != bindings[name]:
            raise ValueError('candidate evidence differs from validated row: ' + name)
    read = lambda name: json.loads((d / name).read_text())
    record, native, correctness = read('record.json'), read('native-analysis.json'), read('correctness.json')
    resolved = read('resolved-compiler-resources.json')
    image = read('image-identity.json')
    if (record.get('status') != 'success' or record['cost'].get('saturated') is not False or
        record['complete_assignment_sha256'] != row['assignment_sha256'] or
        image['kernel.cubin'] != bindings['kernel.cubin'] or
        correctness['cubin_sha256'] != bindings['kernel.cubin'] or
        correctness['image_identity_sha256'] != bindings['image-identity.json'] or
        correctness.get('no_kernel_timing') is not True or
        native.get('identity', {}).get('cubin_sha256') != bindings['kernel.cubin'] or
        native.get('identity', {}).get('entry') != 'frisk_execution' or
        record['cost'] != row['cost'] or record['cost_model'] != row['model_id']):
        raise ValueError('assignment or accepted image binding differs')
    envelope = read('compile-cache-receipt.json')
    if digest(envelope['payload']) != envelope['payload_sha256'] or envelope['payload']['execution_sha256'] != bindings['execution.json']:
        raise ValueError('compiled execution receipt differs')
    for name in ('kernel.cubin', 'kernel.ttgir'):
        if envelope['payload']['native_artifacts'][name]['sha256'] != bindings[name]:
            raise ValueError('native artifact receipt binding differs')
    generation = read('generation-inputs.json')
    for name in ('record.json', 'execution.json', 'surviving-coverage-audit.json', 'demand-coverage-audit.json'):
        if generation[name] != bindings[name]:
            raise ValueError('GF2 generation receipt binding differs')
    from resource_prediction import confirm_driver_resources
    confirmation = confirm_driver_resources(resolved, correctness['physical_resources'], loaded_cubin_sha256=bindings['kernel.cubin'], loaded_entry='frisk_execution')
    if confirmation != correctness['compiler_resource_confirmation']:
        raise ValueError('driver resource confirmation differs')
    frame = confirmation['local_bytes']
    compiler = resolved['compiler_exact']['resources']
    if not integer(frame) or not integer(compiler['stack_bytes_per_thread']) or not integer(compiler['raw_local_bytes']):
        raise ValueError('invalid compiled resource units')
    local = dict(status='available', bytes_per_thread=frame,
                 source='CUDA-driver-LOCAL_SIZE_BYTES',
                 compiler_stack_bytes_per_thread=compiler['stack_bytes_per_thread'],
                 compiler_raw_local_bytes=compiler['raw_local_bytes'],
                 traffic_bytes=None, spill_count=None,
                 semantics='allocation footprint; not traffic, spill count or a latency theorem')
    ledger, proof = record.get('surviving_coverage', {}), read('surviving-coverage-audit.json')
    gf2 = dict(status='unavailable', global_sectors32=None, reason='no-proven-surviving-address-ledger')
    if ledger.get('proof_status') == 'Proven':
        body = ledger.get('surviving')
        if (not isinstance(body, dict) or body.get('exact') is not True or proof.get('status') != 'passed' or
            proof.get('priced_features_checked') is not True or proof.get('no_gpu') is not True or
            proof.get('work_basis') != ledger.get('work_basis')):
            raise ValueError('GF2 surviving proof incomplete')
        accesses = body.get('accesses')
        if not isinstance(accesses, list) or len({x['constraint'] for x in accesses}) != len(accesses) or any(not integer(x.get('constraint')) or not integer(x.get('global_sectors32')) for x in accesses):
            raise ValueError('invalid GF2 access ledger')
        sectors = sum(x['global_sectors32'] for x in accesses)
        if sectors != proof['expected_priced_features'].get('global_sectors32', 0) or sectors != record['cost']['features'].get('global_sectors32', 0):
            raise ValueError('GF2 raw ledger and priced feature differ')
        demand = read('demand-coverage-audit.json')
        if demand.get('status') != 'passed' or demand.get('emitted_ttgir_sha256') != bindings['kernel.ttgir']:
            raise ValueError('GF2 proof does not bind emitted execution')
        gf2 = dict(status='available', global_sectors32=sectors,
                   unit='proven-layout-32B-warp-request-sectors',
                   work_basis=ledger['work_basis'], ledger_sha256=digest(body),
                   accesses=[{'constraint': x['constraint'], 'global_sectors32': x['global_sectors32']} for x in accesses],
                   semantics='Frisk affine address request burden; not measured native transactions or latency')
    elif proof.get('status') != 'explicit-fallback':
        raise ValueError('unavailable GF2 evidence lacks declared fallback')
    comparison={}
    if 'common_static_contract' in record:
        common_static_contract(record['common_static_contract'])
        comparison={n:record[n] for n in ('common_static_contract','static_base_model_id','static_base_profile_sha256',
                                         'static_effective_model_id','static_effective_profile_sha256')}
    if 'complete_program_sha256' in record:comparison['complete_program_sha256']=record['complete_program_sha256']
    return _seal(dict(schema_version=1, contract=CONTRACT, implementation_sha256=IMPLEMENTATION_SHA256,
        **comparison, kind='candidate-evidence', arm=row['arm'], assignment_sha256=row['assignment_sha256'],
        static_cost_tuple=record['cost']['cost'], static_model_id=record['cost_model'],
        static_profile_sha256=record.get('cost_profile_sha256'), frequency_mode=record['frequency_mode'],
        prepared_digest=record['prepared_digest'], native_sha256=digest(native),
        gf2=gf2, local=local, bindings=bindings, compile_key=envelope['payload']['key'],
        cubin_sha256=bindings['kernel.cubin'], uses_gpu_timing=False,
        information_stage='after-functional-resource-validation-before-event-timing'))


def verify_candidate_evidence(directory, row, receipt):
    expected = candidate_evidence(directory, row)
    if receipt != expected:
        raise ValueError('candidate evidence changed')
    return expected


def native_priority(intervals, evidence, old_priority, *, enabled=True):
    """Reorder exact equal-score classes only; bounded rows never join them."""
    if type(enabled) is not bool:
        raise ValueError('explicit boolean rule switch required')
    ev = _evidence(evidence)
    order = _order(intervals, old_priority)
    if set(ev) != set(intervals):
        raise ValueError('priority pool differs')
    groups = {}
    for arm in order:
        x = intervals[arm]
        partial._validate_interval(x)
        if x['native_sha256'] != ev[arm]['native_sha256']:
            raise ValueError('native interval/evidence differs')
        if x['status'] == 'exact':
            groups.setdefault(x['lower_ticks'], []).append(arm)
    result, receipts = list(order), []
    for score, members in sorted(groups.items()):
        usable = enabled and all(ev[a]['gf2']['status'] == 'available' for a in members)
        ranked = sorted(members, key=lambda a: (ev[a]['gf2']['global_sectors32'], order.index(a))) if usable else list(members)
        positions = [order.index(a) for a in members]
        for position, arm in zip(positions, ranked):
            result[position] = arm
        receipts.append(dict(score_ticks=score, members=members, ordered_members=ranked,
            applied=usable, reason='proven-all-members' if usable else 'disabled-or-incomparable-class',
            sectors={a: ev[a]['gf2']['global_sectors32'] for a in members}))
    return _seal(dict(schema_version=1, contract=CONTRACT, implementation_sha256=IMPLEMENTATION_SHA256,
        kind='native-exact-tie-priority', enabled=enabled, priority=result, classes=receipts,
        old_priority=order, evidence={a: digest(evidence[a]) for a in sorted(ev)},
        intervals={a: digest(intervals[a]) for a in sorted(intervals)}, uses_gpu_timing=False,
        claim='preference within exact equal-score members; preserves native argmin membership, not whole-pool lex(cost,sectors) optimality'))


def verify_native_priority(intervals, evidence, old_priority, receipt, *, enabled=True):
    expected = native_priority(intervals, evidence, old_priority, enabled=enabled)
    if receipt != expected:
        raise ValueError('native priority receipt changed')
    return expected


def static_fallback(evidence, old_order, *, enabled=True):
    """Refine primary-minimal fallback by footprint, retaining old full tuple."""
    if type(enabled) is not bool:
        raise ValueError('explicit boolean rule switch required')
    ev = _evidence(evidence)
    order = _order(ev, old_order)
    if any(ev[a]['static_cost_tuple'] > ev[b]['static_cost_tuple'] for a, b in zip(order, order[1:])):
        raise ValueError('old static order is not full-tuple sorted')
    primary = min(p['static_cost_tuple'][0] for p in ev.values())
    members = [a for a in order if ev[a]['static_cost_tuple'][0] == primary]
    usable = enabled and all(ev[a]['local']['status'] == 'available' for a in members)
    ranked = sorted(members, key=lambda a: (ev[a]['local']['bytes_per_thread'], ev[a]['static_cost_tuple'], order.index(a))) if usable else list(members)
    return _seal(dict(schema_version=1, contract=CONTRACT, implementation_sha256=IMPLEMENTATION_SHA256,
        kind='static-primary-fallback', enabled=enabled, selected_source=ranked[0], original_source=order[0],
        old_order=order, primary_minimum=primary, primary_members=members, ordered_members=ranked,
        applied=usable, reason='confirmed-all-members' if usable else 'disabled-or-incomparable-class',
        local_bytes_per_thread={a: ev[a]['local']['bytes_per_thread'] for a in members},
        evidence={a: digest(evidence[a]) for a in sorted(ev)}, uses_gpu_timing=False,
        preserves_primary_minimum=True, full_tuple_minimum_preserved=ev[ranked[0]]['static_cost_tuple'] == ev[order[0]]['static_cost_tuple'],
        claim='compiled archive primary argmin member; footprint preference is not a hardware latency theorem'))


def verify_static_fallback(evidence, old_order, receipt, *, enabled=True):
    expected = static_fallback(evidence, old_order, enabled=enabled)
    if receipt != expected:
        raise ValueError('static fallback receipt changed')
    return expected


def affine_matrix(intervals, evidence, profile):
    """Complete omitted GF2 sectors with one shared nonnegative coefficient.

    Training-zero coefficient-zero native dimensions remain independent shared
    unknown coefficients, with each candidate's exact observed count, including
    zeros. Unsupported native/CFG evidence never acquires a numeric lower bound.
    """
    weights, maxima = partial._profile(profile)
    ev = _evidence(evidence)
    if set(ev) != set(intervals):
        raise ValueError('affine pool differs')
    unknown = sorted(n for n in calibration.MODEL_FEATURES if maxima[n] == 0 and weights[n] == 0)
    dimensions = ['native-untrained:' + n for n in unknown] + [GF2_DIMENSION]
    rows, unavailable = {}, []
    for arm in sorted(intervals):
        x = intervals[arm]
        partial._validate_interval(x)
        if x['profile_sha256'] != digest(profile) or x['native_sha256'] != ev[arm]['native_sha256']:
            raise ValueError('affine profile/native binding differs')
        if x['status'] == 'unsupported' or ev[arm]['gf2']['status'] != 'available':
            reason = x.get('reason') if x['status'] == 'unsupported' else 'GF2-sector-proof-unavailable'
            rows[arm] = dict(status='unsupported', reason=reason, base_ticks=None, counts=None)
            unavailable.append(arm)
            continue
        f = x['features']
        if set(f) != set(calibration.MODEL_FEATURES) or any(not integer(v) for v in f.values()):
            raise ValueError('affine native feature schema differs')
        base = sum(f[n] * weights[n] for n in calibration.MODEL_FEATURES if n not in unknown)
        if base != x['lower_ticks']:
            raise ValueError('affine known intercept differs')
        counts = {'native-untrained:' + n: f[n] for n in unknown}
        counts[GF2_DIMENSION] = ev[arm]['gf2']['global_sectors32']
        rows[arm] = dict(status='available', base_ticks=base, counts=counts)
    return _seal(dict(schema_version=1, contract=AFFINE_CONTRACT, implementation_sha256=IMPLEMENTATION_SHA256,
        kind='affine-matrix', status='unsupported' if unavailable else 'available', rows=rows,
        dimensions=dimensions, unsupported_candidates=unavailable, profile_sha256=digest(profile),
        intervals={a: digest(intervals[a]) for a in sorted(intervals)},
        evidence={a: digest(evidence[a]) for a in sorted(ev)}, uses_gpu_timing=False,
        coefficient_domain={n: dict(lower=0, upper=None, shared_across_all_candidates=True) for n in dimensions},
        equation='cost_i=base_i+sum(count_i[d]*u[d]); every u[d]>=0 independently', basis=BASIS))


def verify_affine_matrix(intervals, evidence, profile, receipt):
    expected = affine_matrix(intervals, evidence, profile)
    if receipt != expected:
        raise ValueError('affine matrix receipt changed')
    return expected


def select_affine(matrix, *, fallback_arm, priority):
    p = _open(matrix, AFFINE_CONTRACT)
    if p.get('kind') != 'affine-matrix' or p.get('basis') != BASIS:
        raise ValueError('affine family interpretation differs')
    rows, dimensions = p['rows'], p['dimensions']
    if not isinstance(dimensions, list) or any(not isinstance(d, str) or not d for d in dimensions):
        raise ValueError('invalid affine dimensions')
    expected_domain = {d: dict(lower=0, upper=None, shared_across_all_candidates=True) for d in dimensions}
    domain = p.get('coefficient_domain')
    if (not isinstance(domain, dict) or set(domain) != set(dimensions) or
        any(not isinstance(v, dict) or set(v) != {'lower', 'upper', 'shared_across_all_candidates'} or
            type(v['lower']) is not int or v['lower'] != 0 or v['upper'] is not None or
            v['shared_across_all_candidates'] is not True for v in domain.values())):
        raise ValueError('affine nonnegative shared coefficient domain differs')
    if not isinstance(rows, dict) or any(not isinstance(a, str) or not a for a in rows):
        raise ValueError('named affine rows required')
    order = _order(rows, priority)
    if fallback_arm not in rows or len(set(dimensions)) != len(dimensions):
        raise ValueError('invalid affine family or fallback')
    for arm, row in rows.items():
        if row['status'] == 'available':
            if not integer(row['base_ticks']) or set(row['counts']) != set(dimensions) or any(not integer(v) for v in row['counts'].values()):
                raise ValueError('invalid affine row')
        elif row['status'] != 'unsupported' or row['base_ticks'] is not None or row['counts'] is not None:
            raise ValueError('unsupported affine row must not assert a bound')
    hard = [a for a in order if rows[a]['status'] != 'available']
    if p.get('status') != ('unsupported' if hard else 'available') or p.get('unsupported_candidates') != sorted(hard):
        raise ValueError('affine support matrix status differs')
    members, witnesses = [], {}
    if not hard:
        for a in order:
            ra = rows[a]
            failure = None
            for b in order:
                rb = rows[b]
                if ra['base_ticks'] > rb['base_ticks']:
                    failure = dict(opponent=b, dimension='base', coefficients={d: 0 for d in dimensions})
                    break
                for d in dimensions:
                    gap = ra['counts'][d] - rb['counts'][d]
                    if gap > 0:
                        value = max(1, (rb['base_ticks'] - ra['base_ticks']) // gap + 1)
                        failure = dict(opponent=b, dimension=d, coefficients={k: value if k == d else 0 for k in dimensions})
                        break
                if failure:
                    break
            if failure:
                witnesses[a] = failure
            else:
                members.append(a)
    selected = members[0] if members else fallback_arm
    reasons = ['unsupported:' + a for a in hard] if hard else ([] if members else ['no-universal-affine-argmin-member'])
    comparisons = []
    if members:
        a = rows[selected]
        comparisons = [dict(opponent=b, base_gap=rows[b]['base_ticks'] - a['base_ticks'],
                            count_gaps={d: rows[b]['counts'][d] - a['counts'][d] for d in dimensions}) for b in order]
    return _seal(dict(schema_version=1, contract=AFFINE_CONTRACT, implementation_sha256=IMPLEMENTATION_SHA256,
        kind='affine-selection', status='affine-certified' if members else 'fallback', selected_source=selected,
        fallback_arm=fallback_arm, priority=order, reasons=reasons, matrix_sha256=digest(matrix),
        universal_argmin_members=members, unsupported_candidates=hard, comparisons=comparisons,
        necessity_counterexamples=witnesses, uses_gpu_timing=False,
        argmin_membership_only=True, unique_optimum_proven=False, basis=BASIS,
        proof='coordinatewise base/count dominance is necessary and sufficient for this fixed candidate to minimize every independent nonnegative shared-coefficient completion'))


def verify_affine_selection(matrix, receipt, *, fallback_arm, priority):
    expected = select_affine(matrix, fallback_arm=fallback_arm, priority=priority)
    if receipt != expected:
        raise ValueError('affine selection receipt changed')
    return expected
