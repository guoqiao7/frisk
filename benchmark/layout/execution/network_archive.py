"""Strict timing-free validation of the fixed K4 layout/network archive.

The plans are static source-grammar proofs. This module does not predict native
registers, launch a compiler, or use performance labels to choose a pool.
"""
from __future__ import annotations
import hashlib
import json
import re
from functools import lru_cache

CONTRACT = 'fixed-K4-network-template-v1'
VERSION = 'shared-network-candidate-archive-v2'
FEATURES = ('access_warp_issues', 'global_sectors32', 'shared_bank_excess',
 'copy_setup_units', 'mma_instructions', 'reduction_element_ops',
 'alu_warp_issues', 'div_warp_issues', 'exp_warp_issues', 'cast_warp_issues',
 'conversion_register_bytes', 'conversion_shuffle_bytes', 'conversion_shared_bytes',
 'conversion_conservative_bytes', 'synchronization_barriers',
 'shared_read_bank_excess', 'shared_write_bank_excess', 'shared_warp_issues',
 'shuffle_warp_issues', 'communication_alu_warp_issues')
ENTRY_FIELDS = set(('index assignment instruction_bindings reduction_bindings cost '
 'prepared_digest conversion_policy local_incumbent joint_incumbent probes_at_observation '
 'refinement_at_observation certificate_verified network_objective network_variant cost_model '
 'cost_profile cost_profile_sha256 static_base_model_id static_base_profile_sha256 '
 'static_effective_model_id static_effective_profile_sha256 common_static_contract origins '
 'complete_assignment_sha256 complete_program_sha256').split())


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _nat(value):
    return type(value) is int and 0 <= value <= (1 << 63) - 1


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _json(value, pretty=False):
    return json.dumps(value, sort_keys=True, indent=2 if pretty else None,
                      separators=None if pretty else (',', ':'),
                      ensure_ascii=False, allow_nan=False)


def _digest(value, pretty=False):
    return hashlib.sha256(_json(value, pretty).encode()).hexdigest()


def _common(value):
    fields = {'contract', 'grammar', 'frequency_mode', 'weight_provenance',
              'units', 'feature_order', 'coefficients', 'sha256'}
    _require(isinstance(value, dict) and set(value) == fields, 'common static schema differs')
    _require(value['contract'] == 'sm90-network-static-common-v1'
        and value['grammar'] == 'ordered-surviving-network-source-issues-v1'
        and value['weight_provenance'] == 'migrated-engineering-proxy-not-refit'
        and value['units'] == 'nonnegative-score-ticks-per-feature-unit-not-cycles'
        and value['frequency_mode'] in ('legacy', 'static-trip-count'), 'common score grammar differs')
    _require(value['feature_order'] == list(FEATURES), 'common feature order differs')
    weights = value['coefficients']
    _require(isinstance(weights, dict) and set(weights) == set(FEATURES)
             and all(_nat(v) for v in weights.values()), 'common coefficient domain differs')
    lines = [value[n] for n in ('contract', 'grammar', 'frequency_mode', 'weight_provenance', 'units')]
    lines += [n + '=' + str(weights[n]) for n in FEATURES]
    _require(hashlib.sha256(('\n'.join(lines) + '\n').encode()).hexdigest() == value['sha256'],
             'common score contract digest differs')
    return value


def _profile(value, common, objective):
    fields = {'schema_version', 'model', 'coefficients', 'provenance', 'network_objective', 'weight_provenance'}
    _require(isinstance(value, dict) and set(value) == fields, 'network profile schema differs')
    _require(type(value['schema_version']) is int and value['schema_version'] == 1
        and value['model'] == 'sm90-layout-network-surviving-proxy-v1'
        and value['network_objective'] == objective
        and value['weight_provenance'] == common['weight_provenance']
        and isinstance(value['provenance'], str) and bool(value['provenance'])
        and '\x00' not in value['provenance'] and len(value['provenance']) <= 16384
        and value['coefficients'] == common['coefficients'], 'network profile identity/weights differ')
    weights = value['coefficients']
    _require(all(weights[n] > 0 for n in ('synchronization_barriers', 'shared_warp_issues',
        'shuffle_warp_issues', 'communication_alu_warp_issues'))
        and all(weights[n] == 0 for n in ('shared_bank_excess', 'conversion_register_bytes',
        'conversion_shuffle_bytes', 'conversion_shared_bytes', 'conversion_conservative_bytes')),
        'network profile structural coefficient restrictions differ')
    digest = _digest(value)
    suffix = '-leg' if common['frequency_mode'] == 'legacy' else '-stc'
    return digest, 'sm90-layout-network-v1-' + digest[:32] + suffix


def _identity(entry):
    assignment = entry.get('assignment')
    _require(isinstance(assignment, list) and bool(assignment), 'complete assignment required')
    ids = []
    for item in assignment:
        _require(isinstance(item, dict) and set(item) == {'variable', 'name', 'candidate'}
            and _nat(item['variable']) and isinstance(item['name'], str), 'assignment variable schema differs')
        candidate = item['candidate']
        _require(isinstance(candidate, dict) and set(candidate) == {'attribute', 'id'}
            and isinstance(candidate['attribute'], str) and _sha(candidate['id']), 'candidate identity differs')
        ids.append(item['variable'])
    _require(sorted(ids) == list(range(len(ids))), 'assignment variable coverage differs')
    for name in ('instruction_bindings', 'reduction_bindings'):
        _require(isinstance(entry.get(name), list), 'complete instruction/reduction bindings required')
        seen = set()
        for binding in entry[name]:
            _require(isinstance(binding, dict) and set(binding) == {'constraint', 'binding'}
                and _nat(binding['constraint']) and isinstance(binding['binding'], str)
                and binding['constraint'] not in seen, 'binding identity differs')
            seen.add(binding['constraint'])
    body = {n: entry[n] for n in ('assignment', 'instruction_bindings', 'reduction_bindings')}
    _require(_digest(body, True) == entry.get('complete_assignment_sha256'), 'complete assignment digest differs')
    return body


def _entry(entry, common, *, fields=ENTRY_FIELDS):
    _require(isinstance(entry, dict) and set(entry) == fields, 'strict v2 archive entry schema differs')
    _require(_nat(entry['index']) and entry['conversion_policy'] == 'independent'
        and entry['certificate_verified'] is True and _sha(entry['prepared_digest'])
        and entry['network_objective'] in ('min-shfl', 'min-work')
        and all(type(entry[n]) is bool for n in ('local_incumbent', 'joint_incumbent', 'network_variant'))
        and _nat(entry['probes_at_observation']) and _nat(entry['refinement_at_observation'])
        and isinstance(entry['origins'], list) and bool(entry['origins'])
        and all(isinstance(o, str) and o for o in entry['origins']), 'archive entry metadata differs')
    _require(entry['network_variant'] == (entry['network_objective'] == 'min-work'), 'variant objective differs')
    _require(entry['common_static_contract'] == common, 'archive common score units differ')
    actual_sha, actual_id = _profile(entry['cost_profile'], common, entry['network_objective'])
    base = dict(entry['cost_profile'], network_objective='min-shfl')
    base_sha, base_id = _profile(base, common, 'min-shfl')
    _require(entry['cost_profile_sha256'] == entry['static_effective_profile_sha256'] == actual_sha
        and entry['cost_model'] == entry['static_effective_model_id'] == actual_id
        and entry['static_base_profile_sha256'] == base_sha and entry['static_base_model_id'] == base_id,
        'archive effective/base profile identity differs')
    _identity(entry)
    program = {n: entry[n] for n in ('complete_assignment_sha256', 'network_objective', 'cost_model')}
    _require(_digest(program, True) == entry['complete_program_sha256'], 'complete program digest differs')
    cost = entry['cost']
    _require(isinstance(cost, dict) and set(cost) == {'cost', 'features', 'saturated'}
        and cost['saturated'] is False and isinstance(cost['cost'], list) and len(cost['cost']) == 9
        and all(_nat(v) for v in cost['cost']) and isinstance(cost['features'], dict)
        and set(cost['features']) <= set(FEATURES) and all(_nat(v) for v in cost['features'].values()),
        'finite complete static score required')
    _require(cost['cost'][0] == sum(common['coefficients'][n] * v for n, v in cost['features'].items()),
             'complete static score and priced features differ')
    return entry


def _counters(value):
    fields = {'probes', 'refinement', 'relationship', 'complete_scores', 'refinement_probes'}
    _require(isinstance(value, dict) and set(value) == fields and all(_nat(v) for v in value.values()),
             'search counter schema differs')
    _require(value['probes'] == sum(value[n] for n in ('relationship', 'complete_scores', 'refinement_probes')),
             'search probes do not reconcile')
    _require(value['refinement'] == value['refinement_probes'], 'refinement receipt does not reconcile')
    return value


def _execution(plan):
    if plan is None:
        return None
    return {n: plan[n] for n in ('registers', 'affine_base', 'affine_columns', 'steps', 'counts')}


@lru_cache(maxsize=128)
def _recompute_pair_cached(encoded):
    from shuffle_network import plan_shuffle_network
    network = json.loads(encoded)
    registers = network['registers'];columns = network['affine_columns'];base = network['affine_base']
    _require(type(registers) is int and 1 <= registers <= 512 and registers & (registers - 1) == 0,
             'network register domain differs')
    _require(isinstance(columns, list) and len(columns) == registers.bit_length() + 6
        and type(base) is int and 0 <= base < 32 * registers
        and all(type(c) is int and 0 <= c < 32 * registers for c in columns), 'affine owner domain differs')
    values = [base]
    for column in columns:
        values += [v ^ column for v in values]
    destinations = []
    for i, low in enumerate(values):
        destinations.append({'source': {'reg': low % registers, 'lane': low // registers,
            'warp': i // (32 * registers), 'warp_group': 0}})
    plan = dict(mode='Shuffle', source_registers=registers, destination_registers=registers,
        threads=128, warps=4, payload=dict(bits=32, words=[dict(bit_offset=0, live_bits=32)]),
        destinations=destinations)
    return tuple(_json(plan_shuffle_network(plan, objective=o)) for o in ('min-shfl', 'min-work'))


def _recompute_pair(encoded):
    # Cached proofs must never expose mutable aliasing to caller-owned packets.
    return tuple(json.loads(p) for p in _recompute_pair_cached(encoded))


def _plans(proofs, count):
    _require(isinstance(proofs, list) and len(proofs) == count, 'boundary plan proof coverage differs')
    seen = set();changed = False
    for p in proofs:
        _require(isinstance(p, dict) and set(p) == {'constraint', 'source', 'target', 'source_registers',
            'target_registers', 'min-shfl', 'min-work'} and all(_nat(p[n]) for n in
            ('constraint', 'source', 'target', 'source_registers', 'target_registers'))
            and p['constraint'] not in seen, 'boundary proof identity differs')
        seen.add(p['constraint']);a, b = p['min-shfl'], p['min-work']
        if a is None and b is None:
            continue
        _require(isinstance(a, dict) and isinstance(b, dict), 'objective changed network eligibility')
        expected = _recompute_pair(_json({n: a[n] for n in ('registers', 'affine_base', 'affine_columns')}))
        _require(a == expected[0] and b == expected[1], 'network steps/counts/full owner proof differs')
        changed |= _execution(a) != _execution(b)
    return changed


def _check_archive_common(record, archive, mode, capacity, *, expected_version, entry_check):
    """Shared fixed-K4 proof core; callers must supply a strict version/entry validator."""
    _require(mode in ('J', 'LI') and type(capacity) is int and capacity == 4, 'v2 requires common J/LI K4')
    _require(isinstance(record, dict) and isinstance(archive, dict)
        and set(archive) == {'schema_version', 'version', 'prepared_digest', 'cost_model', 'capacity',
                            'entries', 'search', 'export_contract'}, 'strict v2 archive schema differs')
    _require(type(archive['schema_version']) is int and archive['schema_version'] == 1
        and archive['version'] == expected_version and archive['capacity'] == capacity
        and record.get('status') == 'success' and record.get('selection') == mode
        and record.get('conversion_policy') == 'independent' and record.get('network_objective') == 'min-shfl'
        and archive['prepared_digest'] == record.get('prepared_digest') and _sha(archive['prepared_digest'])
        and archive['cost_model'] == record.get('cost_model')
        and archive['search'] == record.get('candidate_archive_search'), 'archive/search record identity differs')
    common = _common(record.get('common_static_contract'))
    profile_sha, model_id = _profile(record.get('cost_profile'), common, 'min-shfl')
    _require(record.get('cost_profile_sha256') == record.get('static_base_profile_sha256')
        == record.get('static_effective_profile_sha256') == profile_sha
        and record.get('cost_model') == record.get('static_base_model_id')
        == record.get('static_effective_model_id') == model_id
        and record.get('frequency_mode') == common['frequency_mode']
        and record.get('prepared_cost_model') == model_id
        and record.get('prepared_cost_profile_sha256') == profile_sha, 'search/prepared profile identity differs')
    entries = archive['entries'];_require(isinstance(entries, list) and 1 <= len(entries) <= capacity, 'native capacity differs')
    for i, e in enumerate(entries):
        entry_check(e, common);_require(e['index'] == i and e['prepared_digest'] == archive['prepared_digest'], 'entry index/domain differs')
        _require(dict(e['cost_profile'], network_objective='min-shfl') == record['cost_profile']
            and e['static_base_model_id'] == model_id and e['static_base_profile_sha256'] == profile_sha,
            'entry changed the frozen base profile beyond its objective')
    _require(len({e['complete_program_sha256'] for e in entries}) == len(entries), 'duplicate executable archive entries')
    search = archive['search'];total = _counters(search.get('total'));local = _counters(search.get('local_end'))
    _require(search.get('version') == expected_version and search.get('capacity') == capacity
        and search.get('retained') == len(entries) and search.get('limit') == 65536
        and search.get('refinement_limit') == 256 and search.get('local_terminal_refinement') == 256
        and search.get('local_terminal_probes') == (16384 if mode == 'J' else 65535)
        and total == record.get('total') and total['probes'] <= 65536 and total['refinement'] <= 256
        and all(local[n] <= total[n] for n in local)
        and local['probes'] <= search['local_terminal_probes'], 'fixed search/refinement budget differs')
    budget = record.get('budget_contract', {})
    _require(budget.get('kind') == 'shared-total-search-probes-v1' and budget.get('search_performed') is True
        and budget.get('limit') == 65536 and budget.get('refinement_limit') == 256
        and budget.get('used') == total['probes'] and budget.get('refinement_used') == total['refinement'],
        'record shared budget differs')
    # Successful LI may retain a verified paid sample after a typed recoverable
    # failure. Budget-exhausted incumbents normally remain complete-witness;
    # search_complete=False and local_cost_changed=True are both legitimate.
    local_record = search.get('local')
    _require(isinstance(local_record, dict)
        and local_record.get('selection_mode') == 'informed-producer-first'
        and local_record.get('local_objective') == 'own-operations-input-repairs-and-resources-v2'
        and _counters(local_record.get('total')) == local, 'local informed scope/counters differ')
    local_status = search.get('local_status')
    _require(local_status in ('complete-witness', 'no-local-witness'), 'untyped local status')
    if local_status == 'complete-witness' or mode == 'LI':
        _require(local_record.get('incumbent_full_score_budget_charged') is True,
                 'local incumbent full score was not charged')
        first = _counters(local_record.get('stage1'));second = _counters(local_record.get('stage2'))
        _require(all(first[n] + second[n] == local[n] for n in local),
                 'local stage counters do not reconcile')
    if local_status == 'complete-witness':
        _require(local_record.get('legal_return') is True, 'local witness is not a legal return')
    else:
        _require(local_record.get('legal_return') is False
            and local_record.get('failure_kind') in ('search-budget', 'unknown', 'unsupported', 'no-feasible')
            and isinstance(local_record.get('failure'), str) and bool(local_record['failure'])
            and not any(e['local_incumbent'] for e in entries),
                 'local fallback lacks explicit recoverable failure evidence')
    _identity(record)
    program = {n: record.get(n) for n in ('complete_assignment_sha256', 'network_objective', 'cost_model')}
    _require(_digest(program, True) == record.get('complete_program_sha256')
        and any(e['complete_program_sha256'] == record['complete_program_sha256'] for e in entries),
             'actual solver incumbent lost or program identity differs')
    if mode == 'J':
        joint = search.get('joint');joint_status = search.get('joint_status')
        _require(isinstance(joint, dict) and joint_status in ('complete-witness', 'no-joint-witness'),
                 'untyped joint status')
        phase_end = _counters(joint.get('total'))
        if joint_status == 'complete-witness':
            _require(joint.get('legal_return') is True, 'joint witness is not a legal return')
        else:
            _require(joint.get('legal_return') is False
                and joint.get('failure_kind') in ('search-budget', 'unknown', 'unsupported', 'no-feasible')
                and isinstance(joint.get('failure'), str) and bool(joint['failure'])
                and not any(e['joint_incumbent'] for e in entries),
                     'joint fallback lacks explicit recoverable failure evidence')
    else:
        phase_end = local
    for component in search.get('joint_components', []):
        _require(isinstance(component, dict), 'joint component schema differs')
        if component.get('has_seed_witness'):
            start = component.get('seed_start_cost');stop = component.get('selected_component_cost')
            _require(all(isinstance(c, dict) and c.get('saturated') is False
                and isinstance(c.get('cost'), list) and len(c['cost']) == 9
                and all(_nat(v) for v in c['cost']) for c in (start, stop))
                and stop['cost'] <= start['cost'], 'constructed joint witness lost')
    v = search.get('network_variant');_require(isinstance(v, dict), 'network template receipt required')
    _require(all(_nat(v.get(n)) for n in ('reserved_score_probes', 'layout_search_limit', 'native_capacity',
        'budget_limit_restored', 'budget_refinement_unchanged', 'variant_plan_checks', 'variant_scores',
        'unused_reserved_score_probes')), 'strict network template counters required')
    _require(v.get('contract') == CONTRACT and v.get('reserved_score_probes') == 1
        and v.get('layout_search_limit') == 65535 and v.get('native_capacity') == capacity
        and v.get('objective') == 'min-work' and v.get('budget_limit_restored') == 65536
        and v.get('budget_refinement_unchanged') == 256
        and v.get('resource_proxy_is_physical_register_prediction') is False
        and v.get('variant_plan_checks') == 1 and _nat(v.get('plan_check_wall_ns'))
        and type(v.get('plan_comparison_proven')) is bool and type(v.get('plan_changed')) is bool,
        'network template contract differs')
    variants = [e for e in entries if e['network_variant']];autos = [e for e in entries if not e['network_variant']]
    _require(bool(autos) and len({e['complete_assignment_sha256'] for e in autos}) == len(autos), 'auto geometry witnesses differ')
    _require(all(e['probes_at_observation'] <= total['probes'] and e['refinement_at_observation'] <= total['refinement'] for e in entries)
        and all(e['probes_at_observation'] <= 65535 for e in autos), 'unpaid archive observation')
    order = v.get('auto_original_indices');first = v.get('static_first_auto_index')
    _require(isinstance(order, list) and 1 <= len(order) <= capacity and all(_nat(i) for i in order) and sorted(order) == list(range(len(order)))
        and _nat(first) and first < len(autos) and first < 3
        and autos[first]['cost']['cost'] == min(e['cost']['cost'] for e in autos), 'static-first auto witness differs')
    pins = v.get('pins');_require(isinstance(pins, list) and 1 <= len(pins) <= 3, 'required auto pins differ')
    expected_slots = {i for i, e in enumerate(autos) if e['local_incumbent'] or e['joint_incumbent'] or i == first}
    seen = set()
    for p in pins:
        _require(isinstance(p, dict) and set(p) == {'original_index', 'retained_slot', 'local_incumbent',
            'joint_incumbent', 'static_first', 'assignment'}, 'pin identity schema differs')
        _require(all(type(p[n]) is bool for n in ('local_incumbent', 'joint_incumbent', 'static_first'))
            and _nat(p['original_index']), 'strict pin metadata required')
        slot = p['retained_slot'];_require(_nat(slot) and slot < len(autos) and slot not in seen, 'pin retained slot differs');seen.add(slot)
        e = autos[slot]
        _require(p['original_index'] == order[slot] and p['assignment'] == e['assignment']
            and p['local_incumbent'] == e['local_incumbent'] and p['joint_incumbent'] == e['joint_incumbent']
            and p['static_first'] == (slot == first), 'pin was replaced or relabeled')
    _require(seen == expected_slots, 'required auto witness missing')
    _require(search.get('local_status') != 'complete-witness' or any(e['local_incumbent'] for e in autos), 'local incumbent discarded')
    _require(mode != 'J' or search.get('joint_status') != 'complete-witness' or any(e['joint_incumbent'] for e in autos), 'joint incumbent discarded')
    _require(mode != 'LI' or search.get('joint_status') == 'not-requested-local-control', 'LI received extra joint search')
    if v['plan_comparison_proven']:
        _require(_nat(v.get('compared_shuffle_boundaries')) and _plans(v.get('boundary_plans'), v['compared_shuffle_boundaries']) == v['plan_changed'],
                 'actual objective plan comparison differs')
    else:
        _require(not v['plan_changed'] and isinstance(v.get('plan_comparison_reason'), str)
            and bool(v['plan_comparison_reason']), 'unproven transport cannot become a variant')
    changed = v['plan_comparison_proven'] and v['plan_changed']
    # The optional network rescore happens AFTER either solver's own receipt.
    _require(all(total[n] == phase_end[n] + (int(changed) if n in ('probes', 'complete_scores') else 0)
                 for n in total), 'solver/variant cumulative counters do not reconcile')
    _require(v.get('variant_scores') == int(changed) and v.get('unused_reserved_score_probes') == int(not changed),
             'variant rescore accounting differs')
    if changed:
        _require(len(variants) == 1 and v.get('status') == ('min-work-replaces-fourth-auto' if len(order) == capacity else 'min-work-added-within-K4')
            and v.get('variant_slot') == len(entries) - 1 and entries[-1] is variants[0]
            and _nat(v.get('score_wall_ns')) and total['complete_scores'] >= 1,
            'genuine variant must occupy exactly one charged native slot')
        variant = variants[0];base = autos[first]
        _require(_identity(variant) == _identity(base) and variant['origins'] == ['fixed-min-work-template']
            and not variant['local_incumbent'] and not variant['joint_incumbent']
            and variant['probes_at_observation'] == total['probes'], 'variant lacks retained auto counterpart')
        if len(order) == capacity:
            _require(len(entries) == capacity and len(autos) == 3
                and v.get('replaced_auto_original_index') == order[-1]
                and isinstance(v.get('replaced_auto_assignment'), list)
                and all(v['replaced_auto_assignment'] != p['assignment'] for p in pins), 'fourth auto replacement differs')
        else:
            _require(len(entries) == len(order) + 1 and 'replaced_auto_original_index' not in v,
                     'partial pool gained extra layout candidates')
    else:
        _require(not variants and len(entries) == len(order) and total['probes'] <= 65535
            and v.get('status') == ('identical-plans-keep-auto' if v['plan_comparison_proven'] else 'unavailable-plan-proof-keep-auto')
            and 'variant_slot' not in v and 'replaced_auto_original_index' not in v, 'unused reserved slot changed the auto pool')
    return entries


def check_archive(record, archive, mode, capacity=4):
    """Validate the original strict v2 archive contract without admitting v3 fields."""
    return _check_archive_common(record, archive, mode, capacity,
        expected_version=VERSION, entry_check=_entry)


def search_args(capacity=4):
    _require(type(capacity) is int and capacity == 4, 'v2 native capacity must be K4')
    return ['--candidate-archive=4', '--network-archive-v2=true']


def replay_args(entry):
    common = _common(entry.get('common_static_contract'));_entry(entry, common)
    return ['--network-objective=' + entry['network_objective']]


def check_replay(record, entry):
    """Verify a fixed entry was reconstructed with zero new search probes."""
    replay_args(entry)
    _require(isinstance(record, dict) and record.get('status') == 'success'
        and record.get('selection_bypassed') is True and record.get('network_objective') == entry['network_objective']
        and record.get('conversion_policy') == 'independent' and record.get('prepared_digest') == entry['prepared_digest']
        and record.get('cost_model') == entry['cost_model'] and record.get('cost_profile_sha256') == entry['cost_profile_sha256']
        and record.get('cost_profile') == entry['cost_profile'] and record.get('common_static_contract') == entry['common_static_contract'],
        'fixed replay program/profile differs')
    _require(_identity(record) == _identity(entry) and record.get('cost') == entry['cost'], 'fixed replay geometry/score differs')
    program = {n: record[n] for n in ('complete_assignment_sha256', 'network_objective', 'cost_model')}
    _require(record.get('complete_program_sha256') == entry['complete_program_sha256'] == _digest(program, True),
             'fixed replay complete program digest differs')
    b = record.get('budget_contract', {})
    _require(all(_nat(b.get(n)) for n in ('limit', 'refinement_limit', 'used', 'refinement_used'))
        and b.get('search_performed') is False and b.get('limit') == 65536
        and b.get('refinement_limit') == 256 and b.get('used') == 0 and b.get('refinement_used') == 0,
        'fixed replay performed uncharged search')
    for n in ('static_base_model_id', 'static_base_profile_sha256', 'static_effective_model_id', 'static_effective_profile_sha256'):
        _require(record.get(n) == entry[n], 'fixed replay objective comparison identity differs')
    return True
