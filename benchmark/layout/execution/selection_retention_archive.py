"""Strict, timing-free validation for the paid-feature K4 retention experiment.

The common network/assignment/profile/budget proof is shared with v2. New
receipts are checked before that proof; they are never silently discarded or
relabeled into v2. The receipt audits source-side counts and bounded retention,
not native latency or physical register allocation.
"""
from __future__ import annotations
from collections import Counter
import json
import re
import network_archive as na

VERSION = 'shared-network-candidate-archive-v3'
POLICY = 'paid-feature-frontier-v1'
STATIC_POLICY = 'paid-static-feature-frontier-v1'
ROUTING_POLICY = 'calibrated-source-grammar-v1'
ROUTING_CONTRACT = 'calibrated-source-grammar-retention-routing-v1'
SOURCE_GRAMMAR_CONTRACT = 'scalar-source-grammar-archive-eligibility-v1'
SOURCE_GRAMMAR = 'static-rank2-f32-singleCTA128-Global-rowmajor-pure-load-mul-sub-exp-splat-view-convert-finalstore-v1'
STATIC_CONTRACT = 'paid-static-feature-reservoir-K4-v1'
CONTRACT = 'paid-gf2-structure-reservoir-K4-v1'
STRUCTURE_CONTRACT = 'gf2-ownership-cut-v1'
STRUCTURAL_FRONTIER = 'uncovered communication cuts: main score versus cross-thread boundary count; ownership Grassmann/nullity distance to fixed pins'
AXES = na.FEATURES + ('logical_register_proxy', 'logical_shared_byte_proxy',
                     'logical_replication_proxy')
ENTRY_FIELDS = na.ENTRY_FIELDS | {'retention_reason', 'retention_structure'}
RETENTION_FIELDS = set(('contract policy reservoir_capacity reservoir_peak '
    'observer_working_set_peak pending_incoming_capacity feature_axes '
    'observations_with_recognized_features feature_values_are_physical_register_predictions '
    'feature_zero_contract identity_duplicate_scope extra_candidate_scores '
    'certificate_rebuilds discard_reason_counts discard_examples structure_contract '
    'structure_proven_observations structure_unknown_observations structure_layout_cache_hits '
    'structure_layout_cache_misses structure_distance_cache_hits structure_distance_cache_misses '
    'structural_frontier').split())
DISCARD_REASONS = {'legacy-main-cost-frontier', 'bounded-reservoir-dominated',
    'bounded-reservoir-diversity', 'legacy-hamming-capacity',
    'final-pin-identity-duplicate', 'final-native-capacity'}
EXAMPLE_FIELDS = {'reason', 'assignment_stable_ordinals', 'feature_values',
                  'origin', 'main_score', 'probes_at_observation'}
ZERO_CONTRACT = 'sparse model counts or unmodeled proxies; not proof of free native work'
IDENTITY_SCOPE = 'exact assignments in live reservoir; equal features never deduplicate programs'


def _require(condition, reason):
    na._require(condition, reason)


def _uint64(value):
    return type(value) is int and 0 <= value <= (1 << 64) - 1


def _basis(columns):
    pivots = {}
    for value in columns:
        while value:
            bit = value.bit_length() - 1
            if bit in pivots:
                value ^= pivots[bit]
            else:
                pivots[bit] = value
                break
    for bit in sorted(pivots):
        for other in sorted(pivots):
            if other > bit and (pivots[other] >> bit) & 1:
                pivots[other] ^= pivots[bit]
    return [pivots[bit] for bit in sorted(pivots)]


def _decode_columns(attribute):
    if not attribute.startswith('#frisk.distributed<') or '#frisk.bit_linear<' not in attribute:
        return None
    names = re.search(r'inputs = \[([^]]*)\]', attribute)
    widths = re.search(r'input_bits = \[([^]]*)\]', attribute)
    outputs = re.search(r'output_bits = \[([^]]*)\]', attribute)
    matrix = re.search(r'matrix = dense<(.*?)> : tensor<(\d+)x(\d+)xi1>', attribute)
    _require(all(x is not None for x in (names,widths,outputs,matrix)), 'v3 cannot decode claimed bit-linear matrix')
    names = json.loads('[' + names[1] + ']')
    widths = [int(n.strip()) for n in widths[1].split(',') if n.strip()]
    logical = sum(int(n.strip()) for n in outputs[1].split(',') if n.strip())
    _require(len(names) == len(widths) and len(set(names)) == len(names)
        and all(n in ('register','lane','warp','warp_group','cta') for n in names)
        and all(0 <= w <= 256 for w in widths) and sum(widths) <= 256
        and 0 <= logical <= 256, 'v3 bit-linear matrix axes differ')
    if logical > 64:
        return None
    hardware = sum(widths);content = matrix[1]
    _require((int(matrix[2]),int(matrix[3])) == (logical,hardware), 'v3 matrix extent differs')
    if content.startswith('"0x') and content.endswith('"'):
        raw = int.from_bytes(bytes.fromhex(content[3:-1]), 'little')
    elif content in ('true','false'):
        raw = (1 << (logical*hardware))-1 if content == 'true' else 0
    else:
        bits = re.findall(r'\b(true|false)\b', content)
        _require(len(bits) == logical*hardware, 'v3 unsupported dense matrix spelling')
        raw = sum((b == 'true') << i for i,b in enumerate(bits))
    _require(raw >> (logical*hardware) == 0, 'v3 nonzero matrix padding')
    columns = [sum(((raw >> (row*hardware+col)) & 1) << row
                   for row in range(logical)) for col in range(hardware)]
    grouped = dict(register=[],lane=[],warp=[],warp_group=[],cta=[]);position = 0
    for name,width in zip(names,widths):
        grouped[name] = columns[position:position+width];position += width
    return logical,[grouped[n] for n in ('register','lane','warp','warp_group','cta')]


def _structure(entry):
    value = entry.get('retention_structure')
    fields = {'contract','proven','reason','communication_signature',
              'cross_thread_boundaries','ownership','boundaries'}
    _require(isinstance(value,dict) and set(value) == fields
        and value['contract'] == STRUCTURE_CONTRACT and type(value['proven']) is bool
        and isinstance(value['reason'],str) and isinstance(value['ownership'],list)
        and isinstance(value['boundaries'],list), 'v3 structure schema differs')
    owners = {};assignment = {x['variable']:x['candidate']['attribute'] for x in entry['assignment']}
    owner_fields = {'variable','proven','active','logical_bits','register_nullity',
                    'carrier_columns','register_basis','warp_basis'}
    for owner in value['ownership']:
        _require(isinstance(owner,dict) and set(owner) == owner_fields
            and na._nat(owner['variable']) and owner['variable'] in assignment
            and owner['variable'] not in owners and type(owner['proven']) is bool
            and type(owner['active']) is bool and na._nat(owner['logical_bits'])
            and na._nat(owner['register_nullity']) and isinstance(owner['carrier_columns'],list)
            and len(owner['carrier_columns']) == 5
            and all(isinstance(c,list) and len(c) <= 256 and all(_uint64(n) for n in c)
                    for c in owner['carrier_columns']), 'v3 ownership schema differs')
        decoded = _decode_columns(assignment[owner['variable']])
        if owner['proven']:
            _require(decoded is not None and owner['logical_bits'] == decoded[0]
                and owner['carrier_columns'] == decoded[1], 'v3 columns do not bind to complete layout attributes')
            registers = _basis(decoded[1][0]);warp = _basis(decoded[1][0]+decoded[1][1])
            _require(owner['register_basis'] == registers and owner['warp_basis'] == warp
                and owner['register_nullity'] == len(decoded[1][0])-len(registers),
                'v3 canonical ownership image/kernel differs')
        else:
            _require(decoded is None and owner['logical_bits'] == owner['register_nullity'] == 0
                and owner['carrier_columns'] == [[] for _ in range(5)]
                and owner['register_basis'] == owner['warp_basis'] == [],
                'v3 unavailable ownership silently supplied invented algebra')
        owners[owner['variable']] = owner
    expected = {vid for vid,attribute in assignment.items() if attribute.startswith('#frisk.distributed<')}
    _require(expected <= set(owners), 'v3 omitted distributed ownership evidence')
    boundary_fields = {'constraint','source','target','register_defect','warp_defect','scope'}
    signatures = [];seen = set();cross = 0
    for edge in value['boundaries']:
        _require(isinstance(edge,dict) and set(edge) == boundary_fields
            and all(na._nat(edge[n]) for n in boundary_fields)
            and edge['constraint'] not in seen and edge['scope'] <= 3,
            'v3 communication boundary schema differs')
        seen.add(edge['constraint']);source,target = owners.get(edge['source']),owners.get(edge['target'])
        if source and target and source['proven'] and target['proven'] and source['logical_bits'] == target['logical_bits']:
            a,b = source['carrier_columns'],target['carrier_columns']
            register_need = list(b[0]);warp_need = b[0]+b[1]
            for carrier in range(1,5):
                diff = [(a[carrier][i] if i < len(a[carrier]) else 0) ^
                        (b[carrier][i] if i < len(b[carrier]) else 0)
                        for i in range(max(len(a[carrier]),len(b[carrier])))]
                register_need += diff
                if carrier > 1:warp_need += diff
            rd = len(_basis(source['register_basis']+register_need))-len(source['register_basis'])
            wd = len(_basis(source['warp_basis']+warp_need))-len(source['warp_basis'])
            scope = 0 if rd == 0 else 1 if wd == 0 else 2
            _require((edge['register_defect'],edge['warp_defect'],edge['scope']) == (rd,wd,scope),
                     'v3 communication inclusion defect differs')
        else:
            _require(edge['scope'] == 3 and edge['register_defect'] == edge['warp_defect'] == 0,
                     'v3 unknown boundary supplied a free known scope')
        signatures.append(edge['scope']);cross += edge['scope'] in (1,2)
    proven = all(o['proven'] for o in owners.values()) and all(n != 3 for n in signatures)
    _require(value['communication_signature'] == signatures
        and na._nat(value['cross_thread_boundaries']) and value['cross_thread_boundaries'] == cross
        and value['proven'] is proven
        and ((value['reason'] == '') if proven else bool(value['reason'])),
        'v3 communication-cut/proof status differs')
    return value


def _entry(entry, common, *, policy=None):
    na._entry(entry, common, fields=ENTRY_FIELDS)
    structure = _structure(entry)
    expected = ('fixed-min-work-template' if entry['network_variant'] else
        'local-incumbent-pin' if entry['local_incumbent'] else
        'joint-incumbent-pin' if entry['joint_incumbent'] else
        'paid-gf2-cut-frontier-diversity' if structure['proven'] else
        'paid-feature-frontier-diversity')
    if policy == STATIC_POLICY and not (entry['network_variant'] or entry['local_incumbent'] or entry['joint_incumbent']):expected = 'paid-feature-frontier-diversity'
    valid = entry['retention_reason'] == expected
    if policy is None and not (entry['network_variant'] or entry['local_incumbent'] or entry['joint_incumbent']):valid |= entry['retention_reason'] == 'paid-feature-frontier-diversity'
    _require(valid, 'v3 retention reason/pin differs')
    if entry['local_incumbent'] or entry['joint_incumbent']:
        origins = (["local-incumbent"] if entry['local_incumbent'] else [])
        origins += (["joint-incumbent"] if entry['joint_incumbent'] else [])
        _require(entry['origins'] == origins, 'v3 incumbent provenance differs')
    elif not entry['network_variant']:
        _require(entry['origins'] in (['local-completion'], ['joint-search']),
                 'v3 optional candidate is not a paid search observation')
    return entry


def _retention(search, entries, mode, policy):
    legacy = policy == 'legacy-hamming-v2'
    receipt = search.get('retention')
    _require(isinstance(receipt, dict) and set(receipt) == RETENTION_FIELDS,
             'strict v3 retention receipt schema differs')
    _require(receipt['contract'] == ('paid-fullcost-hamming-v2' if legacy else CONTRACT if policy == POLICY else STATIC_CONTRACT) and receipt['policy'] == policy
        and receipt['feature_axes'] == list(AXES)
        and receipt['reservoir_capacity'] == (4 if legacy else 32)
        and receipt['pending_incoming_capacity'] == 1
        and receipt['feature_values_are_physical_register_predictions'] is False
        and receipt['feature_zero_contract'] == ZERO_CONTRACT
        and receipt['identity_duplicate_scope'] == IDENTITY_SCOPE
        and receipt['structure_contract'] == STRUCTURE_CONTRACT
        and receipt['structural_frontier'] == (STRUCTURAL_FRONTIER if policy == POLICY else 'disabled; static-only feature ablation'),
        'v3 feature/reservoir/identity contract differs')
    count_fields = ('reservoir_capacity', 'reservoir_peak', 'observer_working_set_peak',
        'pending_incoming_capacity', 'observations_with_recognized_features',
        'extra_candidate_scores', 'certificate_rebuilds', 'structure_proven_observations',
        'structure_unknown_observations', 'structure_layout_cache_hits', 'structure_layout_cache_misses',
        'structure_distance_cache_hits', 'structure_distance_cache_misses')
    _require(all(na._nat(receipt[n]) for n in count_fields)
        and receipt['extra_candidate_scores'] == 0, 'v3 retention introduced uncharged scores')
    _require(all(na._nat(search.get(n)) for n in ('paid_complete_observations',
        'observed_duplicates', 'frontier_rejections', 'capacity_discards')),
        'v3 paid observation counters differ')
    observed, duplicates = search['paid_complete_observations'], search['observed_duplicates']
    total = na._counters(search.get('total'));local = na._counters(search.get('local_end'))
    _require(duplicates <= observed <= total['complete_scores'],
        'v3 observations exceed paid complete scores')
    admitted = observed - duplicates
    _require(receipt['structure_proven_observations'] + receipt['structure_unknown_observations'] == (0 if legacy else admitted),
        'retention structural known/unknown admission counts differ')
    if legacy:
        _require(receipt['reservoir_peak'] == min(receipt['observer_working_set_peak'],4)
            and 0 <= receipt['observer_working_set_peak'] <= min(admitted,5)
            and (admitted == 0 or receipt['observer_working_set_peak'] > 0)
            and all(receipt[k] == 0 for k in ('structure_layout_cache_hits','structure_layout_cache_misses',
                'structure_distance_cache_hits','structure_distance_cache_misses')),
            'legacy bounded reservoir or disabled structure counters differ')
    else:
        _require(receipt['reservoir_peak'] == min(admitted,32)
            and receipt['observer_working_set_peak'] == min(admitted,33),
            'v3 resident/pending observation bound differs')
    _require(receipt['observations_with_recognized_features'] <= admitted,
             'retention feature observations exceed paid admissions')
    reasons = receipt['discard_reason_counts']
    _require(isinstance(reasons, dict) and set(reasons) == DISCARD_REASONS
        and all(na._nat(n) for n in reasons.values())
        and reasons['legacy-main-cost-frontier'] == search['frontier_rejections']
        and (legacy or reasons['legacy-main-cost-frontier'] == 0)
        and (legacy or reasons['legacy-hamming-capacity'] == 0)
        and (not legacy or reasons['bounded-reservoir-dominated'] == reasons['bounded-reservoir-diversity'] == 0)
        and search['capacity_discards'] == reasons['bounded-reservoir-dominated']
            + reasons['bounded-reservoir-diversity'] + reasons['legacy-hamming-capacity'],
        'retention discard reason accounting differs')
    variant = search['network_variant'];auto_before = len(variant['auto_original_indices'])
    pin_count = sum(e['local_incumbent'] or e['joint_incumbent'] for e in entries)
    optional_selected = auto_before - pin_count
    _require(0 <= optional_selected <= 4
        and receipt['certificate_rebuilds'] == auto_before + optional_selected,
        'v3 candidate reconstruction introduced unexplained evaluations')
    remaining = admitted - search['capacity_discards'] - search['frontier_rejections']
    _require(0 <= remaining <= (4 if legacy else 32) and remaining == optional_selected
        + reasons['final-pin-identity-duplicate'] + reasons['final-native-capacity'],
        'v3 observation/eviction/final selection counts do not reconcile')
    for entry in entries:
        if mode == 'LI':
            _require('joint-search' not in entry['origins'], 'LI borrowed an independent joint search')
        if 'local-completion' in entry['origins']:
            _require(0 < entry['probes_at_observation'] <= local['probes']
                and entry['refinement_at_observation'] <= local['refinement'],
                'v3 local sample was not paid in its own phase')
        elif 'joint-search' in entry['origins']:
            _require(local['probes'] < entry['probes_at_observation'] <= total['probes'],
                'v3 joint sample was not paid after local completion')
    examples = receipt['discard_examples'];examples_by_reason = Counter()
    noted = reasons['bounded-reservoir-dominated'] + reasons['bounded-reservoir-diversity'] + reasons['final-native-capacity']
    _require(isinstance(examples, list) and len(examples) == min(noted, 16),
             'v3 bounded discard witness coverage differs')
    width = len(entries[0]['assignment'])
    for example in examples:
        _require(isinstance(example, dict) and set(example) == EXAMPLE_FIELDS,
                 'v3 discard witness schema differs')
        reason = example['reason']
        _require(reason in ('bounded-reservoir-dominated', 'bounded-reservoir-diversity',
            'final-native-capacity') and reasons[reason] > 0,
            'v3 discard witness lacks a counted reason')
        examples_by_reason[reason] += 1
        key, features = example['assignment_stable_ordinals'], example['feature_values']
        _require(isinstance(key, list) and len(key) == width and all(na._nat(n) for n in key)
            and isinstance(features, list) and len(features) == len(AXES)
            and all(_uint64(n) for n in features) and na._nat(example['main_score'])
            and example['origin'] in ('local-completion', 'joint-search')
            and na._nat(example['probes_at_observation'])
            and 0 < example['probes_at_observation'] <= total['probes'],
            'v3 discard witness is not a finite paid complete assignment')
        if example['origin'] == 'local-completion':
            _require(example['probes_at_observation'] <= local['probes'],
                     'v3 discarded local witness escaped its budget')
        else:
            _require(mode == 'J' and example['probes_at_observation'] > local['probes'],
                     'v3 discarded joint witness escaped its budget')
    _require(all(count <= reasons[reason] for reason, count in examples_by_reason.items()),
             'v3 discard examples exceed reason counters')
    audit = search.get('paid_observation_audit')
    if audit is not None:
        _require(isinstance(audit,dict) and set(audit) == {'contract','selection_influence','bound','observed','entries'}
            and audit['contract'] == 'paid-full-assignment-observer-ledger-v1'
            and audit['selection_influence'] == 'none; append-only diagnostics; no scores'
            and audit['bound'] == 65536 and audit['observed'] == observed
            and isinstance(audit['entries'],list) and len(audit['entries']) == observed,
            'v3 append-only paid audit contract differs')
        previous = 0
        for event in audit['entries']:
            _require(isinstance(event,dict) and set(event) == {'assignment_stable_ordinals','feature_values','cost','origin','probes_at_observation','refinement_at_observation'},
                     'v3 paid event schema differs')
            key = event['assignment_stable_ordinals'];cost = event['cost']
            _require(isinstance(key,list) and len(key) == width and all(na._nat(n) for n in key)
                and isinstance(cost,dict) and set(cost) == {'cost','features','saturated'}
                and cost['saturated'] is False and isinstance(cost['cost'],list) and len(cost['cost']) == 9
                and all(na._nat(n) for n in cost['cost']) and isinstance(cost['features'],dict)
                and set(cost['features']) <= set(na.FEATURES) and all(na._nat(n) for n in cost['features'].values()),
                'v3 paid audit event is not a complete finite static score')
            expected_features = [cost['features'].get(n,0) for n in na.FEATURES] + [cost['cost'][i] for i in (4,5,6)]
            _require(event['feature_values'] == expected_features
                and na._nat(event['probes_at_observation']) and previous <= event['probes_at_observation'] <= total['probes']
                and event['probes_at_observation'] > 0 and na._nat(event['refinement_at_observation'])
                and event['refinement_at_observation'] <= total['refinement'],
                'v3 audit features/probe sequence differs')
            previous = event['probes_at_observation']
            if event['origin'] == 'local-completion':
                _require(event['probes_at_observation'] <= local['probes'] and event['refinement_at_observation'] <= local['refinement'],
                         'v3 audit local event escaped phase budget')
            else:
                _require(event['origin'] == 'joint-search' and mode == 'J' and event['probes_at_observation'] > local['probes'],
                         'v3 audit joint event escaped phase budget')
    return receipt


def _routing(search):
    route = search.get('retention_resolution')
    _require(isinstance(route, dict) and set(route) == {'contract','requested','effective',
        'resolved_before_search','resolver_calls','additional_searches','native_output_guaranteed','gate'},
        'strict source-grammar routing schema differs')
    _require(route['contract'] == ROUTING_CONTRACT and route['requested'] == ROUTING_POLICY
        and route['resolved_before_search'] is True and type(route['resolver_calls']) is int
        and route['resolver_calls'] == 1 and type(route['additional_searches']) is int
        and route['additional_searches'] == 0 and route['native_output_guaranteed'] is False,
        'routing reran search or claimed native-output proof')
    gate = route['gate']
    _require(isinstance(gate,dict) and set(gate) == {'contract','status','reason',
        'source_eligibility_only','native_output_guaranteed','checked_operations',
        'checked_variables','checked_candidates','kernel_count','parallel_count','threads',
        'grid','source_target','execution_domain','grammar'}, 'source grammar receipt schema differs')
    _require(gate['contract'] == SOURCE_GRAMMAR_CONTRACT and gate['status'] in ('eligible','unknown')
        and isinstance(gate['reason'],str) and gate['source_eligibility_only'] is True
        and gate['native_output_guaranteed'] is False and isinstance(gate['grammar'],str)
        and gate['grammar'] == SOURCE_GRAMMAR and all(na._nat(gate[key]) for key in ('checked_operations',
            'checked_variables','checked_candidates','kernel_count','parallel_count','threads'))
        and isinstance(gate['grid'],list) and all(na._nat(n) for n in gate['grid'])
        and isinstance(gate['source_target'],str) and isinstance(gate['execution_domain'],str),
        'source grammar eligibility evidence differs')
    eligible = gate['status'] == 'eligible'
    _require(route['effective'] == (POLICY if eligible else 'legacy-hamming-v2'),
             'unknown source grammar did not route to complete legacy')
    if eligible:
        _require(gate['reason'] == '' and gate['checked_operations'] > 0
            and gate['checked_variables'] > 0 and gate['checked_candidates'] > 0
            and gate['kernel_count'] == gate['parallel_count'] == 1
            and gate['threads'] == 128 and gate['grid'] == [1]
            and gate['source_target'] == 'sm_90a'
            and gate['execution_domain'] == 'sm90-scalar-f32-dag-v1',
            'eligible source escaped calibrated source grammar')
    else:
        _require(bool(gate['reason']), 'unknown grammar requires a rejection reason')
    return route['effective']


def check_archive(record, archive, mode, capacity=4, *, expected_policy=POLICY):
    if expected_policy == ROUTING_POLICY:
        effective = _routing(archive.get('search', {}))
        # Dispatch the actual archived representation. No v3 fields are removed
        # to make the legacy branch pass, and no v2 receipt is relabeled as v3.
        if effective == 'legacy-hamming-v2':
            entries = na.check_archive(record, archive, mode, capacity)
            _retention(archive['search'], entries, mode, effective)
            return entries
        return check_archive(record, archive, mode, capacity, expected_policy=effective)
    _require(expected_policy in (POLICY,STATIC_POLICY), 'unknown v3 retention policy')
    if 'retention_resolution' in archive.get('search', {}):
        _require(_routing(archive['search']) == expected_policy, 'route differs from requested effective v3 policy')
    entries = na._check_archive_common(record, archive, mode, capacity,
        expected_version=VERSION, entry_check=lambda e,c:_entry(e,c,policy=expected_policy))
    _retention(archive['search'], entries, mode, expected_policy)
    return entries


def search_args(capacity=4, *, policy=POLICY):
    _require(policy in (POLICY,STATIC_POLICY,ROUTING_POLICY), 'unknown retention recipe')
    return na.search_args(capacity) + ['--archive-retention=' + policy]


def replay_args(entry):
    if 'retention_reason' not in entry:
        return na.replay_args(entry)
    _entry(entry, na._common(entry.get('common_static_contract')))
    return ['--network-objective=' + entry['network_objective']]


def check_replay(record, entry):
    if 'retention_reason' not in entry:
        return na.check_replay(record, entry)
    replay_args(entry)
    # Every extra field has already passed the v3 validator. The fixed replay
    # itself uses the shared assignment/profile/program proof and zero budget.
    base = {key: value for key, value in entry.items() if key not in ('retention_reason','retention_structure')}
    return na.check_replay(record, base)
