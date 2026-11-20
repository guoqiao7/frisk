"""Fixed fresh and observed cohorts for GF(2) decision refinement.

Pure fixture construction only: no CUDA, subprocesses, fitting or timing reads.
Numerics and source generation retain the previously accepted implementation.
"""
import math
from copy import deepcopy
import partial_cost_fixture as previous
from selection_quality_fixture import program_identity

SEED = 20261106
STAGE = 'gf2-decision-refinement-qualification'
GPU_UUID = previous.GPU_UUID
source = previous.source
inputs = previous.inputs
numerical_cases = previous.numerical_cases
reference = previous.reference
offsets = previous.offsets
check_numerics = previous.check_numerics
safety_input_identity = previous.safety_input_identity
digest = previous.digest

# Scheme A is fixed before capability preflight and any new GPU observations.
FRESH_COHORT = (
    ('diamond', 'diamond', (256, 128), 'none'),
    ('remat', 'remat', (512, 64), 'none'),
    ('conflict', 'conflict', (128, 256), 'none'),
    ('mixed', 'no-conflict', (1024, 64), 'mixed'),
    ('div', 'no-conflict', (256, 128), 'div6'),
    ('exp', 'no-conflict', (256, 128), 'exp1'),
)
FACTORIAL_RULES = (
    ('interval-only', False, False),
    ('gf2-tie', True, False),
    ('frame-only', False, True),
    ('tie-frame', True, True),
)
POLICY_LABELS = tuple(mode + '-' + rule for mode in ('joint', 'local')
                      for rule in ('interval-only', 'gf2-tie', 'frame-only', 'tie-frame', 'affine-frame')) + (
    'legacy-static', 'joint-static', 'local-static')
MAIN_POLICIES = ('joint-interval-only', 'joint-tie-frame', 'joint-affine-frame')


def workload_identity(spec):
    return (spec['case_id'], spec.get('tail', 'none'), spec.get('abi_stratum', 'weak-vector'),
            spec.get('dtype', 'float32'), spec.get('threads', 128), math.prod(spec['shape']))


def historical_specifications():
    # previous.specifications() is the sealed R3 cohort; its former holdouts are
    # now observed. Do not change the old fixture or reinterpret its protocol.
    identities = {program_identity(s) for s in previous.historical_specifications() + previous.specifications()}
    return [dict(case_id=k[0], shape=list(k[1]), tail=k[2], abi_stratum=k[3], dtype=k[4], threads=k[5])
            for k in sorted(identities)]


def fresh_specifications():
    return [dict(key='holdout-refined-' + name + '-' + str(shape[0]) + 'x' + str(shape[1]),
                 split='holdout', role='fresh-performance-holdout', graph_name=name,
                 case_id=case, shape=list(shape), tail=tail, abi_stratum='weak-vector',
                 dtype='float32', threads=128)
            for name, case, shape, tail in FRESH_COHORT]


def regression_specifications():
    result = []
    for old in previous.specifications():
        s = deepcopy(old)
        if old['case_id'] == 'attention':
            s.update(key='regression-r3-attention-' + old['abi_stratum'], split='observed-anchor',
                     role='observed-R3-ABI-regression-only')
        else:
            s.update(key='regression-r3-' + old['graph_name'] + '-' + str(old['shape'][0]) + 'x' + str(old['shape'][1]),
                     split='observed-regression', role='observed-R3-functional-and-performance-regression-only')
        result.append(s)
    return result


def specifications():
    rows = fresh_specifications() + regression_specifications()
    validate_specifications(rows)
    return rows


def validate_specifications(rows):
    if not isinstance(rows, list) or len(rows) != 15 or any(not isinstance(s, dict) for s in rows):
        raise ValueError('complete six fresh and nine R3 regression programs required')
    required = ('key', 'split', 'role', 'graph_name', 'case_id', 'shape', 'tail', 'abi_stratum', 'dtype', 'threads')
    if any(any(k not in s for k in required) for s in rows):
        raise ValueError('concrete complete program metadata required')
    for s in rows:
        if not isinstance(s['key'], str) or not s['key'] or not isinstance(s['shape'], list) or len(s['shape']) != 2:
            raise ValueError('concrete key and two-dimensional shape required')
        if any(type(x) is not int or x <= 0 or x & (x - 1) for x in s['shape']):
            raise ValueError('positive dyadic dimensions required')
        if type(s['threads']) is not int or s['threads'] != 128:
            raise ValueError('registered 128-thread domain required')
    if len({s['key'] for s in rows}) != 15 or len({program_identity(s) for s in rows}) != 15:
        raise ValueError('duplicate program or key')
    fresh = [s for s in rows if s['split'] == 'holdout']
    regressions = [s for s in rows if s['split'] in ('observed-regression', 'observed-anchor')]
    if len(fresh) != 6 or len(regressions) != 9 or len(fresh) + len(regressions) != 15:
        raise ValueError('six fresh and all nine observed R3 programs required')
    families = {(case, tail, name) for name, case, shape, tail in FRESH_COHORT}
    if {(s['case_id'], s['tail'], s['graph_name']) for s in fresh} != families:
        raise ValueError('all six unchanged algorithm families required')
    history = historical_specifications()
    identities = {program_identity(s) for s in history}
    flattened = {workload_identity(s) for s in history}
    if len({workload_identity(s) for s in fresh}) != 6:
        raise ValueError('duplicate flattened fresh program')
    for s in fresh:
        if program_identity(s) in identities or workload_identity(s) in flattened:
            raise ValueError('observed complete or flattened workload cannot be fresh')
        if s['dtype'] != 'float32' or s['abi_stratum'] != 'weak-vector' or math.prod(s['shape']) > 65536:
            raise ValueError('fresh bounded f32 support domain changed')
        if not s['key'].startswith('holdout-refined-') or s['role'] != 'fresh-performance-holdout':
            raise ValueError('fresh registration label changed')
    expected = {program_identity(s): s for s in regression_specifications()}
    if {program_identity(s) for s in regressions} != set(expected):
        raise ValueError('R3 regression omitted, added or substituted')
    for s in regressions:
        if s != expected[program_identity(s)]:
            raise ValueError('R3 regression specification or role changed')
    return True


def protocol(configuration):
    c = deepcopy(configuration)
    if c.get('archive_capacity', 4) != 4 or type(c.get('archive_capacity', 4)) is not int:
        raise ValueError('fixed same-pool K4 compile budget required')
    main = c.get('main_policy')
    if main not in MAIN_POLICIES:
        raise ValueError('main_policy must be explicitly preregistered from MAIN_POLICIES')
    base = previous.protocol(c)
    rows = specifications()
    base.update(stage=STAGE, schema_version=1, seed=SEED, specifications=rows,
                historical_inventory=historical_specifications(), configuration=c,
                regression_specifications=regression_specifications())
    base['training_specifications'] = [dict(s, split='observed-training-only') for s in historical_specifications()]
    base['freshness'] = ('six complete and flattened workloads disjoint from the entire historical inventory including R3; '
                         'R3 is observed regression only; capability preflight precedes final source freeze and GPU timing')
    base['execution_export'] = 'minified-json same schema'
    base['holdout_rule'] = 'primary statistics use only six fresh programs; all nine R3 regressions remain mandatory guards'
    base['measurement']['seed'] = SEED
    base['feature_contract'].update(version='gf2-native-decision-completion-v1', frozen_native_coefficients=14,
        calibration='all 14 coefficients including LOCAL6418 remain fixed; no fresh fitting',
        local_footprint='driver allocation for fallback preference only; not transfer traffic, spill count or additive cost')
    base['experiment'].update(ablations=list(POLICY_LABELS), main_policy=main,
        primary=main + '/legacy-static over all six fresh programs',
        same_pool_ablations='each mode reuses its identical complete K-candidate archive and all native evidence for every decision rule; no additional search or compilation')
    base['priority_experiment'] = dict(
        version='same-pool-gf2-decision-refinement-v1', uses_gpu_timing=False, case_or_shape_special_cases=False,
        main_policy=main, main_policy_selected_before_fresh_gpu=True,
        labels=list(POLICY_LABELS), candidate_modes=['joint', 'local'],
        same_pool=True, same_compile_budget=True, all_candidates_retained=True,
        factorial=[dict(rule=name, gf2_exact_score_tie=g, compiled_local_fallback=l)
                   for name, g, l in FACTORIAL_RULES],
        native_priority='old native priority fixed across the four cells; only GF2 changes exact equal-score class order',
        gf2_tie='proven global sectors within exact equal-native-score classes only; all class members need valid evidence, otherwise omit the dimension',
        local_fallback='driver-confirmed LOCAL_SIZE_BYTES within the static primary-min archive class; then old complete tuple and witness/index order; no frame-to-traffic inference',
        missing='legitimate missing evidence omits its dimension for the complete class; malformed or mismatched evidence is an integrity error',
        receipts='bind full pool, profile/source, raw GF2 and compiled driver evidence, original priority, eligibility, refined priority and fallback before any timing',
        affine_completion=dict(
            family='native_base + sum(untrained_feature_count[f] * u[f]) + global_sectors32 * beta',
            parameters='one independent nonnegative u per untrained feature and one nonnegative beta; each parameter is shared across all candidates in its pool',
            zero_counts='inactive unknown-feature counts require a proof of zero; absence is not silently zero',
            nondecomposable='unsupported CFG or missing whole-program evidence has no numerical bound and triggers whole-pool fallback',
            frame='compiled LOCAL footprint remains fallback evidence and is not a term in this family',
            interpretation='conditional additive model family; not a hardware-time or latency bound; no universal runtime-optimum claim'))
    base['acceptance']['native_feedback_fairness'] = ('local and joint retain identical probe/refine and native compile caps; '
        'within each mode all interval, tie, frame and affine decisions reuse the exact same candidate pool')
    base['acceptance']['bounded_selection_quality']['scope'] = 'all six fresh programs; reference is the union of registered archives and baseline, not the raw layout domain'
    promotion = base['acceptance']['promotion']
    promotion.update(primary=main, guard_comparisons=len(rows), guard_scope='all six fresh and all nine R3 regression programs',
        requirements='all declared candidate correctness/safety and decision certificates; bounded regret upperGM<=1.03 and uppermax<=1.10 on six fresh; main time upper<1/1.03; all fifteen guards; explicit overhead evidence')
    base['scope'] = ('SM90 bounded oneCTA backend; six fresh and all nine R3 regressions; '
                     'unmodeled CFG/MMA remains typed; decision certificates are model-conditional and imply no external-framework superiority')
    return base
