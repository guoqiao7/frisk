"""Preregistered fresh workloads for the network/backend decision epoch.

All preceding GF2 workloads are observed regressions. A changed two-dimensional
factorization of an observed flattened workload never restores holdout status.
No compiler, CUDA, measurement reads or model fitting occurs here.
"""
from copy import deepcopy
import math
import gf2_decision_fixture as previous
from selection_quality_fixture import program_identity

STAGE = 'network-selection-execution-qualification'
SEED = 20261107
GPU_UUID = previous.GPU_UUID
source = previous.source
inputs = previous.inputs
numerical_cases = previous.numerical_cases
reference = previous.reference
offsets = previous.offsets
check_numerics = previous.check_numerics
safety_input_identity = previous.safety_input_identity
digest = previous.digest
workload_identity = previous.workload_identity
FACTORIAL_RULES = previous.FACTORIAL_RULES
POLICY_LABELS = previous.POLICY_LABELS + ('joint-resource-frontier','local-resource-frontier')
MAIN_POLICIES = previous.MAIN_POLICIES
FRESH_COHORT = (
    ('diamond', 'diamond', (512, 128), 'none'),
    ('remat', 'remat', (1024, 64), 'none'),
    ('conflict', 'conflict', (256, 256), 'none'),
    ('mixed', 'no-conflict', (16, 16), 'mixed'),
    ('div', 'no-conflict', (512, 128), 'div6'),
    ('exp', 'no-conflict', (512, 128), 'exp1'),
)


def historical_specifications():
    return list({program_identity(s): s for s in
        previous.historical_specifications() + previous.specifications()}.values())


def fresh_specifications():
    return [dict(key='holdout-network-' + name + '-' + str(shape[0]) + 'x' + str(shape[1]),
                 split='holdout', role='fresh-performance-holdout', graph_name=name,
                 case_id=case, shape=list(shape), tail=tail, abi_stratum='weak-vector',
                 dtype='float32', threads=128)
            for name, case, shape, tail in FRESH_COHORT]


def regression_specifications():
    return [dict(s, key='regression-network-' + s['key'],
                 split='observed-anchor' if s['case_id']=='attention' else 'observed-regression',
                 role='observed-preceding-version-regression-only')
            for s in previous.specifications()]


def specifications():
    rows = fresh_specifications() + regression_specifications()
    validate_specifications(rows)
    return rows


def validate_specifications(rows):
    expected = fresh_specifications() + regression_specifications()
    if not isinstance(rows, list) or rows != expected:
        raise ValueError('complete registered six fresh and fifteen observed programs required')
    if len({s['key'] for s in rows}) != len(rows) or len({program_identity(s) for s in rows}) != len(rows):
        raise ValueError('duplicate workload identity')
    old = {program_identity(s) for s in historical_specifications()}
    flat = {workload_identity(s) for s in historical_specifications()}
    fresh = [s for s in rows if s['split']=='holdout']
    if len({workload_identity(s) for s in fresh}) != 6:
        raise ValueError('six distinct flattened holdouts required')
    for s in fresh:
        if program_identity(s) in old or workload_identity(s) in flat:
            raise ValueError('observed flattened workload cannot be a fresh holdout')
        if (len(s['shape']) != 2 or any(type(x) is not int or x<=0 or x&(x-1) for x in s['shape'])
            or math.prod(s['shape']) > 65536 or s['threads'] != 128):
            raise ValueError('registered bounded execution domain changed')
    return True


def protocol(configuration):
    cfg = deepcopy(configuration)
    base = previous.protocol(cfg)
    rows = specifications()
    base.update(stage=STAGE, seed=SEED, specifications=rows,
        historical_inventory=historical_specifications(),
        regression_specifications=regression_specifications())
    base['measurement']['seed'] = SEED
    base['measurement']['confidence'] = ('95% whole paired-batch bootstrap; all21 natural guards use Bonferroni alpha=.05/21; no invalid-batch replacement')
    base['search_contract'].update(version='shared-network-candidate-archive-v2',
        layout_search_probe_limit=65535, variant_score_reserve=1,
        total_probe_limit=65536, joint_local_witness_quota=16384, local_layout_quota=65535,
        refinement_limit=256, native_capacity=4)
    base['holdout_rule'] = 'six genuinely new complete and flattened workloads; all fifteen preceding programs remain observed guards'
    base['freshness'] = ('mixed has exhausted the larger dyadic flattened sizes in the declared domain; '
        'its unobserved 256-element program is fresh and its 65536-element program remains an observed regression. '
        'Five other families use the unobserved 65536-element size. No relabeling restores freshness.')
    base['acceptance']['promotion']['guard_comparisons'] = len(rows)
    base['acceptance']['promotion']['guard_scope'] = 'all six fresh and fifteen preceding observed programs'
    base['acceptance']['promotion']['requirements'] = ('all candidate correctness/safety and decision certificates; '
        'fresh upperGM regret<=1.03 and uppermax<=1.10; main/legacy upper<1/1.03; '
        'all21 Bonferroni guards upper<=1.05; complete separate efficiency evidence')
    base['acceptance']['bounded_selection_quality']['scope'] = 'six new programs against the finite registered archive-plus-legacy union'
    base['experiment']['ablations'] = list(POLICY_LABELS)
    base['priority_experiment']['labels'] = list(POLICY_LABELS)
    base['priority_experiment']['resource_frontier'] = ('exact driver REG/LOCAL allocation/shared bytes and theoretical residency deficit; '
        'independent shared nonnegative unknown penalties, not fitted cycles; Pareto fallback only within static primary minimum')
    base['feature_contract'].update(source_epoch='new ordered/network grammar; v4 is explicitly historical only',
        frozen_native_coefficients=None, calibration='known-only predicate-DAG-v2 feature re-extraction and native14/LOCAL refit; no new holdout fitting',
        static_weights='20 migrated engineering-proxy coefficients with new profile/units contract; not a new latency calibration')
    base['budget']['network_archive'] = dict(version=2, total_probe_limit=65536, refinement_limit=256, native_requests_per_mode=4,
        variant_score_reserve=1, template='preserve incumbent pins and static-best auto; use an available fourth slot for a genuinely different min-work variant; otherwise retain auto',
        no_extra_compile_budget=True)
    base['scope'] = ('bounded SM90 layout-selection qualification plus separately qualified independent-tile multiCTA execution; '
        'singleCTA quality measurements are not full-GPU or long-sequence Attention performance evidence')
    return base
