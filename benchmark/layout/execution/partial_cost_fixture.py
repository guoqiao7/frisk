"""Fixed fresh programs and explicit budgets for partial-cost efficiency repair.
No CUDA, subprocesses, source changes or fitting at import.
"""
import math,re
from copy import deepcopy
import native_cost_fixture as observed
from selection_quality_fixture import program_identity
SEED=20261105
GPU_UUID=observed.GPU_UUID
source=observed.source
inputs=observed.inputs
numerical_cases=observed.numerical_cases
reference=observed.reference
offsets=observed.offsets
check_numerics=observed.check_numerics
safety_input_identity=observed.safety_input_identity
digest=observed.digest


def historical_specifications():
    rows=observed.historical_specifications()+observed.specifications()
    return list({program_identity(s):s for s in rows}.values())


def workload_identity(s):
    return (s['case_id'],s.get('tail','none'),s.get('abi_stratum','weak-vector'),s.get('dtype','float32'),s.get('threads',128),math.prod(s['shape']))


def specifications():
    def row(name,case,shape,tail='none'):
        return dict(key='holdout-efficient-'+name+'-'+str(shape[0])+'x'+str(shape[1]),split='holdout',role='fresh-performance-holdout',graph_name=name,
                    case_id=case,shape=list(shape),tail=tail,abi_stratum='weak-vector',dtype='float32',threads=128)
    rows=[row('diamond','diamond',(128,128)),row('remat','remat',(256,64)),row('conflict','conflict',(128,128)),
          row('mixed','no-conflict',(512,64),'mixed'),row('div','no-conflict',(128,128),'div6'),row('exp','no-conflict',(128,64),'exp1')]
    rows += [dict(s,role='observed-ABI-regression-only') for s in observed.specifications() if s['case_id']=='attention']
    validate_specifications(rows);return rows


def validate_specifications(rows):
    if len(rows)!=9 or len({s['key'] for s in rows})!=9 or sum(s['split']=='holdout' for s in rows)!=6:raise ValueError('complete six fresh +three anchors required')
    history=historical_specifications();ids={program_identity(s) for s in history};flat={workload_identity(s) for s in history}
    for s in rows:
        if s['split']=='holdout':
            if program_identity(s) in ids or workload_identity(s) in flat:raise ValueError('observed complete or flattened program cannot be fresh')
            if s['dtype']!='float32' or s['threads']!=128 or s['abi_stratum']!='weak-vector' or math.prod(s['shape'])>65536:raise ValueError('fresh support domain changed')
            if any(type(x) is not int or x<=0 or x&(x-1) for x in s['shape']):raise ValueError('dyadic positive shape required')
        elif s['split']!='observed-anchor' or program_identity(s) not in ids:raise ValueError('unknown anchor')
    return True


def protocol(configuration):
    c=deepcopy(configuration)
    for name in ('static','native','old_native'):
        if not isinstance(c.get(name),dict) or not all(isinstance(c[name].get(k),str) and c[name][k] for k in ('id','profile')):raise ValueError('concrete profile IDs and paths required')
    if type(c.get('archive_capacity',4)) is not int or not 2<=c.get('archive_capacity',4)<=6:raise ValueError('archive capacity2..6 required')
    k=c.get('archive_capacity',4)
    # Retain numerical, device, safety and fixed statistical contracts from the
    # independently accepted prior harness; replace only declared stage design.
    base=observed.protocol(models=[dict(name='legacy',id='sm90-layout-lex9-legacy-v2',profile=None),
          dict(name='v5d',id=c['static']['id'],profile=c['static']['profile']),
          dict(name='surviving',id=c['static']['id'],profile=c['static']['profile'])],
          native=c['native'],profiles=c['profiles'],training_inputs=c['training_inputs'],preserved_evidence=c['preserved_evidence'])
    base.update(stage='partial-cost-local-efficiency-qualification',schema_version=1,seed=SEED,specifications=specifications(),historical_inventory=historical_specifications(),configuration=c)
    base['training_specifications']=[dict(s,split='observed-training-only') for s in historical_specifications()]
    base['acceptance']['native_feedback_fairness']='local and joint have identical total probe/refine and compile caps; own candidate pools differ; decision/LOCAL ablations share the exact joint pool; every unknown candidate stays in its interval certificate'
    base['acceptance']['search_quality']='archive complete assignments independently replayed and verified; candidate_archive_search accounts all local/J stages under one nonreset shared budget; pinned actual incumbents never lost; no raw-domain optimum claim'
    base['feature_contract']=dict(version='partial-LOCAL-native-v1',static_surviving_features=20,native_static_evidence_features=15,composite_score_features=14,unknown_family='only valid training-zero coefficient-zero nonnegative additive terms; ordinary unsupported has no numeric interval')
    base['search_contract']=dict(version='shared-complete-candidate-archive-v1',default_capacity=k,probe_limit=65536,refinement_limit=256,local_probe_limit=16384,local_refinement_limit=256,joint_uses_actual_remaining=True)
    base['models']=[dict(name='legacy',id='sm90-layout-lex9-legacy-v2',profile=None),dict(c['static'],name='surviving')]
    base['experiment']=dict(conversion='C',candidate_archive_capacity=k,design=['legacy-static-single-J','local-native-shared-budget','joint-native-shared-budget'],
        sources='one staticv6 prepared domain reused by local and joint archives; extra legacy prepare is a qualification reference only',
        local='fully-informed local producer stage and full input/consumer repair; full65536/256 total, same native compile capK',
        joint='coordinated local witness probe quota1/4 (refinement may use the shared256) followed by J; one shared65536/256 budget; pinned actual local/joint complete incumbents',
        compiled_reference='union of at most2K+1 registered selected archive assignments; not raw-domain GPU optimum',
        ablations=['joint-old-strict','joint-old-partial','joint-local-calibrated-partial','local-local-calibrated-partial','joint-static','local-static','legacy-static'],
        primary='joint-local-calibrated-partial/legacy-static over all six fresh programs; informed local comparison reported separately',
        fallback='unbounded nondecomposable unsupported evidence explicitly falls back to own static incumbent; trainingzero free nonnegative coefficients may be certified by complete-pool lower bounds',
        no_hidden_pool='all archive candidates retained and replayed; no deletion based on measured latency; compile requests stop at declaredK and exact duplicates use receipts')
    base['budget']=dict(local_total_probes=65536,joint_total_probes=65536,per_mode_refinement=256,witness_probe_quota=16384,witness_refine_quota=256,
        native_compile_requests_per_mode=k,unique_native_api_misses_per_mode_max=k,qualification_reference_extra_probes=65536,
        accounting='local/J stage receipts shared counters never reset; native misses, exact-cache hits and internal cache unknown distinguished; batch startup counted',
        previous_production_path=dict(source_solves=6,total_probe_cap=393216,total_refine_cap=1536,compile_request_cap=6),
        efficiency='cold and warm same-input observed-R1 comparisons required separately; kernel/setup/validation costs separated; no cold claim for ordinary cache')
    base['measurement']['seed']=SEED
    base['acceptance']['bounded_selection_quality']['scope']='all six fresh fixed programs; union of registered archives plus baseline only'
    base['acceptance']['promotion']=dict(primary='joint-local-calibrated-partial/legacy-static',geomean_time_ratio_upper_lt=1/1.03,per_case_upper_max=1.05,guard_alpha=.05,guard_comparisons=9,
        requirements='all declared candidate correctness/safety and interval certificates; bounded regret upperGM<=1.03 and uppermax<=1.10; main time upper<1/1.03; nine guards; explicit overhead reduction evidence')
    base['scope']='SM90 bounded oneCTA backend; unmodeled CFG/MMA remains typed; no external-framework superiority claim'
    return base
