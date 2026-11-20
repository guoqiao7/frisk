"""Preregistered native-cost qualification: pure fixtures, no CUDA/subprocess work.
All earlier performance observations are development data.  A common finite
portfolio supplies static and native-feedback policies with identical candidates.
"""
import math
import re
from copy import deepcopy
import demand_repair_fixture as previous
from selection_quality_fixture import program_identity

SEED=20261104
GPU_UUID=previous.GPU_UUID
source=previous.source
inputs=previous.inputs
numerical_cases=previous.numerical_cases
reference=previous.reference
offsets=previous.offsets
check_numerics=previous.check_numerics
safety_input_identity=previous.safety_input_identity
digest=previous.digest
MODEL_NAMES=('legacy','v5d','surviving')


def historical_specifications():
    rows=previous.historical_specifications()+previous.specifications()
    unique={program_identity(s):s for s in rows}
    return [dict(case_id=k[0],shape=list(k[1]),tail=k[2],abi_stratum=k[3],dtype=k[4],threads=k[5]) for k in sorted(unique)]


def workload_identity(spec):
    return (spec['case_id'],spec.get('tail','none'),spec.get('abi_stratum','weak-vector'),spec.get('dtype','float32'),spec.get('threads',128),math.prod(spec['shape']))


def specifications():
    def row(name,case,shape,tail='none'):
        return dict(key='holdout-native-'+name+'-'+str(shape[0])+'x'+str(shape[1]),split='holdout',role='new-performance-holdout',graph_name=name,
                    case_id=case,shape=list(shape),tail=tail,abi_stratum='weak-vector',dtype='float32',threads=128)
    rows=[row('diamond','diamond',(256,32)),row('remat','remat',(128,64)),row('conflict','conflict',(256,32)),
          row('mixed','no-conflict',(256,64),'mixed'),row('div','no-conflict',(128,64),'div6'),row('exp','no-conflict',(128,32),'exp1')]
    for abi in ('weak-vector','strong-cpasync','strong-tma'):
        rows.append(dict(key='anchor-attention-'+abi,split='observed-anchor',role='observed-compatibility-anchor',graph_name='attention',
                         case_id='attention',shape=[64,16],tail='none',abi_stratum=abi,dtype='float16',threads=128))
    validate_specifications(rows)
    return rows


def validate_specifications(rows):
    if len(rows)!=9 or sum(x.get('split')=='holdout' for x in rows)!=6:raise ValueError('six fixed holdouts and three anchors required')
    if len({x['key'] for x in rows})!=len(rows) or len({program_identity(x) for x in rows})!=len(rows):raise ValueError('duplicate program')
    fresh=[workload_identity(x) for x in rows if x['split']=='holdout']
    if len(fresh)!=len(set(fresh)):raise ValueError('duplicate flattened execution within fresh suite')
    history=historical_specifications();ids={program_identity(x) for x in history};flat={workload_identity(x) for x in history}
    for s in rows:
        if s['split']=='holdout':
            if program_identity(s) in ids or workload_identity(s) in flat:raise ValueError('observed spec or flattened execution cannot be a fresh holdout')
            if s['dtype']!='float32' or s['threads']!=128 or s['abi_stratum']!='weak-vector':raise ValueError('fresh support domain changed')
            if any(type(n) is not int or n<=0 or n&(n-1) for n in s['shape']) or math.prod(s['shape'])>65536:raise ValueError('unsupported dyadic shape')
        elif s['split']!='observed-anchor' or program_identity(s) not in ids:raise ValueError('unknown anchor or split')
    return True


def arms(models):
    if tuple(m['name'] for m in models)!=MODEL_NAMES:raise ValueError('ordered legacy/v5d/surviving models required')
    return [dict(arm=m['name']+'-'+selection,model=m['name'],selection='LI' if selection=='LI' else'J',search='local' if selection=='LI' else'witness',
                 incumbent_witness=selection!='LI',force_producer=None) for m in models for selection in ('LI','J-new')]


def planned_native_fallback(spec):
    if spec['case_id']=='attention':return 'attention-outside-native-straight-line-domain'
    if spec.get('tail') in ('div6','mixed'):return 'div-lowering-outside-native-straight-line-domain'
    return None


def orthogonal_training_plan():
    return dict(whole_programs='all exact-bridged images from sealed previous34 and observed demand-repairR4 pools; no fresh holdout fitting',
        surviving_work=dict(axis='producer/unused-input elimination',paired='same observable algorithm and output, vary dead input/producer path and load-use overlap',
                            independent_check='pre-native demand versus surviving cubin/SASS work; never infer dynamic counts from emitter counts'),
        communication=dict(routes=['Register','Shuffle','Shared'],axes=['source/destination register vectors','broadcast/replication','bank conflict','dependent exchange count'],
                           repeats=[8,32,128],shared_strides=[1,4,8],paired='same registered 128-lane uint32 input/output with root calibration reference; baseline and each dependent communication family compiled and checked',
                           controls='zero-communication baseline plus dependent barrier/Shared/Shuffle chains; report positive marginal timing confidence and identifiability'),
        native_schedule=dict(axis='load/use placement',paired='all observed full-program images; measure native dependency/order variation and count/resource changes jointly, without inventing an isolated schedule effect',
                             independent_check='native dependency DAG and observed ptxas order; do not equate PTX load groups with concurrency'),
        fitting='freeze regression family, coefficients, feature support and all fallback rules on known/development data before new timings',
        separation='microbenchmark names/shapes, repetitions, full paired controls and tool identities registered in calibration manifest; no post-hoc subset removal')


def protocol(*,models,native,profiles,training_inputs,preserved_evidence):
    arms(models)
    for model in models:
        if not isinstance(model.get('id'),str) or not model['id'] or '<' in model['id']:raise ValueError('concrete semantic model ID required')
    if not isinstance(native,dict) or not all(isinstance(native.get(k),str) and native[k] for k in ('id','profile','analysis_version')):raise ValueError('concrete native profile and analyzer identity required')
    if not isinstance(training_inputs,dict) or not training_inputs:raise ValueError('explicit frozen known-only fitting inputs required')
    for pair in profiles+preserved_evidence:
        if set(pair)!={'path','sha256'} or not isinstance(pair['path'],str) or not re.fullmatch('[0-9a-f]{64}',pair['sha256']):raise ValueError('concrete immutable file hash required')
    p=deepcopy(previous.protocol(models=[models[0],models[2]],profiles=profiles,
        feature_contract={'version':'surviving-demand-plus-native-feedback-v1'},search_contract={'version':'six-source-common-portfolio-v1'}))
    p.update(schema_version=1,stage='native-cost-repair-qualification',seed=SEED,specifications=specifications(),historical_inventory=historical_specifications(),
             models=deepcopy(models),native=deepcopy(native),training_specifications=[dict(s,split='known-development') for s in historical_specifications()],training_inputs=deepcopy(training_inputs),preserved_evidence=deepcopy(preserved_evidence),training_design=orthogonal_training_plan())
    p['freshness']='full spec and case/tail/ABI/dtype/threads/N disjoint from every previous observed program; CPU capability preflight permitted and recorded; no GPU timing before final freeze'
    p['experiment']=dict(conversion='C',scope='natural only; common six-source selected-assignment pool, not the complete raw domain',
        sources=[a['arm'] for a in arms(models)],candidate_cap=6,compile_cap=6,
        baseline='legacy-J-new is the single-search static reference; all common-pool policies receive the same complete native feedback',
        informed_local='each static LI includes its own producer/access/input repair; native-LI-opportunity and native-J-opportunity see exactly the same common pool and must choose identically',
        attribution='native common-pool ranking is not evidence of joint-search superiority; pool construction costs exceed a single static solve and are reported explicitly',
        fallback='mixed/div and Attention predeclared typed unsupported; any unsupported/unavailable candidate forces the entire program to surviving-J-new; never filter candidates',
        primary='native-pool / legacy-J-new across all six fixed new natural programs including div fallback; predeclared diamond/remat/conflict/exp four-case subset descriptive only',
        failed_coverage='every declared source candidate remains in denominators; failures stop qualification and remain retained')
    p['budget']=dict(per_source_total_probes=65536,per_source_refinement=256,sources=6,portfolio_total_probes=393216,portfolio_total_refinement=1536,
        max_native_compile_requests=6,max_unique_native_images=6,k_cap=6,
        accounting='six complete solves; all prepare/solve/probe/refinement/compile/cache/analysis wall times reported; no claim of equal cost to one static solve',
        fairness='one frozen pool and native feedback ledger shared by static/native LI/J opportunity controls; native feedback never uses holdout latency labels')
    p['measurement']['seed']=SEED
    p['acceptance']['feature_coverage']='Frisk demand coverage exact where supported; post-native work uses independently bound cubin/disassembly typed parser; partial features never silently zero'
    p['acceptance']['search_quality']='all six source solves individually enforce shared total budget and valid incumbent; portfolio totals enforced separately; no full-domain optimum claim'
    p['acceptance']['bounded_selection_quality']['scope']='six new natural fixed source-pool images only; no raw-domain optimum claim'
    p['acceptance']['promotion']=dict(primary='native-pool/legacy-J-new on six fresh programs including div fallback',geomean_time_ratio_upper_lt=1/1.03,
        per_case_upper_max=1.05,guard_alpha=.05,guard_comparisons=9,
        requirements='all54 source configurations functional/safety pass; complete native fallback accounting; native-pool bounded regret qualified; all9 guard upperCI<=1.05; otherwise preserve default')
    p['acceptance']['native_feedback_fairness']='shared identical candidates/compile outputs/profile; native-LI/J-opportunity decisions equal; no kernel timing available to selection'
    p['acceptance']['cpu_setup_reporting']='report summed prepare/selection/compile/native-analysis work and descriptive extra-setup/kernel-saving break-even; no preregistered CPU-wall promotion threshold; probe/compile caps remain mandatory'
    p['resources']['spill']='native LOCAL instructions and driver localbytes reported separately; unknown spill count retained'
    p['scope']='SM90 oneCTA supported backend; static versus bounded compilation-feedback modes have distinct disclosed costs; no cross-framework claim'
    return p
