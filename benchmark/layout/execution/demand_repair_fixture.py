"""New whole-program holdouts for Frisk demand-cost and budgeted-search repair.
Historical observations stay development-only. No CUDA, fitting, source mutation,
or subprocess work occurs at import. Model/profile identities are supplied by
run registration after implementation and development decisions are frozen.
"""
import hashlib
import json
import re
import math
import numpy as np
from calibration_fixture import specifications as old_calibration_specs
from selection_quality_fixture import (historical_specifications as previous_history,
    specifications as previous_quality_specs, program_identity,
    source as previous_source, inputs, numerical_cases, reference, offsets, check_numerics,
    safety_input_identity, GPU_UUID)

SEED=20261103




def source(spec,root,controlled=False):
    text=previous_source(spec,root,controlled)
    if spec['case_id']=='attention':return text
    marker='module attributes {'
    if text.count(marker)!=1:raise ValueError('expected one module attribute block')
    # A common frontend/execution-support domain contract, identical for both
    # cost models; it never changes candidate layout or arithmetic semantics.
    return text.replace(marker,marker+'frisk.execution_domain="sm90-scalar-f32-dag-v1",',1)

def workload_identity(spec):
    """Stronger f32 pointwise identity: avoid re-labelling equal flat workloads."""
    return (spec['case_id'],spec.get('tail','none'),spec.get('abi_stratum','weak-vector'),
            spec.get('dtype','float32'),spec.get('threads',128),math.prod(spec['shape']))

def historical_specifications():
    unique={program_identity(s):s for s in previous_history()+previous_quality_specs()}
    return [dict(case_id=k[0],shape=list(k[1]),tail=k[2],abi_stratum=k[3],dtype=k[4],threads=k[5])
            for k in sorted(unique)]


def development_specifications():
    result=[]
    for s in previous_quality_specs():
        if s['split']=='holdout':
            result.append(dict(s,key='development-known-'+s['graph_name']+'-'+str(s['shape'][0])+'x'+str(s['shape'][1]),
                split='development',role='previous-quality-observed-regression-and-design'))
    return result


def training_specifications():
    # The prior accepted quality run has all34timed group-images from these
    # nine now-observed programs; no new holdout enters fitting.
    return [dict(s,key='training-known-'+s['key'],split='train',
                 role='previous-quality-observed-training-only')
            for s in previous_quality_specs()]


def specifications():
    def item(name,case,shape,tail='none'):
        return dict(key='holdout-'+name+'-'+str(shape[0])+'x'+str(shape[1]),split='holdout',
            role='new-whole-program-holdout',case_id=case,shape=list(shape),tail=tail,
            abi_stratum='weak-vector',dtype='float32',threads=128,graph_name=name)
    rows=[item('diamond','diamond',(128,32)),item('remat','remat',(64,64)),
          item('conflict','conflict',(128,32)),item('mixed','no-conflict',(128,64),'mixed'),
          item('div','no-conflict',(64,16),'div6'),item('exp','no-conflict',(32,32),'exp1')]
    for abi in ('weak-vector','strong-cpasync','strong-tma'):
        rows.append(dict(key='anchor-attention-'+abi,split='observed-anchor',
            role='previously-observed-attention-anchor',case_id='attention',shape=[64,16],tail='none',
            abi_stratum=abi,dtype='float16',threads=128,graph_name='attention'))
    validate_specifications(rows)
    return rows


def validate_specifications(rows):
    keys=[s['key'] for s in rows];identities=[program_identity(s) for s in rows]
    if len(keys)!=len(set(keys)) or len(identities)!=len(set(identities)):
        raise ValueError('duplicate whole-program specification')
    historical={program_identity(s) for s in historical_specifications()}
    historical_workloads={workload_identity(s) for s in historical_specifications()}
    training={program_identity(s) for s in training_specifications()}
    development={program_identity(s) for s in development_specifications()}
    for row in rows:
        ident=program_identity(row)
        if row['split']=='holdout' and ident in historical|training|development:
            raise ValueError('observed/training/development program cannot be a new holdout')
        if row['split']=='holdout' and workload_identity(row) in historical_workloads:
            raise ValueError('observed flattened workload size cannot be a new holdout')
        if row['split']=='observed-anchor' and ident not in historical:
            raise ValueError('anchor must be previously observed')
        if row['split'] not in ('holdout','observed-anchor'):
            raise ValueError('final matrix allows holdout and explicitly observed anchors only')
    return True



def arms(spec,scope,models):
    """Orthogonal model/search treatments; forced references never use seeding."""
    if scope not in ('controlled','natural'):raise ValueError('unknown scope')
    if scope=='controlled' and spec['case_id']=='attention':raise ValueError('Attention is natural only')
    result=[]
    for model in models:
        name=model['name']
        result += [dict(arm=name+'-LI',model=name,selection='LI',search='local',incumbent_witness=False,force_producer=None),
                   dict(arm=name+'-J-old',model=name,selection='J',search='oldbeam',incumbent_witness=False,force_producer=None),
                   dict(arm=name+'-J-new',model=name,selection='J',search='witness',incumbent_witness=True,force_producer=None)]
        if scope=='controlled':
            result += [dict(arm=name+'-'+layout,model=name,selection='J',search='forced',incumbent_witness=False,force_producer=layout)
                       for layout in ('A','B')]
    return result

def protocol(*,models,profiles,feature_contract,search_contract):
    """Build final preregistration with explicit frozen implementation identities.
    The caller must supply real identities, never placeholder model IDs/hashes.
    """
    if not isinstance(models,list) or len(models)!=2 or any(not isinstance(m,dict) for m in models):
        raise ValueError('exactly two declared model descriptors required')
    names=[m.get('name') for m in models]
    if len(set(names))!=2 or any(not isinstance(n,str) or not n for n in names):
        raise ValueError('distinct nonempty model names required')
    for model in models:
        if not isinstance(model.get('id'),str) or not model['id'] or '<' in model['id'] or 'TBD' in model['id']:
            raise ValueError('concrete semantic model IDs required')
    if not isinstance(profiles,list):raise ValueError('profiles must be explicit list')
    for profile in profiles:
        if set(profile)!={'path','sha256'} or not isinstance(profile['path'],str) or not profile['path'] or not isinstance(profile['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',profile['sha256']):
            raise ValueError('profiles need concrete path and SHA256')
    if not isinstance(feature_contract,dict) or not feature_contract.get('version'):
        raise ValueError('explicit versioned feature contract required')
    if not isinstance(search_contract,dict) or not search_contract.get('version'):
        raise ValueError('explicit versioned search contract required')
    return dict(schema_version=2,stage='demand-cost-and-budget-search-repair',seed=SEED,
        specifications=specifications(),historical_inventory=historical_specifications(),
        development_specifications=development_specifications(),training_specifications=training_specifications(),
        models=models,profiles=profiles,feature_contract=feature_contract,search_contract=search_contract,
        split_rule='complete case/shape/tail/ABI/dtype/thread program; all model/policy/candidate arms share split',
        holdout_rule='freeze source/tools/profiles/features/search/budgets/thresholds before any final holdout timing; never refit or alter rules from holdout results',
        freshness='six complete-program combinations and case/tail/ABI/dtype/thread/total-element workloads absent from all29previous registered identities; prevents equal-N flattened pointwise relabelling; three Attention ABI strata are observed anchors only',
        experiment=dict(conversion='C',selections=['LI','J'],controlled='exhausted two-producer A/B domain; all other endpoints singleton',
            natural='same initialized full domains/proofs for both models; measured reference is six-policy selected image union, not full raw-domain optimum',
            oracle_scope='controlled only; natural lower bounds/certified incumbents and search gap reported without optimality claim',
            baseline='fully informed local LI includes own producer/access/input repair; LI seeding off; J-old seeding off; J-new seeding on; each arm has the same declared total budget',
            factorial_ablation='two cost models crossed with J-old/J-new search; LI for each cost model; controlled each model A/B forced references; 114 configurations total',
            primary='repaired-J-new / legacy-J-old; cost effect also D-new/L-new and D-old/L-old; search effect D-new/D-old and L-new/L-old',
            failed_coverage='unsupported, failed, saturated and budget-exhausted assignments remain in declared denominators; no silent zero or selective exclusion'),
        budget=dict(exact_combination_limit=256,beam_width=32,max_expanded_states=65536,refine_beam_solution=True,max_refinement_probes=256,
            accounting='constructive seeds, incumbent checks, beam and refinement consume the same shared total budget; no second hidden local budget'),
        measurement=dict(batches=10,warmup=100,pilot_nodes=100,target_graph_ms=5,max_nodes=4096,min_nodes=1,replays=5,
            bootstrap_resamples=10000,seed=SEED,foreign_pid_rule='invalidate whole paired timing batch; retain and never replace',
            aliases='deduplicate exact cubin/entry/config/physicalABI per program and batch; keep all policy aliases',
            confidence='paired nominal95% descriptive intervals; ranking within-program familywise intervals and3%practical margin; nine natural per-case guards alpha.05/9'),
        acceptance=dict(functional='every registered executable arm passes numerical/ABI/guard/stream/graph and exact safety checks',
            feature_coverage='supported demand domain: per-SSA execution coverage and feature totals agree with backend demand boundary/query ledger; unsupported domains are explicit',
            search_quality='same total budget; selected full-model cost never exceeds any actually constructed verified incumbent; exact controlled oracle still required',
            bounded_selection_quality=dict(geomean_regret_max=1.03,max_case_regret_max=1.10,
                scope='controlled complete finite domain; natural selected-image union only',
                confidence='qualified only when upperCI clears both bounds, failed if lowerCI violates either, otherwise uncertain'),
            promotion=dict(primary='repaired-J-new/baseline-J-old on six fresh natural holdouts',
                geomean_time_ratio_upper_lt=1./1.03,per_case_upper_max=1.05,guard_alpha=.05,guard_comparisons=9,
                requirements='complete coverage, exact constrained identities, repaired-J-new controlled+natural regret qualified, no hard-resource violation; otherwise preserve existing default'),
            no_hardcoded_tie_fix='equal surrogate cost plus real performance difference is a model discriminator failure; do not special-case historical case names/layout IDs'),
        numerics=dict(ordinary='finite uint32 bit equality including signed zero',exp1_and_mixed='finite abs(error)<=1e-5+2e-6*abs(reference)',
            attention='finite abs(error)<=.002+.002*abs(reference)',
            inputs='unchanged seven f32 and fifteen Attention cases from accepted closure; timing seed20261031+i for i0..9'),
        safety=dict(tools=['memcheck','racecheck','synccheck','initcheck'],reuse='exact cubin+complete emission config+physicalABI+same spec/safety input protocol'),
        execution=dict(gpu_uuid=GPU_UUID,device_ordinal=1,grid=[1,1,1],threads=128,entry='frisk_execution',num_warps=4,
            enable_fp_fusion=False,compiler_emission_query_limit=65536,output_aliasing='O disjoint from Q/K/V; read-only inputs may overlap'),
        resources=dict(shared='execution structural plan=emitter=compiler=driver',physical_registers='compiler and loaded-driver feedback; no fabricated static point prediction',
            spill='localbytes are not spill counts; unknown retained',residency='theoretical resource capacity, not measured occupancy'),
        originality=dict(adopted_principles=['incremental cost for surviving and replaced work','hardware resource capacity and limited empirical validation'],
            frisk_design=['consumer-sensitive GF2 demand propagated across selected SSA layouts','cost semantics aligned to Frisk execution and conversion ownership','graph-wide joint assignment and conversion planning under one budget'],
            proof_requirement='distinguish implementation facts, mathematical equivalence, measured bounded advantages and unproven wider claims; no exclusivity claim for GF2/kernel/orthogonal-complement itself'),
        scope='bounded SM90 single-CTA supported backend; no cross-framework or universal-latency advantage claim')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
