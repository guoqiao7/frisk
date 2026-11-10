"""Timing-free joint Frisk address/native cost decisions.

Weights live over nonnegative real numbers; GF(2) is used only by the bound
address proof. Fitted scores and empirical risk envelopes are not hardware time
bounds. A nonzero training column never makes its coefficient physically exact.
"""
from __future__ import annotations
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import decision_priority as priority
import native_cost_calibration as native_calibration

CONTRACT = 'frisk-joint-native-gf2-risk-v1'
MODEL = 'sm90-joint-native14-gf2-v1'
SUMMARY_MODEL = 'sm90-joint-complete-summary-gf2-v1'
LOCALITY_MODEL = 'sm90-joint-native-gf2-locality-v1'
SUMMARY_LOCALITY_MODEL = 'sm90-joint-complete-summary-locality-v1'
ADDRESS_MARGIN_MODEL = 'frisk-controlled-source-address-ordering-margin-v1'
NATIVE_FEATURES = tuple(native_calibration.MODEL_FEATURES)
FEATURES = NATIVE_FEATURES + ('gf2_global_sectors32',)
STATIC_FEATURES = ('access_warp_issues','global_sectors32','shared_bank_excess','copy_setup_units',
    'mma_instructions','reduction_element_ops','alu_warp_issues','div_warp_issues','exp_warp_issues',
    'cast_warp_issues','conversion_register_bytes','conversion_shuffle_bytes','conversion_shared_bytes',
    'conversion_conservative_bytes','synchronization_barriers','shared_read_bank_excess',
    'shared_write_bank_excess','shared_warp_issues','shuffle_warp_issues','communication_alu_warp_issues')
UNITS = {n: 'static-native-instruction-occurrences' for n in NATIVE_FEATURES}
UNITS.update({n:'static-native-operand-bytes' for n in
    ('global_load_bytes','global_store_bytes','shared_transfer_bytes','local_transfer_bytes')})
UNITS.update(load_use_exposure_32='static-native-load-dependency-exposure-with-window32',
    load_use_exposure_128='static-native-load-dependency-exposure-with-window128',
    dependency_depth='static-native-dependency-DAG-depth',
    symbolic_bank_excess='byte-matched-emitted-warp-bank-serialization-excess',
    gf2_global_sectors32='proven-layout-32B-warp-request-sectors')
BASIS = 'nonnegative empirical static-work proxy and graph-held-out risk calibration; not hardware-time bounds'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def implementation():
    root=Path(__file__).resolve().parent
    names=('joint_cost_model.py','decision_priority.py','native_cost_calibration.py','partial_cost_selection.py','execution_cost_summary.py')
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}


def _integer(v):
    return type(v) is int and v>=0


def _seal(payload):
    return dict(payload=payload,payload_sha256=digest(payload))

def execution_domain(configuration,native,correctness,emitted):
    threads=configuration.get('threads');grid=configuration.get('grid')
    if not _integer(threads) or threads==0 or not isinstance(grid,list) or len(grid)!=3 or any(not _integer(v) or v==0 for v in grid):raise ValueError('invalid validated launch configuration')
    if configuration.get('entry')!='frisk_execution' or correctness.get('physical_resources',{}).get('threads_per_block')!=threads:raise ValueError('validated launch/native threads differ')
    version=native.get('native_metadata',{}).get('SM',{}).get('version',{})
    return dict(threads=threads,grid=list(grid),sm_version=[version.get('major'),version.get('minor')],
        dtype=configuration.get('spec',{}).get('dtype'),gpu_uuid=correctness.get('gpu_uuid'),
        math_flags={k:emitted.get(k) for k in ('exp','ftz','fp_fusion','p_rounding','reassociation')})


def supported_execution_domain(domain):
    return isinstance(domain,dict) and domain.get('threads')==128 and domain.get('grid')==[1,1,1] and domain.get('sm_version')==[9,0] and domain.get('dtype') in ('float32','float16') and isinstance(domain.get('gpu_uuid'),str) and bool(domain['gpu_uuid']) and domain.get('math_flags')==dict(exp='ex2.approx.f32(mul.rn.f32(x, log2e-f32))',ftz=False,fp_fusion=False,p_rounding='cvt.rn.f16.f32',reassociation=False)


def candidate_evidence(directory,row):
    """Extend existing compiled/GF2 proof with sparse ordered-network static20.

    The allowlisted record is already bound by the preceding receipt. No event
    timings, resource-frame-to-traffic conversion, or compiler work occurs here.
    """
    base=priority.candidate_evidence(directory,row)
    record=json.loads((Path(directory)/'record.json').read_text())
    common=record.get('common_static_contract')
    if common is None or common.get('feature_order')!=list(STATIC_FEATURES):
        raise ValueError('ordered-network static feature contract required')
    priority.common_static_contract(common)
    raw=record.get('cost',{}).get('features')
    if not isinstance(raw,dict) or set(raw)-set(STATIC_FEATURES) or any(not _integer(v) for v in raw.values()):
        raise ValueError('invalid static20 feature units')
    values={n:raw.get(n,0) for n in STATIC_FEATURES}
    g=base['payload']['gf2']
    if g['status']=='available' and values['global_sectors32']!=g['global_sectors32']:
        raise ValueError('static/GF2 proven address burden differs')
    auxiliary={}
    for name in ('kernel.ptx','execution_config.json','compile.json','emitted-config.json','configuration.json'):
        path=Path(directory)/name;value=hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get('artifact_hashes',{}).get(name)!=value:raise ValueError('joint auxiliary image binding differs')
        auxiliary[name]=value
    read=lambda name:json.loads((Path(directory)/name).read_text())
    domain=execution_domain(read('configuration.json'),read('native-analysis.json'),read('correctness.json'),read('execution_config.json'))
    return _seal(dict(schema_version=1,contract=CONTRACT,kind='joint-candidate-evidence',
        implementation_sha256=implementation(),preceding=base,static20=values,auxiliary_bindings=auxiliary,execution_domain=domain,
        static_contract=common,uses_gpu_timing=False))


def _open_evidence(value):
    if not isinstance(value,dict) or set(value)!={'payload','payload_sha256'} or digest(value['payload'])!=value['payload_sha256']:
        raise ValueError('joint evidence envelope differs')
    p=value['payload']
    if p.get('contract')!=CONTRACT or p.get('schema_version')!=1 or p.get('implementation_sha256')!=implementation() or p.get('uses_gpu_timing') is not False:
        raise ValueError('joint evidence implementation differs')
    if p.get('kind')!='joint-candidate-evidence':raise ValueError('joint candidate evidence required')
    base=p['preceding'];arm=base['payload']['arm'];opened=priority._evidence({arm:base})[arm]
    if p.get('static_contract',{}).get('feature_order')!=list(STATIC_FEATURES):raise ValueError('static feature contract differs')
    priority.common_static_contract(p['static_contract'])
    f=p.get('static20')
    if not isinstance(f,dict) or set(f)!=set(STATIC_FEATURES) or any(not _integer(v) for v in f.values()):raise ValueError('static features differ')
    if opened['gf2']['status']=='available' and f['global_sectors32']!=opened['gf2']['global_sectors32']:raise ValueError('GF2/static feature mismatch')
    return p,opened


def features(native,evidence,analysis_contract=None):
    """Extract supported native15 or typed-unavailable; reject corrupt evidence."""
    p,e=_open_evidence(evidence)
    if digest(native)!=e['native_sha256']:raise ValueError('joint native/evidence binding differs')
    result=dict(native_sha256=digest(native),evidence_sha256=digest(evidence),uses_gpu_timing=False)
    def unavailable(reason):return dict(result,status='unsupported',features=None,reason=reason)
    if native.get('status')!='available':return unavailable('native-analysis-'+str(native.get('status')))
    support=native.get('support',{});identity=native.get('identity',{})
    if support.get('status')!='supported' or support.get('reasons')!=[] or support.get('domain')!=identity.get('domain'):
        return unavailable('native-support-proof-unavailable')
    if native.get('schema_version')!=identity.get('schema_version') or identity.get('cubin_sha256')!=e['cubin_sha256'] or identity.get('entry')!='frisk_execution':
        raise ValueError('native image/schema binding differs')
    if native.get('symbolic_banks',{}).get('contract')!='frisk-emitted-affine-shared-bank-v1':return unavailable('symbolic-bank-proof-unavailable')
    actual=native_calibration.analysis_contract(native)
    if analysis_contract is not None and actual!=analysis_contract:return unavailable('native-analysis-contract-mismatch')
    raw=native.get('features',{})
    names=(set(NATIVE_FEATURES)-{'shared_transfer_bytes','local_transfer_bytes','symbolic_bank_excess'}) | {'shared_load_bytes','shared_store_bytes','local_load_bytes','local_store_bytes'}
    if set(raw)!=names or any(not _integer(v) for v in raw.values()):raise ValueError('invalid native feature units')
    banks=native['symbolic_banks']
    if any(not _integer(banks.get(n)) for n in ('shared_read_bank_excess','shared_write_bank_excess')):raise ValueError('invalid bank feature units')
    if 'native_cubin_sha256' in banks and banks['native_cubin_sha256']!=e['cubin_sha256']:raise ValueError('symbolic/native binding differs')
    g=e['gf2']
    if g['status']!='available':return unavailable('proven-GF2-address-burden-unavailable')
    values=native_calibration.model_features(native);values['gf2_global_sectors32']=g['global_sectors32']
    return dict(result,status='available',features=values,analysis_contract=actual,reason=None)


def feature_contract(profile):
    if profile.get('model')==ADDRESS_MARGIN_MODEL:
        import access_reuse as locality
        names=('gf2_global_sectors32',)+tuple(locality.FOOTPRINT_FEATURES)
        return names,{n:UNITS[n] if n=='gf2_global_sectors32' else locality.UNITS[n] for n in names}
    if profile.get('model') in (MODEL,LOCALITY_MODEL):names,units=FEATURES,dict(UNITS)
    else:
        if profile.get('model') not in (SUMMARY_MODEL,SUMMARY_LOCALITY_MODEL):raise ValueError('unknown joint feature family')
        import execution_cost_summary as summary
        family=profile.get('cost_family')
        if family=='sm90-acyclic-call-expanded-occurrences-v1':names=tuple(summary.BASE_FEATURES)
        elif family=='sm90-immediate-wait-wgmma-v1':names=tuple(summary.BASE_FEATURES)+tuple(summary.ATTENTION_FEATURES)
        else:raise ValueError('unknown complete-summary cost family')
        if profile.get('uses_gf2',True):names=names+('gf2_global_sectors32',)
        units={n:summary.UNITS[n] if n!='gf2_global_sectors32' else UNITS[n] for n in names}
    extra=profile.get('locality_features',[])
    if bool(extra)!=(profile.get('model') in (LOCALITY_MODEL,SUMMARY_LOCALITY_MODEL)):raise ValueError('locality model/feature family differs')
    if extra:
        import access_reuse as locality
        if not isinstance(extra,list) or len(set(extra))!=len(extra) or set(extra)-set(locality.FOOTPRINT_FEATURES):raise ValueError('invalid positive-cost locality axes')
        contract=profile.get('locality_analysis_contract',{})
        expected=dict(contract=locality.CONTRACT,cost_family=locality.FAMILY,analyzer_sha256=hashlib.sha256(Path(locality.__file__).read_bytes()).hexdigest())
        if contract!=expected:raise ValueError('locality profile analyzer/family differs')
        names=names+tuple(extra);units.update({n:locality.UNITS[n] for n in extra})
    return names,units


def extend_locality(row,document,evidence,profile):
    """Bind logical/emitter-order address footprints, never native cache misses."""
    if not profile.get('locality_features'):return row
    if row['status']!='available':return row
    def unavailable(reason):return dict(row,status='unsupported',features=None,reason=reason)
    if not isinstance(document,dict) or document.get('status')!='complete':return unavailable('locality-evidence-unavailable')
    body={k:v for k,v in document.items() if k!='summary_sha256'}
    if document.get('summary_sha256')!=digest(body):raise ValueError('locality evidence hash differs')
    import access_reuse as locality
    provenance=document.get('provenance',{})
    expected=profile['locality_analysis_contract']
    actual=dict(contract=document.get('contract'),cost_family=document.get('cost_family'),analyzer_sha256=provenance.get('analyzer_sha256'))
    if actual!=expected:return unavailable('locality-analysis-contract-differs')
    if provenance.get('analyzer_sha256')!=hashlib.sha256(Path(locality.__file__).read_bytes()).hexdigest():return unavailable('locality-analyzer-changed')
    if document.get('feature_names')!=list(locality.FEATURE_NAMES) or document.get('feature_units')!=locality.UNITS:raise ValueError('locality feature units differ')
    sources=provenance.get('source_dependencies',{})
    if set(sources)!=set(locality.SOURCE_FILES):raise ValueError('locality source dependency set differs')
    for name,value in sources.items():
        if hashlib.sha256((Path(locality.__file__).parent/name).read_bytes()).hexdigest()!=value:return unavailable('locality-source-dependency-changed')
    values=document.get('features')
    if not isinstance(values,dict) or set(values)!=set(locality.FEATURE_NAMES) or any(not _integer(v) for v in values.values()):raise ValueError('locality feature count units differ')
    p,e=_open_evidence(evidence);bindings=dict(e['bindings']);bindings.update(p.get('auxiliary_bindings',{}))
    dependencies=provenance.get('dependencies',{})
    if not {'execution.json','record.json','kernel.ttgir','kernel.cubin','image-identity.json'}<=set(dependencies):raise ValueError('locality compiled-image dependencies missing')
    for name,value in dependencies.items():
        if bindings.get(name)!=value:raise ValueError('locality compiled-image binding differs: '+name)
    emission=document.get('emission_proof',{})
    if emission.get('ttgir_exact_sha256')!=bindings['kernel.ttgir'] or emission.get('native_order_proven') is not False:raise ValueError('locality emitter-order proof differs')
    return dict(row,features=dict(row['features'],**{n:values[n] for n in profile['locality_features']}),
        locality_sha256=digest(document),locality_semantics='address footprint/potential under the proven Frisk query order; no native-cache-miss or latency claim')


def locality_reduction_holds(rows,documents,profile,order):
    """Recheck every empirical dropped-axis relation on this actual pool."""
    reduction=profile.get('locality_reduction')
    if not reduction:return True
    first=order[0]
    for entry in reduction['ledger']:
        if not entry['status'].startswith('dropped'):continue
        axis=entry['axis'];coefficients=entry['coefficients']
        for arm in order:
            actual=documents[arm]['features'][axis]-documents[first]['features'][axis]
            expected=sum(Fraction(pair[0],pair[1])*(rows[arm]['features'][name]-rows[first]['features'][name]) for name,pair in coefficients.items())
            if actual!=expected:return False
    return True


def summary_features(native,summary,evidence,profile):
    p,e=_open_evidence(evidence)
    if digest(native)!=e['native_sha256']:raise ValueError('summary native/evidence binding differs')
    result=dict(native_sha256=digest(native),evidence_sha256=digest(evidence),summary_sha256=digest(summary),uses_gpu_timing=False)
    def unavailable(reason):return dict(result,status='unsupported',features=None,reason=reason)
    if not isinstance(summary,dict) or summary.get('status')!='complete':return unavailable('complete-execution-summary-unavailable')
    body={k:v for k,v in summary.items() if k!='summary_sha256'}
    if summary.get('summary_sha256')!=digest(body):raise ValueError('summary payload hash differs')
    if summary.get('contract')!='frisk-complete-execution-cost-summary-v1' or summary.get('cost_family')!=profile['cost_family']:
        return unavailable('complete-summary-cost-family-differs')
    all_names,all_units=feature_contract(profile);names=tuple(n for n in all_names if n not in profile.get('locality_features',[]));units={n:all_units[n] for n in names};use_gf2='gf2_global_sectors32' in names;base=names[:-1] if use_gf2 else names;provenance=summary.get('provenance',{})
    if provenance.get('analyzer_sha256')!=implementation()['execution_cost_summary.py']:return unavailable('summary-analyzer-implementation-differs')
    actual_contract=dict(contract=summary['contract'],analyzer_sha256=provenance['analyzer_sha256'],nvdisasm=provenance.get('nvdisasm'))
    if actual_contract!=profile['analysis_contract']:return unavailable('summary-analysis-contract-differs')
    if summary.get('feature_names')!=list(base) or summary.get('feature_units')!={n:units[n] for n in base}:raise ValueError('summary feature units differ')
    values=summary.get('features')
    if not isinstance(values,dict) or set(values)!=set(base) or any(not _integer(v) for v in values.values()):raise ValueError('summary features malformed')
    bindings=dict(e['bindings']);bindings.update(p.get('auxiliary_bindings',{}))
    dependencies=provenance.get('dependencies',{})
    required={'kernel.cubin','execution.json','kernel.ttgir','kernel.ptx','execution_config.json','compile.json','image-identity.json'}
    if set(dependencies)!=required:raise ValueError('complete summary image-dependency set required')
    if native.get('native_json_sha256') is not None and provenance.get('native_json_sha256')!=native['native_json_sha256']:
        raise ValueError('summary/native disassembly document differs')
    for name,value in dependencies.items():
        if bindings.get(name)!=value:raise ValueError('summary compiled-image dependency differs: '+name)
    if use_gf2 and e['gf2']['status']!='available':return unavailable('proven-GF2-address-burden-unavailable')
    counts=dict(values)
    if use_gf2:counts['gf2_global_sectors32']=e['gf2']['global_sectors32']
    return dict(result,status='available',features=counts,reason=None,
        static_summary_complete=True,dynamic_execution_bounds=summary['execution_bounds'],
        interpretation='complete reachable static-site proxy; dynamic work bounds remain separate and do not bound time')


def validate_profile(profile):
    if isinstance(profile,dict) and profile.get('model')==ADDRESS_MARGIN_MODEL:return validate_address_margin_profile(profile)
    if not isinstance(profile,dict) or profile.get('model') not in (MODEL,SUMMARY_MODEL,LOCALITY_MODEL,SUMMARY_LOCALITY_MODEL) or profile.get('contract')!=CONTRACT or profile.get('schema_version')!=1:
        raise ValueError('joint profile family differs')
    if profile.get('implementation_sha256')!=implementation():raise ValueError('joint profile implementation changed')
    expected_names,expected_units=feature_contract(profile)
    if profile.get('features')!=list(expected_names) or profile.get('feature_units')!=expected_units:raise ValueError('joint profile feature units differ')
    if not isinstance(profile.get('analysis_contract'),dict):raise ValueError('joint profile analysis contract missing')
    if not supported_execution_domain(profile.get('execution_domain')):raise ValueError('joint profile single-CTA execution domain missing or unsupported')
    expected_dtype='float16' if profile.get('cost_family')=='sm90-immediate-wait-wgmma-v1' else 'float32'
    if profile['execution_domain']['dtype']!=expected_dtype:raise ValueError('joint model precision family differs')
    for key in ('coefficients_us_per_unit','training_maxima'):
        f=profile.get(key)
        if not isinstance(f,dict) or set(f)!=set(expected_names) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in f.values()):raise ValueError('invalid joint '+key)
    if profile.get('uses_observed_development_timings') is not True or profile.get('hardware_time_bound') is not False:
        raise ValueError('joint profile interpretation differs')
    risk=profile.get('risk_calibration',{})
    if risk.get('method') not in ('leave-one-graph-out-max-pair-log-ratio-residual-v1','out-of-program-selection-margin-v1') or risk.get('hardware_time_bound') is not False:
        raise ValueError('joint risk contract differs')
    q=risk.get('log_ratio_error_envelope')
    if risk.get('method')=='out-of-program-selection-margin-v1':
        margin=risk.get('minimum_predicted_gain')
        if margin is not None and (type(margin) not in (float,int) or not math.isfinite(margin) or not 0<=margin<=1):raise ValueError('invalid calibrated gain threshold')
    if q is not None and (type(q) not in (float,int) or not math.isfinite(q) or q<0):raise ValueError('invalid risk envelope')
    return profile


def _dominates(a,b):
    return all(a[n]<=b[n] for n in a)


ORDER_CONE_CONTRACT='frisk-calibrated-nonnegative-ordering-cone-v1'
ORDER_SELECTION_RULE='frisk-calibrated-order-cone-argmin-or-strict-fallback-improvement-v2'


def verify_order_cone(cone,names):
    """Reject vacuous implications: an exact strict feasible witness is required."""
    if cone.get('contract')!=ORDER_CONE_CONTRACT or cone.get('status')!='available' or cone.get('features')!=list(names):return False
    rows=cone.get('constraints');witness=cone.get('strict_feasible_witness')
    if not isinstance(rows,list) or not rows or any(not isinstance(r,list) or len(r)!=len(names) or any(type(v) is not int for v in r) for r in rows):raise ValueError('ordering cone signed-integer feature units differ')
    if not isinstance(witness,list) or len(witness)!=len(names):raise ValueError('ordering cone feasible witness missing')
    w=[]
    for v in witness:
        if not isinstance(v,list) or len(v)!=2 or any(type(x) is not int for x in v) or v[0]<0 or v[1]<=0:raise ValueError('invalid exact nonnegative cone witness')
        w.append(Fraction(*v))
    if any(sum(a*x for a,x in zip(row,w))>-1 for row in rows):raise ValueError('ordering cone is not strictly feasible under declared witness')
    return True


def exact_order_proof(rows,target):
    """Find and independently verify d=sum(lambda*A)-mu, lambda,mu>=0.

    Training inequalities are A*w<=-1 and w>=0. Thus positive lambda mass
    proves d*w<0 in that calibrated model family. Units are dimensionless score
    and positive common score offsets disappear in the contrast d.
    """
    import numpy as np
    from sympy import Matrix
    from joint_cost_calibration import nnls
    n=len(target);m=len(rows)
    matrix=np.concatenate([np.asarray(rows,dtype=float).T,-np.eye(n)],axis=1)
    try:solution,_=nnls(matrix,np.asarray(target,dtype=float),max_iterations=3000)
    except ValueError:return None
    if float(np.linalg.norm(matrix@solution-target))>1e-5:return None
    active=np.where(solution>1e-12)[0].tolist()
    if not active:return None
    exact=Matrix([[int(row[i]) for i in active] for row in matrix.tolist()])
    try:
        vector,parameters=exact.gauss_jordan_solve(Matrix(target))
        if parameters.rows:vector=vector.subs({p:0 for p in parameters})
    except ValueError:return None
    if any(v<0 for v in vector) or exact*vector!=Matrix(target):return None
    values={i:Fraction(int(v.p),int(v.q)) for i,v in zip(active,vector) if v}
    mass=sum(v for i,v in values.items() if i<m)
    if mass<=0:return None
    lambdas={str(i):[v.numerator,v.denominator] for i,v in values.items() if i<m}
    slack={str(i-m):[v.numerator,v.denominator] for i,v in values.items() if i>=m}
    reconstructed=[sum(Fraction(*v)*rows[int(i)][j] for i,v in lambdas.items())-Fraction(*slack.get(str(j),[0,1])) for j in range(n)]
    if reconstructed!=list(target):raise ValueError('independent ordering-cone dual verification failed')
    return dict(contract=ORDER_CONE_CONTRACT,lambdas=lambdas,nonnegative_prior_slack=slack,
        strict_calibration_mass=[mass.numerator,mass.denominator],exact_dual_equality_verified=True,
        conditional_dimensionless_score_upper_bound=[-mass.numerator,mass.denominator],
        hardware_time_bound=False,units='dimensionless calibrated score, not microseconds/cycles',
        interpretation='conditional on registered calibration inequalities A*w<=-1 and w>=0; not a hardware or generalization theorem')


def verify_order_proof(rows,target,proof):
    """Check a provider's exact proposal independently before any decision."""
    if not isinstance(rows,list) or not rows or not isinstance(target,list) or not target:
        raise ValueError('nonempty exact order dimensions required')
    n=len(target);m=len(rows)
    if any(type(x) is not int for x in target) or any(not isinstance(r,list) or len(r)!=n or any(type(x) is not int for x in r) for r in rows):
        raise ValueError('exact order signed integer units required')
    if not isinstance(proof,dict) or proof.get('contract')!=ORDER_CONE_CONTRACT:
        raise ValueError('order proposal contract differs')
    def coefficients(value,limit):
        if not isinstance(value,dict):raise ValueError('order proposal rational map required')
        parsed={}
        for key,pair in value.items():
            if not isinstance(key,str) or not key.isdecimal() or str(int(key))!=key or not 0<=int(key)<limit:
                raise ValueError('order proposal index differs')
            if not isinstance(pair,list) or len(pair)!=2 or any(type(x) is not int for x in pair) or pair[0]<=0 or pair[1]<=0:
                raise ValueError('order proposal nonnegative rational differs')
            parsed[int(key)]=Fraction(*pair)
        return parsed
    weights=coefficients(proof.get('lambdas'),m);slack=coefficients(proof.get('nonnegative_prior_slack'),n)
    mass=sum(weights.values(),Fraction(0))
    if mass<=0:raise ValueError('order proposal has no strict calibration mass')
    reconstructed=[sum(weight*rows[i][j] for i,weight in weights.items())-slack.get(j,Fraction(0)) for j in range(n)]
    if reconstructed!=target:raise ValueError('order proposal exact equality differs')
    expected=dict(contract=ORDER_CONE_CONTRACT,
        lambdas={str(i):[w.numerator,w.denominator] for i,w in weights.items()},
        nonnegative_prior_slack={str(i):[w.numerator,w.denominator] for i,w in slack.items()},
        strict_calibration_mass=[mass.numerator,mass.denominator],exact_dual_equality_verified=True,
        conditional_dimensionless_score_upper_bound=[-mass.numerator,mass.denominator],
        hardware_time_bound=False,units='dimensionless calibrated score, not microseconds/cycles',
        interpretation='conditional on registered calibration inequalities A*w<=-1 and w>=0; not a hardware or generalization theorem')
    for name in ('strict_calibration_mass','conditional_dimensionless_score_upper_bound'):
        pair=proof.get(name)
        if type(pair) is not list or len(pair)!=2 or any(type(x) is not int for x in pair):
            raise ValueError('order proposal metadata integer pair required')
    for name in ('exact_dual_equality_verified','hardware_time_bound'):
        if type(proof.get(name)) is not bool:raise ValueError('order proposal metadata boolean required')
    if proof!=expected:raise ValueError('order proposal certificate metadata differs')
    return expected


def _order_proof(rows,target,provider):
    if provider is None:return exact_order_proof(rows,target)
    proposal=provider(rows,target)
    if proposal is not None:return verify_order_proof(rows,target,proposal)
    # A cached absence cannot replace the original numerical proposal search.
    proposal=exact_order_proof(rows,target)
    if proposal is not None:
        proposal=verify_order_proof(rows,target,proposal)
        publish=getattr(provider,'record',None)
        if publish is not None:publish(rows,target,proposal)
    return proposal


def ordering_cone_selection(features,profile,fallback,order,*,proof_provider=None):
    cone=profile.get('ordering_cone',{})
    if not verify_order_cone(cone,profile['features']):return None
    coverage=cone.get('leave_program_out',{})
    if coverage.get('proved_pairs',0)<=0 or coverage.get('contradicted_registered_pairs',0)!=0:return None
    rows=cone['constraints'];proofs={};forced=[]
    for arm in order:
        comparisons={};valid=True
        for opponent in order:
            if arm==opponent:continue
            delta=[features[arm][n]-features[opponent][n] for n in profile['features']]
            if not any(delta):comparisons[opponent]=dict(status='identical-model-feature-vector');continue
            if all(value<=0 for value in delta):
                comparisons[opponent]=dict(status='nonnegative-prior-weak-bound',
                    nonnegative_prior_slack={str(i):[-value,1] for i,value in enumerate(delta) if value},
                    strict_calibration_mass=[0,1],conditional_dimensionless_score_upper_bound=[0,1],
                    exact_dual_equality_verified=True,hardware_time_bound=False,
                    interpretation='weak model-family inequality only; this prior bound cannot authorize replacing fallback')
                continue
            proof=_order_proof(rows,delta,proof_provider)
            if proof is None:valid=False;break
            comparisons[opponent]=proof
        if valid:forced.append(arm);proofs[arm]=comparisons
    if not forced:
        # A complete argmin need not be identified to improve the original
        # fallback. Positive-mass calibrated contrasts authorize only a strict
        # model-family improvement, never a claim of optimality.
        improvements={}
        for arm in order:
            if arm==fallback:continue
            delta=[features[arm][n]-features[fallback][n] for n in profile['features']]
            proof=_order_proof(rows,delta,proof_provider)
            if proof is not None:improvements[arm]=proof
        if improvements:
            selected=next(arm for arm in order if arm in improvements)
            comparisons={fallback:improvements[selected]};unresolved=[]
            for opponent in order:
                if opponent in (selected,fallback):continue
                delta=[features[selected][n]-features[opponent][n] for n in profile['features']]
                if not any(delta):comparisons[opponent]=dict(status='identical-model-feature-vector')
                elif all(value<=0 for value in delta):
                    comparisons[opponent]=dict(status='nonnegative-prior-weak-bound',strict_calibration_mass=[0,1])
                else:
                    proof=_order_proof(rows,delta,proof_provider)
                    if proof is not None:comparisons[opponent]=proof
                    else:unresolved.append(opponent)
            return dict(status='calibrated-cone-fallback-improvement-proved',selected_source=selected,
                decision_rule_contract=ORDER_SELECTION_RULE,argmin_proved=False,
                forced_argmin_members=[],complete_pool=list(order),unresolved_opponents=unresolved,
                strict_improvement_candidates=list(improvements),strict_fallback_improvement=improvements[selected],
                comparisons=comparisons,training_constraints_sha256=digest(rows),
                leave_program_out=coverage,uses_gpu_timing=False,hardware_time_bound=False,
                selection_rule='first strictly improving candidate in fixed input priority',
                common_offset_invariant=True,units='dimensionless score under calibrated ordering cone',
                claim='strict improvement over the original fallback in the calibrated model family; unresolved opponents may be faster; no hardware/unseen-workload guarantee')
        return dict(status='not-proved',selected_source=fallback,decision_rule_contract=ORDER_SELECTION_RULE,
            argmin_proved=False,complete_pool=list(order),forced_argmin_members=[],uses_gpu_timing=False,hardware_time_bound=False)
    strict_improvements={}
    if fallback not in forced:
        for arm in forced:
            delta=[features[arm][n]-features[fallback][n] for n in profile['features']]
            proof=_order_proof(rows,delta,proof_provider)
            if proof is not None:strict_improvements[arm]=proof
        if not strict_improvements:
            return dict(status='argmin-proved-without-strict-fallback-improvement',selected_source=fallback,
                decision_rule_contract=ORDER_SELECTION_RULE,argmin_proved=False,complete_pool=list(order),
                forced_argmin_members=forced,comparisons={a:proofs[a] for a in forced},
                reason='strict-calibrated-improvement-over-fallback-not-proved',uses_gpu_timing=False,hardware_time_bound=False)
    selected=fallback if fallback in forced else next(arm for arm in order if arm in strict_improvements)
    return dict(status='calibrated-cone-argmin-proved',selected_source=selected,forced_argmin_members=forced,
        decision_rule_contract=ORDER_SELECTION_RULE,argmin_proved=True,complete_pool=list(order),unresolved_opponents=[],
        comparisons=proofs[selected],strict_fallback_improvement=strict_improvements.get(selected),
        training_constraints_sha256=digest(rows),leave_program_out=coverage,
        uses_gpu_timing=False,hardware_time_bound=False,common_offset_invariant=True,
        units='dimensionless score under calibrated ordering cone',
        claim='weak inequalities prove argmin membership; replacing fallback additionally requires positive-mass calibrated strict proof; no hardware/unseen-workload guarantee')


def select(natives,evidences,profile,fallback_arm,priority_order=None,*,max_relative_risk=1.0,summaries=None,allow_empirical_extrapolation=False,localities=None,proof_provider=None):
    """Choose with an auditable conditional certificate or explicit fallback.

    Component dominance may prove an argmin over every nonnegative coefficient
    inside the declared model family. It never bypasses empirical risk: the
    winner replaces fallback only when the separate risk rule accepts it.
    Neither the model certificate nor the risk rule bounds GPU time.
    Inference consumes only native/address evidence plus the frozen profile.
    """
    validate_profile(profile)
    if type(allow_empirical_extrapolation) is not bool:raise ValueError('explicit boolean exploration policy required')
    names=set(natives)
    order=list(priority_order or sorted(names))
    if set(evidences)!=names or set(order)!=names or len(order)!=len(names) or fallback_arm not in names:raise ValueError('complete pool/order required')
    if type(max_relative_risk) not in (float,int) or not math.isfinite(max_relative_risk) or max_relative_risk<1:raise ValueError('risk threshold must be finite and >=1')
    # Validate comparability across the entire pool, not only individual arms.
    priority._evidence({n:_open_evidence(evidences[n])[0]['preceding'] for n in order})
    if profile['model'] in (SUMMARY_MODEL,SUMMARY_LOCALITY_MODEL):
        if summaries is None or set(summaries)!=names:raise ValueError('complete per-arm summaries required')
        rows={n:summary_features(natives[n],summaries[n],evidences[n],profile) for n in order}
    else:rows={n:features(natives[n],evidences[n],profile['analysis_contract']) for n in order}
    if profile.get('locality_features'):
        if localities is None or set(localities)!=names:raise ValueError('complete per-arm locality evidence required')
        rows={n:extend_locality(rows[n],localities[n],evidences[n],profile) for n in order}
        if all(r['status']=='available' for r in rows.values()) and not locality_reduction_holds(rows,localities,profile,order):
            rows={n:dict(r,status='unsupported',features=None,reason='locality-cone-certificate-domain-differs') for n,r in rows.items()}
    for n in order:
        if _open_evidence(evidences[n])[0].get('execution_domain')!=profile['execution_domain']:
            rows[n]=dict(rows[n],status='unsupported',features=None,reason='execution-domain-outside-calibration')
    reasons=[n+':'+r['reason'] for n,r in rows.items() if r['status']!='available']
    binding=dict(profile_sha256=digest(profile),natives={n:digest(natives[n]) for n in order},
        evidence={n:digest(evidences[n]) for n in order},fallback_arm=fallback_arm,priority=order,
        max_relative_risk=max_relative_risk,allow_empirical_extrapolation=allow_empirical_extrapolation)
    if summaries is not None:binding['summaries']={n:digest(summaries[n]) for n in order}
    if localities is not None:binding['localities']={n:digest(localities[n]) for n in order}
    result=dict(schema_version=1,contract=CONTRACT,implementation_sha256=implementation(),binding=binding,
        binding_sha256=digest(binding),selected_source=fallback_arm,status='fallback',reasons=reasons,
        uses_gpu_timing=False,hardware_time_bound=False,basis=BASIS,rows=rows)
    if reasons:return _seal(result)
    f={n:r['features'] for n,r in rows.items()}
    activated={n:[k for k in profile['features'] if f[n][k]>0 and profile['training_maxima'][k]==0] for n in order}
    if any(activated.values()):
        result['reasons']=['training-zero-feature-activation'];result['untrained_features']=activated;return _seal(result)
    cone=ordering_cone_selection(f,profile,fallback_arm,order,proof_provider=proof_provider)
    if cone is not None:
        result['ordering_cone_certificate']=cone
        if cone.get('status') in ('calibrated-cone-argmin-proved','calibrated-cone-fallback-improvement-proved'):
            selected=cone['selected_source']
            result.update(status=('ordering-cone-improvement-certified' if cone['status']=='calibrated-cone-fallback-improvement-proved' else 'ordering-cone-certified') if selected!=fallback_arm else 'fallback-cone-agrees',selected_source=selected,reasons=[])
            return _seal(result)
    universal=[n for n in order if all(_dominates(f[n],f[b]) for b in order)]
    if universal:
        selected=universal[0]
        result['model_certificate']=dict(status='nonnegative-family-argmin-proved',
            proved_candidate=selected,universal_argmin_members=universal,comparisons={n:{k:f[selected][k]-f[n][k] for k in profile['features']} for n in order},
            hardware_time_bound=False,bypasses_empirical_risk=False,
            claim='coordinatewise nonpositive differences prove an argmin only inside the declared additive model family; empirical acceptance is independently required')
    weights=profile['coefficients_us_per_unit'];scores={n:sum(f[n][k]*weights[k] for k in profile['features']) for n in order}
    chosen=min(order,key=lambda n:scores[n]);q=profile['risk_calibration'].get('log_ratio_error_envelope')
    result.update(fitted_scores_us=scores,fitted_winner=chosen,coefficient_identifiability=profile['identifiability']['interpretation'])
    risk=profile['risk_calibration']
    if risk['method']=='out-of-program-selection-margin-v1':
        threshold=risk.get('minimum_predicted_gain')
        gain=1-scores[chosen]/scores[fallback_arm] if scores[fallback_arm]>0 else None
        margin_passed=gain is not None and threshold is not None and gain>threshold and chosen!=fallback_arm
        acceptance_supported=risk.get('accepted_oof_proposals',0)>0
        passed=margin_passed and (acceptance_supported or allow_empirical_extrapolation)
        result['risk_certificate']=dict(method=risk['method'],predicted_gain=gain,minimum_predicted_gain=threshold,
            passed=passed,margin_passed=margin_passed,acceptance_supported=acceptance_supported,
            allow_empirical_extrapolation=allow_empirical_extrapolation,calibration_summary=risk,hardware_time_bound=False,
            interpretation='empirical OOF selection-margin rule; calibration labels are absent from inference and this is not a future time/coverage guarantee')
        if chosen==fallback_arm:result.update(status='fallback-model-agrees',reasons=['fitted-winner-is-fallback'])
        elif passed:result.update(status='empirical-risk-selected' if acceptance_supported else 'empirical-risk-exploration',selected_source=chosen,reasons=[])
        else:result['reasons']=['empirical-acceptance-support-unavailable' if margin_passed and not acceptance_supported else 'empirical-selection-margin-not-met']
        return _seal(result)
    if scores[fallback_arm]<=0 or scores[chosen]<=0 or q is None:
        result['reasons']=['risk-calibration-or-positive-score-unavailable'];return _seal(result)
    log_upper=math.log(scores[chosen]/scores[fallback_arm])+q
    risk_proxy=math.exp(log_upper) if log_upper<700 else None
    result['risk_certificate']=dict(predicted_ratio=scores[chosen]/scores[fallback_arm],
        calibration_log_error=q,empirical_upper_ratio_proxy=risk_proxy,threshold=max_relative_risk,
        passed=log_upper<=math.log(max_relative_risk),method=profile['risk_calibration']['method'],
        calibration_groups=profile['risk_calibration']['graph_groups'],hardware_time_bound=False,
        interpretation='deterministic test against observed graph-held-out residual envelope; no statistical out-of-domain coverage guarantee')
    if chosen==fallback_arm:
        result.update(status='fallback-model-agrees',reasons=['fitted-winner-is-fallback'])
    elif log_upper<=math.log(max_relative_risk):
        result.update(status='empirical-risk-selected',selected_source=chosen,reasons=[])
    else:result['reasons']=['empirical-risk-threshold-not-met']
    return _seal(result)


def verify_selection(receipt,natives,evidences,profile,fallback_arm,priority_order=None,*,max_relative_risk=1.0,summaries=None,allow_empirical_extrapolation=False,localities=None,proof_provider=None):
    expected=select(natives,evidences,profile,fallback_arm,priority_order,max_relative_risk=max_relative_risk,summaries=summaries,allow_empirical_extrapolation=allow_empirical_extrapolation,localities=localities,proof_provider=proof_provider)
    if receipt!=expected:raise ValueError('joint decision/certificate changed')
    return expected


def validate_address_margin_profile(profile):
    import access_reuse as locality
    import address_intervention as intervention
    names,units=feature_contract(profile)
    expected=dict(contract=intervention.CONTRACT,analyzer_sha256=hashlib.sha256(Path(intervention.__file__).read_bytes()).hexdigest())
    local=dict(contract=locality.CONTRACT,cost_family=locality.FAMILY,analyzer_sha256=hashlib.sha256(Path(locality.__file__).read_bytes()).hexdigest())
    if profile.get('schema_version')!=1 or profile.get('contract')!=CONTRACT or profile.get('implementation_sha256')!=implementation():raise ValueError('address margin profile implementation differs')
    if profile.get('features')!=list(names) or profile.get('feature_units')!=units or profile.get('intervention_analysis_contract')!=expected or profile.get('locality_analysis_contract')!=local:raise ValueError('address margin feature/proof units differ')
    if not supported_execution_domain(profile.get('execution_domain')) or profile['execution_domain']['dtype']!='float32':raise ValueError('address margin execution domain differs')
    if profile.get('hardware_time_bound') is not False or profile.get('uses_observed_development_timings') is not True:raise ValueError('address margin interpretation differs')
    return profile


def intervention_features(native,evidence,locality_document,document,profile):
    """Validate the two independent, static controlled-intervention proofs."""
    validate_address_margin_profile(profile)
    if not isinstance(document,dict) or document.get('status')!='available':return dict(status='unsupported',reason='controlled-source-proof-unavailable')
    if document.get('summary_sha256')!=digest({k:v for k,v in document.items() if k!='summary_sha256'}):raise ValueError('controlled intervention payload changed')
    import access_reuse as locality
    import address_intervention as intervention
    provenance=document.get('provenance',{})
    actual=dict(contract=document.get('contract'),analyzer_sha256=provenance.get('analyzer_sha256'))
    if actual!=profile['intervention_analysis_contract']:return dict(status='unsupported',reason='controlled-source-analyzer-differs')
    if provenance.get('source_dependencies')!=intervention.SUPPORT_SOURCES:raise ValueError('controlled source dependencies differ')
    for name,value in provenance['source_dependencies'].items():
        if hashlib.sha256((Path(intervention.__file__).parent/name).read_bytes()).hexdigest()!=value:return dict(status='unsupported',reason='controlled-source-dependency-changed')
    p,e=_open_evidence(evidence)
    if digest(native)!=e['native_sha256'] or provenance.get('native_json_sha256')!=native.get('native_json_sha256'):raise ValueError('controlled intervention native identity differs')
    bindings=dict(e['bindings']);bindings.update(p.get('auxiliary_bindings',{}))
    required={'correctness.json','execution_config.json','image-identity.json','kernel.cubin','kernel.ptx','native-analysis.json'}
    if set(provenance.get('dependencies',{}))!=required:raise ValueError('controlled intervention dependency set differs')
    if any(bindings.get(n)!=v for n,v in provenance['dependencies'].items()):raise ValueError('controlled intervention image binding differs')
    claims=document.get('claims',{})
    if claims!=dict(controlled_source_global_address_intervention=True,complete_native_static_work_and_resources=True,
        native_address_du_proven=False,dynamic_lane_masks_or_slowpath_counts_equal=False,native_cache_or_tlb_misses_proven=False):raise ValueError('controlled intervention scope differs')
    group=digest(dict(source=document['source']['nonaddress_sha256'],native=document['native']['skeleton_sha256'],resources=document['resources']))
    if group!=document.get('controlled_group_sha256'):raise ValueError('controlled intervention group identity differs')
    if p.get('execution_domain')!=profile['execution_domain']:return dict(status='unsupported',reason='execution-domain-outside-calibration')
    prototype=dict(locality_features=list(locality.FOOTPRINT_FEATURES),locality_analysis_contract=profile['locality_analysis_contract'])
    checked=extend_locality(dict(status='available',features={}),locality_document,evidence,prototype)
    if checked['status']!='available':return checked
    if document.get('locality_sha256')!=locality_document['summary_sha256']:raise ValueError('controlled intervention locality binding differs')
    if e['gf2']['status']!='available':return dict(status='unsupported',reason='proven-GF2-burden-unavailable')
    return dict(status='available',controlled_group_sha256=group,
        features=dict(checked['features'],gf2_global_sectors32=e['gf2']['global_sectors32']),
        dynamic_mask_equivalence=False,native_address_du_proven=False)


def address_margin_decision(natives,evidences,profile,fallback,order,localities,interventions,*,proof_provider=None):
    binding=dict(profile_sha256=digest(profile),fallback=fallback,priority=order,
        localities={n:digest(localities[n]) for n in order} if localities is not None and set(localities)==set(order) else None,
        interventions={n:digest(interventions[n]) for n in order} if interventions is not None and set(interventions)==set(order) else None)
    result=dict(contract=CONTRACT,kind='controlled-address-margin',status='fallback',selected_source=fallback,
        binding=binding,uses_gpu_timing=False,hardware_time_bound=False,
        units='dimensionless calibrated marginal score; common non-address work cancels',
        scope='controlled source Global-address intervention and native static work/resource equality; no dynamic mask/cache equivalence')
    if localities is None or interventions is None or set(localities)!=set(order) or set(interventions)!=set(order):return _seal(dict(result,reason='controlled-evidence-unavailable'))
    rows={n:intervention_features(natives[n],evidences[n],localities[n],interventions[n],profile) for n in order}
    if rows[fallback]['status']!='available':return _seal(dict(result,reason=rows[fallback].get('reason','controlled-fallback-proof-unavailable'),rows=rows))
    group=rows[fallback]['controlled_group_sha256'];eligible=[n for n in order if rows[n]['status']=='available' and rows[n]['controlled_group_sha256']==group]
    if len(eligible)<2:return _seal(dict(result,reason='no-other-candidate-in-controlled-group',eligible=eligible,rows=rows))
    proof=ordering_cone_selection({n:rows[n]['features'] for n in eligible},profile,fallback,eligible,proof_provider=proof_provider)
    result.update(rows=rows,eligible=eligible,controlled_group_sha256=group,ordering_cone_certificate=proof)
    if proof and proof.get('status') in ('calibrated-cone-argmin-proved','calibrated-cone-fallback-improvement-proved'):
        selected=proof['selected_source'];result.update(status='controlled-address-cone-certified' if selected!=fallback else 'controlled-address-cone-agrees',selected_source=selected)
    else:result['reason']='controlled-marginal-order-not-proved'
    return _seal(result)


def _base_dispatch(natives,evidences,profiles,fallback_arm,priority_order=None,*,summaries=None,allow_empirical_extrapolation=False,localities=None,proof_provider=None):
    """Whole-pool cost-family dispatch; never mix incomparable family weights."""
    names=set(natives);order=list(priority_order or sorted(names))
    if set(evidences)!=names or set(order)!=names or len(order)!=len(names) or fallback_arm not in names:raise ValueError('complete dispatch pool required')
    primary=profiles.get('native15')
    if primary is not None and all(features(natives[n],evidences[n],primary['analysis_contract'])['status']=='available' for n in order):
        selected=select(natives,evidences,primary,fallback_arm,order,allow_empirical_extrapolation=allow_empirical_extrapolation,localities=localities,proof_provider=proof_provider)
        return _seal(dict(schema_version=1,contract=CONTRACT,kind='family-dispatch',cost_family='native15',decision=selected,
            selected_source=selected['payload']['selected_source'],status=selected['payload']['status'],uses_gpu_timing=False,hardware_time_bound=False))
    if summaries is not None and set(summaries)==names and all(isinstance(summaries[n],dict) and summaries[n].get('status')=='complete' for n in order):
        family=summaries[order[0]]['cost_family']
        if family in profiles and all(summaries[n]['cost_family']==family for n in order):
            selected=select(natives,evidences,profiles[family],fallback_arm,order,summaries=summaries,allow_empirical_extrapolation=allow_empirical_extrapolation,localities=localities,proof_provider=proof_provider)
            return _seal(dict(schema_version=1,contract=CONTRACT,kind='family-dispatch',cost_family=family,decision=selected,
                selected_source=selected['payload']['selected_source'],status=selected['payload']['status'],uses_gpu_timing=False,hardware_time_bound=False))
    return _seal(dict(schema_version=1,contract=CONTRACT,kind='family-dispatch',cost_family=None,
        selected_source=fallback_arm,status='fallback',reason='no-common-supported-calibrated-family',
        binding=dict(natives={n:digest(natives[n]) for n in order},evidence={n:digest(evidences[n]) for n in order},
            summaries={n:digest(summaries[n]) for n in order} if summaries is not None and set(summaries)==names else None,
            profiles={k:digest(v) for k,v in profiles.items()},fallback_arm=fallback_arm,priority=order),
        uses_gpu_timing=False,hardware_time_bound=False))


def dispatch(natives,evidences,profiles,fallback_arm,priority_order=None,*,summaries=None,allow_empirical_extrapolation=False,localities=None,interventions=None,proof_provider=None):
    base=_base_dispatch(natives,evidences,profiles,fallback_arm,priority_order,summaries=summaries,
        allow_empirical_extrapolation=allow_empirical_extrapolation,localities=localities,proof_provider=proof_provider)
    profile=profiles.get('address-margin')
    if profile is None:return base
    order=list(priority_order or sorted(natives));selected=base['payload']['selected_source']
    priority=[selected]+[n for n in order if n!=selected]
    margin=address_margin_decision(natives,evidences,profile,selected,priority,localities,interventions,proof_provider=proof_provider)
    payload=dict(base['payload'],controlled_address_margin=margin)
    if margin['payload']['selected_source']!=selected:payload.update(selected_source=margin['payload']['selected_source'],status='controlled-address-cone-certified')
    return _seal(payload)


def verify_dispatch(receipt,natives,evidences,profiles,fallback_arm,priority_order=None,*,summaries=None,allow_empirical_extrapolation=False,localities=None,interventions=None,proof_provider=None):
    expected=dispatch(natives,evidences,profiles,fallback_arm,priority_order,summaries=summaries,allow_empirical_extrapolation=allow_empirical_extrapolation,localities=localities,interventions=interventions,proof_provider=proof_provider)
    if receipt!=expected:raise ValueError('family dispatch certificate changed')
    return expected


def load_profile(path,*,verify_training_dependencies=False):
    """Load timing-free profile; offline audit may verify all training hashes.

    The frozen envelope binds training inputs, but runtime never opens label or
    measurement files. Source dependencies are always verified.
    """
    path=Path(path);envelope=json.loads(path.read_text())
    if set(envelope)!={'payload','payload_sha256'} or digest(envelope['payload'])!=envelope['payload_sha256']:
        raise ValueError('joint profile envelope differs')
    profile=validate_profile(envelope['payload'])
    for dependency in profile['dependencies']:
        if dependency.get('role')!='source' and not verify_training_dependencies:continue
        p=Path(dependency['path'])
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=dependency['sha256']:
            raise ValueError('joint training/profile dependency changed: '+str(p))
    return profile


def load_bundle(path,*,verify_training_dependencies=False):
    path=Path(path);bundle=json.loads(path.read_text())
    if bundle.get('contract')!=CONTRACT or bundle.get('schema_version')!=1:raise ValueError('joint profile bundle contract differs')
    profiles={}
    for family,record in bundle['profiles'].items():
        source=Path(record['path'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('joint bundle profile changed')
        profiles[family]=load_profile(source,verify_training_dependencies=verify_training_dependencies)
    return profiles
