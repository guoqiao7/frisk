"""Offline, graph-held-out calibration of the Frisk native14+GF2 model.

The R3 archive is observed development data. Reading labels is allowed ONLY in
this module; timing-free inference is in joint_cost_model. No CUDA, compilation,
or writes under input archive roots occur. All model coefficients live over R.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
import re
from pathlib import Path
import numpy as np
import joint_cost_model as model
import native_cost_calibration as native_calibration


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def nnls(a,b,tolerance=1e-10,max_iterations=10000):
    """Lawson-Hanson passive-set NNLS with column scaling and KKT receipt.

    No inverse or normal equations are formed. Rank-deficient passive sets use
    numpy.linalg.lstsq's SVD minimum-norm solution. Non-uniqueness is diagnosed
    independently; the selected NNLS representative is never called identified.
    """
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.ndim!=2 or b.shape!=(a.shape[0],) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('finite NNLS matrix required')
    scale=np.linalg.norm(a,axis=0);scale[scale==0]=1.;z=a/scale
    x=np.zeros(a.shape[1]);passive=np.zeros(a.shape[1],dtype=bool);iterations=0
    eps=tolerance*max(1.,float(np.linalg.norm(z.T@b,np.inf)))
    while True:
        gradient=z.T@(b-z@x);eligible=np.where(~passive,gradient,-np.inf)
        if eligible.max(initial=-np.inf)<=eps:break
        passive[int(np.argmax(eligible))]=True
        while True:
            iterations+=1
            if iterations>max_iterations:raise ValueError('NNLS passive-set iteration limit')
            proposal=np.zeros_like(x)
            proposal[passive]=np.linalg.lstsq(z[:,passive],b,rcond=1e-12)[0]
            if np.all(proposal[passive]>eps):x=proposal;break
            nonpositive=passive&(proposal<=eps)
            denominators=x[nonpositive]-proposal[nonpositive]
            valid=denominators>0
            alpha=float(np.min(x[nonpositive][valid]/denominators[valid])) if valid.any() else 0.
            x=x+alpha*(proposal-x)
            removed=passive&(x<=eps);passive[removed]=False;x[removed]=0.
        if iterations>max_iterations:raise ValueError('NNLS did not converge')
    x[x<0]=0.;residual=z@x-b;gradient=z.T@residual
    kkt=max(float(np.max(np.maximum(-gradient[x<=eps],0),initial=0)),float(np.max(np.abs(gradient[x>eps]),initial=0)))
    return x/scale,dict(method='scaled-Lawson-Hanson-passive-set-SVD-v1',iterations=iterations,
        squared_loss=float(residual@residual),scaled_kkt_residual=kkt,tolerance=eps,
        converged=bool(kkt<=max(1e-7,100*eps)),column_scales=scale.tolist())


def exact_linear_algebra(rows,names):
    """Rational RREF gives reproducible rank and exact coefficient alias vectors."""
    matrix=[[Fraction(int(v)) for v in row] for row in rows];columns=len(names);pivots=[];at=0
    for column in range(columns):
        pivot=next((i for i in range(at,len(matrix)) if matrix[i][column]),None)
        if pivot is None:continue
        matrix[at],matrix[pivot]=matrix[pivot],matrix[at];v=matrix[at][column]
        matrix[at]=[x/v for x in matrix[at]]
        for i in range(len(matrix)):
            if i!=at and matrix[i][column]:
                f=matrix[i][column];matrix[i]=[x-f*y for x,y in zip(matrix[i],matrix[at])]
        pivots.append(column);at+=1
        if at==len(matrix):break
    free=[j for j in range(columns) if j not in pivots];basis=[]
    for j in free:
        v=[Fraction(0)]*columns;v[j]=Fraction(1)
        for i,pivot in enumerate(pivots):v[pivot]=-matrix[i][j]
        denominator=math.lcm(*(x.denominator for x in v));integer=[int(x*denominator) for x in v]
        divisor=math.gcd(*integer);integer=[x//divisor for x in integer]
        basis.append({n:v for n,v in zip(names,integer) if v})
    aliased=sorted({n for vector in basis for n in vector})
    return dict(field='Q (ordinary characteristic-zero linear algebra; not GF2)',
        rank=len(pivots),columns=columns,rows=len(rows),pivot_columns=[names[j] for j in pivots],
        exact_nullspace_basis=basis,alias_involved_features=aliased,
        zero_columns=[n for j,n in enumerate(names) if all(row[j]==0 for row in rows)],
        interpretation='empirical coefficient identifiability only; exact column rank does not establish a physical per-unit cost')


# Optional native15-only auxiliary pair equations. They never supply a fake
# absolute GF2 value and never apply to the different-unit summary families.
_NATIVE_MICRO_PAIRS=[]
_NATIVE_MICRO_MAXIMA={}
_NATIVE_MICRO_DEPENDENCIES=[]


def micro_global_skeleton(text):
    """Verify the full fixed PTX global-address prologue and no later writes."""
    lines=[line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith('//')]
    expected=['ld.param.u64 %q,[Q];','ld.param.u64 %o,[O];','mov.u32 %t,%tid.x;',
        'mul.wide.u32 %i,%t,4;','add.u64 %qp,%q,%i;','add.u64 %op,%o,%i;']
    global_ops=[line for line in lines if 'ld.global' in line or 'st.global' in line]
    if global_ops!=['ld.global.u32 %x,[%qp];','st.global.u32 [%op],%x;']:raise ValueError('micro global skeleton differs')
    writes=[];address_registers={'%q','%o','%t','%i','%qp','%op'}
    for line in lines:
        if not line.endswith(';') or line.startswith('.') or line in ('ret;','exit;'):continue
        parsed=re.fullmatch(r'([A-Za-z][\w.]*)\s+(.+);',line)
        if not parsed:raise ValueError('micro PTX instruction grammar differs')
        opcode,operands=parsed.groups()
        if opcode.startswith(('st.','bar.')):continue
        destination=operands.split('}',1)[0]+'}' if operands.startswith('{') else operands.split(',',1)[0]
        if set(re.findall(r'%[A-Za-z_][\w.]*',destination)) & address_registers:writes.append(line)
    if writes!=expected:raise ValueError('micro global address definitions or later writes differ')
    return dict(global_ops=global_ops,address_definitions=writes,
        proof='fixed Q/O base plus 4*tid address DAG; address registers have exactly these six writes')


def load_native_micro_pairs(labels_path,expected_contract):
    global _NATIVE_MICRO_PAIRS,_NATIVE_MICRO_MAXIMA,_NATIVE_MICRO_DEPENDENCIES
    labels_path=Path(labels_path);completion=read(labels_path.parent/'completion.json')
    if completion.get('status')!='complete' or completion.get('labels_sha256')!=sha(labels_path):raise ValueError('micro labels completion differs')
    labels=read(labels_path)
    if any(native_calibration.analysis_contract(r['native'])!=expected_contract for r in labels):raise ValueError('micro/native analysis contracts differ')
    baseline=next((r for r in labels if r['spec']['kind']=='baseline'),None)
    if baseline is None:raise ValueError('micro baseline missing')
    skeletons=[];dependencies=[dict(path=str(labels_path),sha256=sha(labels_path)),dict(path=str(labels_path.parent/'completion.json'),sha256=sha(labels_path.parent/'completion.json'))]
    for row in labels:
        if row.get('features')!=native_calibration.model_features(row['native']):raise ValueError('micro native feature classification differs')
        if any(type(v) is not int or v<0 for v in row['features'].values()):raise ValueError('micro static native units differ')
        path=Path(row['path'])/'kernel.ptx';text=path.read_text()
        # The declared address registers are defined solely in the common
        # prologue. Reject any later write to them, changed loads/stores, or an
        # extra global operation. This proves the GLOBAL address work cancels
        # in every communication contrast without guessing its absolute g.
        skeletons.append(micro_global_skeleton(text))
        dependencies.append(dict(path=str(path),sha256=sha(path)))
        cubin=Path(row['path'])/'kernel.cubin'
        if sha(cubin)!=row['cubin_sha256']:raise ValueError('micro image differs')
        dependencies.append(dict(path=str(cubin),sha256=sha(cubin)))
    if any(s!=skeletons[0] for s in skeletons):raise ValueError('micro common global address definitions differ')
    pairs=[]
    for kind in ('barrier','shared','shuffle'):
        items=[baseline]+[r for r in labels if r['spec']['kind']==kind];combinations=list(itertools.combinations(items,2))
        normalizer=float(np.median([r['median_us'] for r in items]))
        for a,b in combinations:
            delta={n:a['features'][n]-b['features'][n] for n in model.NATIVE_FEATURES};delta['gf2_global_sectors32']=0
            target=float(np.median(np.array(a['batch_medians_us'])-b['batch_medians_us']))
            weight=1/(max(abs(target),.05*normalizer)*math.sqrt(len(combinations)))
            pairs.append(dict(kind=kind,first=a['spec'],second=b['spec'],delta=delta,target_us=target,weight=weight,
                global_cancellation_proof='identical verified PTX global address skeleton; unknown common absolute g cancels'))
    _NATIVE_MICRO_PAIRS=pairs;_NATIVE_MICRO_MAXIMA={n:max(r['features'][n] for r in labels) for n in model.NATIVE_FEATURES}
    _NATIVE_MICRO_DEPENDENCIES=dependencies
    return dict(pair_equations=len(pairs),images=len(labels),global_skeleton=skeletons[0],
        basis='native15 only; paired differences, no absolute micro GF2 value invented')


def feature_observed_in_training(name,rows,names):
    return any(r['features'][name]>0 for r in rows) or tuple(names[:len(model.FEATURES)])==tuple(model.FEATURES) and _NATIVE_MICRO_MAXIMA.get(name,0)>0


def collect(run):
    """Load only existing validated R3 data; do not import archive code."""
    run=Path(run);manifest=read(run/'manifest.json');quality=read(run/'group-quality.json');aliases=read(run/'measurement-aliases.json')['groups'];pools=read(run/'candidate-pools.json')
    rows=[];dependencies=[]
    for n in ('manifest.json','group-quality.json','measurement-aliases.json','candidate-pools.json','partial-decisions.json'):
        dependencies.append(dict(path=str(run/n),sha256=sha(run/n)))
    contracts=[]
    for row in manifest['rows']:
        if row['arm']=='legacy-static':continue
        group=row['scope']+'-'+row['spec']['key'];directory=Path(row['path'])
        native=read(directory/'native-analysis.json');evidence=model.candidate_evidence(directory,row)
        extracted=model.features(native,evidence);identity=aliases[group][row['arm']]['execution_identity']
        batches=quality[group]['image_batch_medians_us'][identity]
        if extracted['status']=='available':contracts.append(extracted['analysis_contract'])
        rows.append(dict(group=group,graph=row['spec']['graph_name'],spec=row['spec'],arm=row['arm'],
            execution_identity=identity,cubin_sha256=row['cubin_sha256'],path=str(directory),
            features=extracted['features'],feature_status=extracted['status'],feature_reason=extracted['reason'],
            static20=evidence['payload']['static20'],native=native,evidence=evidence,
            median_us=float(np.median(batches)),batch_medians_us=batches))
    if not contracts or any(c!=contracts[0] for c in contracts):raise ValueError('uniform native analysis contract required')
    # Relevant proof files are transitively bound by the candidate evidence;
    # record direct path+hash entries so load_profile detects later changes.
    files={x['path']:x for x in dependencies}
    for row in rows:
        for name,value in row['evidence']['payload']['preceding']['payload']['bindings'].items():
            path=str(Path(row['path'])/name);files[path]=dict(path=path,sha256=value)
    return rows,contracts[0],list(files.values()),pools,quality


def deduplicate(rows):
    # One physical program per program-group. Different ABI/cfg identities stay
    # distinct even when CUBIN bytes match. Never count J/LI aliases twice.
    unique={}
    for row in sorted(rows,key=lambda x:(x['group'],x['arm'])):unique.setdefault((row['group'],row['execution_identity']),row)
    return list(unique.values())


def training_design(rows,names=model.FEATURES,absolute_mass=.05,effect_floor=.05):
    rows=deduplicate(rows);groups=defaultdict(list)
    for row in rows:groups[row['group']].append(row)
    design=[];targets=[];deltas=[]
    for group,items in sorted(groups.items()):
        pairs=list(itertools.combinations(items,2));normalizer=max(float(np.median([r['median_us'] for r in items])),1e-12)
        for first,second in pairs:
            delta=np.array([first['features'][n]-second['features'][n] for n in names],dtype=float)
            paired=float(np.median(np.array(first['batch_medians_us'])-second['batch_medians_us']))
            weight=1/(max(abs(paired),effect_floor*normalizer)*math.sqrt(max(1,len(pairs))))
            design.append(delta*weight);targets.append(paired*weight);deltas.append([int(x) for x in delta])
    for row in rows:
        x=np.array([row['features'][n] for n in names],dtype=float);weight=math.sqrt(absolute_mass/max(1,len(rows)))/max(row['median_us'],1e-12)
        design.append(x*weight);targets.append(row['median_us']*weight)
    if tuple(names[:len(model.FEATURES)])==tuple(model.FEATURES):
        for pair in _NATIVE_MICRO_PAIRS:
            delta=[pair['delta'].get(n,0) for n in names];design.append(np.array(delta,dtype=float)*pair['weight']);targets.append(pair['target_us']*pair['weight']);deltas.append(delta)
    if not design:raise ValueError('training design empty')
    return np.asarray(design),np.asarray(targets),deltas


def fit_weights(rows,names=model.FEATURES):
    a,b,_=training_design(rows,names);weights,receipt=nnls(a,b)
    if not receipt['converged']:raise ValueError('joint NNLS KKT verification failed')
    return {n:float(v) for n,v in zip(names,weights)},receipt


def predict(features,weights):return sum(features[n]*weights[n] for n in weights)


def graph_validation(rows,names=model.FEATURES):
    unique=deduplicate(rows);folds=[];residuals=[];weights_list=[]
    for graph in sorted({r['graph'] for r in unique}):
        test=[r for r in unique if r['graph']==graph];test_ids={r['execution_identity'] for r in test}
        train=[r for r in unique if r['graph']!=graph and r['execution_identity'] not in test_ids]
        if not train:
            folds.append(dict(graph=graph,status='unsupported',reason='no-independent-graph-family-training',train_unique=0,test_unique=len(test),pairs=[]));continue
        weights,fit=fit_weights(train,names);weights_list.append(weights);pairs=[];groups=defaultdict(list)
        for row in test:groups[row['group']].append(row)
        unsupported_features=[n for n in names if any(r['features'][n]>0 for r in test) and not feature_observed_in_training(n,train,names)]
        for group,items in sorted(groups.items()):
            for first,second in itertools.combinations(items,2):
                a=predict(first['features'],weights);b=predict(second['features'],weights)
                if min(a,b)<=0:continue
                actual=first['median_us']/second['median_us'];prediction=a/b
                residual=abs(math.log(actual/prediction));residuals.append(residual)
                pairs.append(dict(group=group,first=first['execution_identity'],second=second['execution_identity'],
                    observed_ratio=actual,predicted_ratio=prediction,absolute_log_ratio_error=residual,
                    inversion=(actual-1)*(prediction-1)<0))
        folds.append(dict(graph=graph,train_unique=len(train),test_unique=len(test),
            alias_overlap_removed=sum(r['graph']!=graph and r['execution_identity'] in test_ids for r in unique),
            training_zero_feature_activations=unsupported_features,weights=weights,fit=fit,pairs=pairs))
    return dict(method='leave-one-graph-out-max-pair-log-ratio-residual-v1',graph_groups=[f['graph'] for f in folds],
        log_ratio_error_envelope=max(residuals) if residuals else None,pairs=len(residuals),folds=folds,
        hardware_time_bound=False,statistical_coverage_claim=False,
        interpretation='maximum observed out-of-graph pair ratio residual; development risk proxy, not future coverage or GPU-time bound'),weights_list


def algebraic_diagnostics(rows,names=model.FEATURES):
    unique=deduplicate(rows);vectors=[[r['features'][n] for n in names] for r in unique]
    _,_,delta=training_design(unique,names)
    full=exact_linear_algebra(vectors,list(names));native=exact_linear_algebra([v[:-1] for v in vectors],list(names[:-1]));pair=exact_linear_algebra(delta,list(names))
    equal_witness=[]
    for group in sorted({r['group'] for r in unique}):
        for a,b in itertools.combinations([r for r in unique if r['group']==group],2):
            if all(a['features'][n]==b['features'][n] for n in model.NATIVE_FEATURES) and a['features']['gf2_global_sectors32']!=b['features']['gf2_global_sectors32']:
                equal_witness.append(dict(group=group,first=a['arm'],second=b['arm'],
                    first_identity=a['execution_identity'],second_identity=b['execution_identity'],
                    native14_difference=[0]*14,gf2_difference=a['features']['gf2_global_sectors32']-b['features']['gf2_global_sectors32'],
                    exact_proof='d^T X_native=0 and d^T g != 0; GF2-address feature is not in the Q/R column span of native14 on this dataset'))
    return dict(full_features=full,native14=native,within_program_differences=pair,
        gf2_rank_increment=full['rank']-native['rank'],equal_native_nonzero_gf2_witnesses=equal_witness,
        interpretation='structural non-redundancy does not prove causality, unique weights, or a performance advantage')


def evaluate(rows,pools,quality,profile,old_decisions):
    by={(r['group'],r['arm']):r for r in rows};results={};counts=defaultdict(int)
    for group,pool in pools.items():
        aliases=quality[group]['identity_aliases'];medians={i:float(np.median(v)) for i,v in quality[group]['image_batch_medians_us'].items()}
        arm_ids={arm:r['execution_identity'] for (g,arm),r in by.items() if g==group}
        union=min(medians.values());entry={}
        legacy=quality[group]['selections']['legacy-static']['median_us']
        for mode,info in pool['modes'].items():
            arms=info['arms'];items={a:by[group,a] for a in arms};natives={a:r['native'] for a,r in items.items()};evidence={a:r['evidence'] for a,r in items.items()}
            baseline=old_decisions['groups'][group]['choices'][mode+'-affine-frame']['selected_source']
            receipt=model.select(natives,evidence,profile,baseline,info['static_order'])
            model.verify_selection(receipt,natives,evidence,profile,baseline,info['static_order'])
            selected=receipt['payload']['selected_source'];latencies={a:medians[arm_ids[a]] for a in arms};best=min(latencies.values());value=latencies[selected]
            all_supported=all(r['feature_status']=='available' for r in items.values())
            fitted=min(arms,key=lambda a:predict(items[a]['features'],profile['coefficients_us_per_unit'])) if all_supported else None
            old=old_decisions['groups'][group]['choices'][mode+'-affine-frame']['selected_source']
            entry[mode]=dict(selected=selected,status=receipt['payload']['status'],pool_coverage_ratio=best/union,
                within_pool_selection_ratio=value/best,union_regret_ratio=value/union,relative_to_static=value/latencies[info['fallback']],
                relative_to_legacy=value/legacy,raw_fitted_winner=fitted,raw_fitted_within_pool_ratio=latencies[fitted]/best if fitted else None,
                old_affine_choice=old,old_affine_within_pool_ratio=latencies[old]/best,
                latencies_us=latencies,receipt=receipt)
            counts[receipt['payload']['status']]+=1
        results[group]=entry
    summaries={}
    for mode in ('joint','local'):
        entries=[g[mode] for g in results.values()]
        summaries[mode]={}
        for name in ('pool_coverage_ratio','within_pool_selection_ratio','union_regret_ratio','relative_to_static','relative_to_legacy'):
            values=[e[name] for e in entries];summaries[mode][name]=dict(geometric_mean=math.exp(sum(math.log(v) for v in values)/len(values)),maximum=max(values))
    # Pair ranking audit uses per-program median direction and retains all noise;
    # it is descriptive development evidence, not a significance count.
    ranking=[]
    for group,pool in pools.items():
        for mode,info in pool['modes'].items():
            for a,b in itertools.combinations(info['arms'],2):
                first,second=by[group,a],by[group,b]
                if first['features'] is None or second['features'] is None or first['execution_identity']==second['execution_identity']:continue
                actual=first['median_us']-second['median_us'];score=predict(first['features'],profile['coefficients_us_per_unit'])-predict(second['features'],profile['coefficients_us_per_unit'])
                ranking.append(dict(group=group,mode=mode,first=a,second=b,observed_delta_us=actual,predicted_delta_us=score,
                    inversion=actual*score<0,tie=score==0,observed_ratio=first['median_us']/second['median_us']))
    return dict(dataset_role='observed-R3-development-training-and-regression; not fresh holdout',groups=results,
        summary=summaries,status_counts=dict(counts),ranking=ranking,ranking_inversions=sum(p['inversion'] for p in ranking),
        ranking_ties=sum(p['tie'] for p in ranking),hardware_time_bounds=False,
        qualification_pass_claim=False,scope='old paid retention pools only; new retention coverage must be evaluated separately')


def selection_margin_calibration(rows,pools,old_decisions,names):
    """OOF empirical threshold for the decision being made, not all pair times.

    Each complete program instance is held out, every test execution-identity
    alias is removed from training, and the old timing-free policy is fallback.
    A fixed 1% material-loss definition is registered for this development rule.
    The threshold is above every observed harmful OOF gain; its reuse still
    needs a genuinely unseen qualification and is no hardware/coverage bound.
    """
    unique=deduplicate(rows);folds=[];decisions=[]
    by={(r['group'],r['arm']):r for r in rows}
    groups=sorted({r['group'] for r in unique})
    for group in groups:
        test=[r for r in unique if r['group']==group];ids={r['execution_identity'] for r in test}
        train=[r for r in unique if r['group']!=group and r['execution_identity'] not in ids]
        if not train:continue
        weights,fit=fit_weights(train,names)
        for mode,info in pools[group]['modes'].items():
            if any((group,a) not in by for a in info['arms']):continue
            baseline=old_decisions['groups'][group]['choices'][mode+'-affine-frame']['selected_source']
            items={a:by[group,a] for a in info['arms']}
            activated=[n for n in names if any(r['features'][n]>0 for r in items.values()) and not feature_observed_in_training(n,train,names)]
            order=[baseline]+[a for a in info['static_order'] if a!=baseline]
            scores={a:predict(items[a]['features'],weights) for a in order}
            selected=min(order,key=lambda a:scores[a]);gain=1-scores[selected]/scores[baseline] if scores[baseline]>0 else None
            decisions.append(dict(group=group,mode=mode,fallback=baseline,selected=selected,predicted_gain=gain,
                observed_ratio=items[selected]['median_us']/items[baseline]['median_us'],
                training_zero_feature_activations=activated,feature_supported=not activated))
        folds.append(dict(group=group,train_unique=len(train),test_unique=len(test),
            alias_overlap_removed=sum(r['group']!=group and r['execution_identity'] in ids for r in unique),fit=fit,weights=weights))
    changed=[d for d in decisions if d['feature_supported'] and d['selected']!=d['fallback'] and d['predicted_gain'] is not None and d['predicted_gain']>0]
    harmful=[d for d in changed if d['observed_ratio']>1.01]
    threshold=max([d['predicted_gain'] for d in harmful],default=0.)+1e-9 if changed else None
    accepted=[d for d in changed if d['predicted_gain']>threshold]
    result=dict(method='out-of-program-selection-margin-v1',minimum_predicted_gain=threshold,
        graph_groups=groups,material_loss_ratio=1.01,out_of_program_decisions=len(decisions),
        nontrivial_supported_proposals=len(changed),observed_harmful_proposals=len(harmful),
        accepted_oof_proposals=len(accepted),accepted_oof_material_failures=sum(d['observed_ratio']>1.01 for d in accepted),
        hardware_time_bound=False,statistical_coverage_claim=False,
        interpretation='observed OOF material-error rejection threshold; family-specific empirical decision-risk rule, not future coverage or GPU-time guarantee')
    return result,dict(summary=result,folds=folds,decisions=decisions)


def program_contrasts(rows,names):
    groups=defaultdict(list)
    for row in deduplicate(rows):groups[row['group']].append(row)
    result=[]
    for items in groups.values():
        for first,second in itertools.combinations(items,2):result.append([first['features'][n]-second['features'][n] for n in names])
    return result


def reduce_locality_axes(rows,base_names,axis_names):
    """Keep Q-independent or non-conically-redundant address burden axes.

    Nonnegative weights form a cone, so ordinary rank alone is insufficient:
    an axis with a negative coefficient in an old-column representation can
    enable meaningful positive-cost tradeoffs. A dropped axis requires an
    EXACT nonnegative rational-combination certificate on program contrasts.
    These are empirical-domain identities, not universal feature identities.
    """
    selected=[];ledger=[];names=list(base_names)
    contrasts=program_contrasts(rows,names);rank=exact_linear_algebra(contrasts,names)['rank']
    for axis in axis_names:
        full=program_contrasts(rows,names+[axis]);target=[r[-1] for r in full]
        if not any(target):
            ledger.append(dict(axis=axis,status='dropped-no-within-program-variation',coefficients={},empirical_only=True));continue
        augmented=exact_linear_algebra(full,names+[axis])['rank']
        if augmented>rank:
            selected.append(axis);names.append(axis);rank=augmented
            ledger.append(dict(axis=axis,status='retained-Q-independent',rank=rank));continue
        matrix=np.asarray([r[:-1] for r in full],dtype=float)
        weights,fit=nnls(matrix,np.asarray(target,dtype=float))
        rational=[Fraction(float(v)).limit_denominator(1048576) if v>1e-10 else Fraction(0) for v in weights]
        exact=all(sum(Fraction(v)*w for v,w in zip(row[:-1],rational))==row[-1] for row in full)
        if exact:
            ledger.append(dict(axis=axis,status='dropped-exact-nonnegative-cone-combination',
                coefficients={n:[w.numerator,w.denominator] for n,w in zip(names,rational) if w},
                empirical_only=True,all_program_contrast_equalities_verified=True))
        else:
            selected.append(axis);names.append(axis)
            ledger.append(dict(axis=axis,status='retained-Q-dependent-nonnegative-redundancy-not-proved',
                rank=rank,nnls_squared_residual=fit['squared_loss'],empirical_only=True))
    return selected,dict(method='exact-Q-rank-plus-nonnegative-rational-cone-certificates-v1',
        base_features=list(base_names),candidate_axes=list(axis_names),selected_axes=selected,
        final_contrast_rank=rank,ledger=ledger,universal_redundancy_claim=False,
        interpretation='empirical within-program difference cone; dropped identities must be rechecked on every inference pool')


def registered_order_constraints(rows,names,*,controlled_only=False):
    """Observed-development orders; no pair label is used by inference."""
    unique=deduplicate(rows);rng=np.random.default_rng(20261105);result=[]
    for group in sorted({r['group'] for r in unique}):
        for first,second in itertools.combinations([r for r in unique if r['group']==group],2):
            if controlled_only and (not first.get('controlled_group') or first['controlled_group']!=second.get('controlled_group')):continue
            ratio=first['median_us']/second['median_us']
            if max(ratio,1/ratio)<1.02:continue
            a=np.asarray(first['batch_medians_us']);b=np.asarray(second['batch_medians_us'])
            if len(a)!=len(b):raise ValueError('paired order batches differ')
            indices=rng.integers(0,len(a),size=(4000,len(a)));ci=np.percentile(np.median((a/b)[indices],axis=1),[2.5,97.5])
            if ci[0]<=1<=ci[1]:continue
            fast,slow=(first,second) if ratio<1 else (second,first)
            delta=[fast['features'][n]-slow['features'][n] for n in names];divisor=math.gcd(*delta)
            if divisor:delta=[v//divisor for v in delta]
            result.append(dict(group=group,fast=fast['arm'],slow=slow['arm'],fast_identity=fast['execution_identity'],slow_identity=slow['execution_identity'],
                observed_ratio=ratio,paired_bootstrap_ci95=ci.tolist(),delta=delta))
    return result


def small_farkas(rows):
    """Exact one/two-order contradictions; inability to find one proves nothing."""
    for i,row in enumerate(rows):
        if all(v>=0 for v in row):return dict(indices=[i],weights=[[1,1]],combined_delta=row)
    for (i,a),(j,b) in itertools.combinations(enumerate(rows),2):
        lower=Fraction(0);upper=None;possible=True
        for x,y in zip(a,b):
            if y>0:lower=max(lower,Fraction(-x,y))
            elif y<0:upper=Fraction(-x,y) if upper is None else min(upper,Fraction(-x,y))
            elif x<0:possible=False;break
        if possible and (upper is None or lower<=upper):
            combined=[Fraction(x)+lower*y for x,y in zip(a,b)]
            if all(v>=0 for v in combined):return dict(indices=[i,j],weights=[[1,1],[lower.numerator,lower.denominator]],combined_delta=[[v.numerator,v.denominator] for v in combined])
    return None


def build_ordering_cone(rows,names,*,controlled_only=False):
    records=registered_order_constraints(rows,names,controlled_only=controlled_only);matrix=[r['delta'] for r in records]
    base=dict(contract=model.ORDER_CONE_CONTRACT,features=list(names),
        criterion='observed-development: aggregate median ratio >=1.02, paired-bootstrap median-ratio CI95 excludes1, seed20261105,4000 resamples',
        units='dimensionless score, calibrated strict ranking inequalities; not time',hardware_time_bound=False,
        assumption='the declared nonnegative additive family satisfies all registered calibration rankings; no hardware or unseen-workload guarantee')
    if not matrix:return dict(base,status='no-registered-order-evidence'),dict(records=records)
    # Exact small contradictions are decisive and avoid an unnecessary large
    # slack-variable NNLS on an already inconsistent calibration family.
    matrix=[list(r) for r in sorted(set(map(tuple,matrix)))]
    contradiction=small_farkas(matrix)
    if contradiction:return dict(base,status='inconsistent',constraints=matrix,farkas_certificate=contradiction),dict(records=records)
    augmented=np.concatenate([np.asarray(matrix,dtype=float),np.eye(len(matrix))],axis=1)
    solution,fit=nnls(augmented,-np.ones(len(matrix)));weights=solution[:len(names)];witness=None
    for limit in (10**6,10**12,10**18):
        candidate=[Fraction(float(v)).limit_denominator(limit) for v in weights]
        margins=[-sum(v*w for v,w in zip(row,candidate)) for row in matrix]
        if min(margins)>0:witness=[w/min(margins) for w in candidate];break
    if witness is None:
        contradiction=small_farkas(matrix)
        return dict(base,status='inconsistent' if contradiction else 'feasibility-not-proved',
            constraints=matrix,farkas_certificate=contradiction),dict(records=records,feasibility_fit=fit)
    # Strict feasibility witnesses and all dual combinations are verified using
    # rational arithmetic; numerical optimization only proposes certificates.
    cone=dict(base,status='available',constraints=matrix,
        strict_feasible_witness=[[w.numerator,w.denominator] for w in witness])
    if not model.verify_order_cone(cone,names):raise ValueError('strict ordering-cone witness rejected')
    ids=defaultdict(set)
    for row in rows:ids[row['group']].add(row['execution_identity'])
    oof=[]
    for i,record in enumerate(records):
        held=ids[record['group']]
        train=[list(r) for r in sorted({tuple(r['delta']) for r in records if r['group']!=record['group'] and r['fast_identity'] not in held and r['slow_identity'] not in held})]
        proof=model.exact_order_proof(train,record['delta']) if train else None
        oof.append(dict(index=i,group=record['group'],proved=proof is not None,proof=proof,training_orders=len(train),training_constraints_sha256=model.digest(train),
            alias_overlap_removed=sum(r['group']!=record['group'] and (r['fast_identity'] in held or r['slow_identity'] in held) for r in records)))
    groups={g:dict(registered=sum(r['group']==g for r in records),proved=sum(r['group']==g and r['proved'] for r in oof)) for g in sorted(ids)}
    cone['leave_program_out']=dict(registered_pairs=len(records),proved_pairs=sum(r['proved'] for r in oof),
        contradicted_registered_pairs=0,programs_with_proof=sum(v['proved']>0 for v in groups.values()),
        method='entire program instance and all execution aliases excluded; exact nonnegative dual implication',
        qualification_claim=False,statistical_coverage_claim=False)
    return cone,dict(records=records,strict_feasibility_fit=fit,oof=oof,per_program=groups)


def fit_profile(rows,contract,dependencies,pools,old_decisions,*,family=None):
    base_names=tuple(model.FEATURES) if family is None else tuple(rows[0]['summary']['feature_names'])+(('gf2_global_sectors32',) if 'gf2_global_sectors32' in rows[0]['features'] else ())
    locality_names=tuple(rows[0].get('locality_selected',[]));names=base_names+locality_names;weights,fit=fit_weights(rows,names)
    fit['objective']='equal program-mass pair differences normalized by max(abs paired delta, 5% program median), plus .05 relative absolute regularizer'
    graph_risk,ensemble=graph_validation(rows,names);risk,margin_audit=selection_margin_calibration(rows,pools,old_decisions,names)
    vectors=[[r['features'][n] for n in names] for r in deduplicate(rows)]
    if tuple(names[:len(model.FEATURES)])==tuple(model.FEATURES):vectors += [[pair['delta'].get(n,0) for n in names] for pair in _NATIVE_MICRO_PAIRS]
    identifiable=exact_linear_algebra(vectors,list(names))
    ranges={n:dict(min=min(w[n] for w in ensemble+[weights]),max=max(w[n] for w in ensemble+[weights]),
        physically_identified=False,alias_involved=n in identifiable['alias_involved_features']) for n in names}
    if family is None:units=dict(model.UNITS);which_model=model.MODEL
    else:
        import execution_cost_summary as summary
        units={n:summary.UNITS[n] if n!='gf2_global_sectors32' else model.UNITS[n] for n in base_names};which_model=model.SUMMARY_MODEL
    if locality_names:
        import access_reuse as locality
        units.update({n:locality.UNITS[n] for n in locality_names})
        which_model=model.LOCALITY_MODEL if family is None else model.SUMMARY_LOCALITY_MODEL
    source_dependencies=[dict(path=str(Path(__file__)),sha256=sha(__file__),role='source')]
    source_dependencies += [dict(path=str(Path(__file__).parent/n),sha256=v,role='source') for n,v in model.implementation().items()]
    domains=[r['evidence']['payload']['execution_domain'] for r in rows]
    if any(d!=domains[0] for d in domains) or not model.supported_execution_domain(domains[0]):raise ValueError('mixed or unsupported calibration execution domains')
    profile=dict(execution_domain=domains[0],schema_version=1,contract=model.CONTRACT,model=which_model,features=list(names),feature_units=units,
        implementation_sha256=model.implementation(),analysis_contract=contract,coefficients_us_per_unit=weights,
        training_maxima={n:max(max(r['features'][n] for r in rows),_NATIVE_MICRO_MAXIMA.get(n,0) if family is None else 0) for n in names},
        identifiability=dict(**identifiable,coefficient_graph_ensemble_ranges=ranges),risk_calibration=risk,
        family_out_risk_calibration={k:v for k,v in graph_risk.items() if k!='folds'},
        uses_observed_development_timings=True,training_role='R3 all already-observed development/training data',
        hardware_time_bound=False,weight_field='nonnegative real numbers (not GF2)',intercept_us=0,
        interpretation=model.BASIS,local_semantics='local_transfer_bytes = sum static native LDL/STL operand bytes; CUDA driver LOCAL_SIZE_BYTES is allocation footprint and is not a traffic feature',
        dependencies=dependencies+source_dependencies+(_NATIVE_MICRO_DEPENDENCIES if family is None else []))
    cone,cone_audit=build_ordering_cone(rows,names)
    profile['ordering_cone']=cone
    fit['ordering_cone_audit']=cone_audit
    if family is not None:profile['cost_family']=family;profile['uses_gf2']='gf2_global_sectors32' in names
    if locality_names:
        profile.update(locality_features=list(locality_names),locality_analysis_contract=rows[0]['locality_contract'],locality_reduction=rows[0]['locality_reduction'])
        profile['dependencies'] += [dict(path=str(Path(locality.__file__)),sha256=sha(locality.__file__),role='source')]
        profile['dependencies'] += [dict(path=str(Path(locality.__file__).parent/n),sha256=v,role='source') for n,v in rows[0]['locality']['provenance']['source_dependencies'].items()]
    return profile,fit,graph_risk,margin_audit


def load_summaries(rows,directory):
    directory=Path(directory);result=defaultdict(list);dependencies=[]
    for row in rows:
        path=directory/(Path(row['path']).name+'.json')
        if not path.exists():raise ValueError('missing complete summary '+str(path))
        summary=read(path)
        if summary['status']!='complete':continue
        e=row['evidence']['payload']['preceding']['payload'];g=e['gf2']
        family=summary.get('cost_family')
        if family not in ('sm90-acyclic-call-expanded-occurrences-v1','sm90-immediate-wait-wgmma-v1'):raise ValueError('unsupported training summary family')
        use_gf2=family!='sm90-immediate-wait-wgmma-v1'
        if use_gf2 and g['status']!='available':continue
        contract=dict(contract=summary['contract'],analyzer_sha256=summary['provenance']['analyzer_sha256'],nvdisasm=summary['provenance']['nvdisasm'])
        prototype=dict(model=model.SUMMARY_MODEL,cost_family=family,uses_gf2=use_gf2,analysis_contract=contract)
        extracted=model.summary_features(row['native'],summary,row['evidence'],prototype)
        if extracted['status']!='available':raise ValueError('training summary evidence rejected: '+str(extracted['reason']))
        item=dict(row,features=extracted['features'],
            feature_status='available',feature_reason=None,summary=summary,summary_contract=contract)
        result[summary['cost_family']].append(item);dependencies.append(dict(path=str(path),sha256=sha(path)))
        for name,digest in summary['provenance']['dependencies'].items():dependencies.append(dict(path=str(Path(row['path'])/name),sha256=digest))
    return result,dependencies


def load_localities(rows,directory):
    """Validate training locality with the SAME timing-free inference validator."""
    import access_reuse as locality
    result={};dependencies=[];directory=Path(directory)
    contract=dict(contract=locality.CONTRACT,cost_family=locality.FAMILY,analyzer_sha256=sha(locality.__file__))
    prototype=dict(locality_features=list(locality.FOOTPRINT_FEATURES),locality_analysis_contract=contract)
    for row in rows:
        path=directory/(Path(row['path']).name+'.json')
        if not path.exists():raise ValueError('missing locality evidence '+str(path))
        document=read(path);result[row['group'],row['arm']]=document
        if document.get('status')!='complete':continue
        checked=model.extend_locality(dict(status='available',features={}),document,row['evidence'],prototype)
        if checked['status']!='available':raise ValueError('training locality evidence rejected: '+str(checked['reason']))
        dependencies.append(dict(path=str(path),sha256=sha(path)))
        for name,digest in document['provenance']['dependencies'].items():dependencies.append(dict(path=str(Path(row['path'])/name),sha256=digest))
    return result,dependencies,contract


def augment_locality_family(rows,family,documents,contract):
    """Attach only complete same-family data; Attention remains unsupported."""
    if not documents:return rows,None
    import access_reuse as locality
    if any(documents.get((r['group'],r['arm']),{}).get('status')!='complete' for r in rows):return rows,None
    base_names=tuple(model.FEATURES) if family=='native15' else tuple(rows[0]['summary']['feature_names'])+(('gf2_global_sectors32',) if 'gf2_global_sectors32' in rows[0]['features'] else ())
    combined=[dict(r,features=dict(r['features'],**{n:documents[r['group'],r['arm']]['features'][n] for n in locality.FOOTPRINT_FEATURES})) for r in rows]
    # A declared structural priority, identical for every program: first shared
    # CTA page working sets, then finer blocks, then warp-local footprints.
    axis_order=sorted(locality.FOOTPRINT_FEATURES,key=lambda n:(0 if '_cta_' in n else 1,
        -int(n.split('_b')[1].split('_')[0]),int(n.split('_w')[-1].split('_')[0])))
    selected,reduction=reduce_locality_axes(combined,base_names,axis_order)
    enriched=[dict(r,locality_selected=selected,locality_contract=contract,
        locality_reduction=reduction,locality=documents[r['group'],r['arm']]) for r in combined]
    return enriched,reduction


def build_address_margin(rows,directory,localities,dependencies,out):
    import access_reuse as locality
    import address_intervention as intervention
    names,units=model.feature_contract(dict(model=model.ADDRESS_MARGIN_MODEL))
    eligible=[r for r in rows if r['evidence']['payload']['execution_domain']['dtype']=='float32']
    domain=eligible[0]['evidence']['payload']['execution_domain']
    if any(r['evidence']['payload']['execution_domain']!=domain for r in eligible):raise ValueError('controlled margin mixed execution domain')
    profile=dict(schema_version=1,contract=model.CONTRACT,model=model.ADDRESS_MARGIN_MODEL,
        features=list(names),feature_units=units,implementation_sha256=model.implementation(),execution_domain=domain,
        locality_analysis_contract=dict(contract=locality.CONTRACT,cost_family=locality.FAMILY,analyzer_sha256=sha(locality.__file__)),
        intervention_analysis_contract=dict(contract=intervention.CONTRACT,analyzer_sha256=sha(intervention.__file__)),
        uses_observed_development_timings=True,hardware_time_bound=False,
        interpretation='controlled source address marginal ordering; common source/native static work cancels; dynamic masks/slowpaths and native cache misses are not proved equal',
        dependencies=list(dependencies))
    documents={};controlled=[]
    for row in eligible:
        path=Path(directory)/(Path(row['path']).name+'.json');document=read(path);documents[row['group'],row['arm']]=document
        checked=model.intervention_features(row['native'],row['evidence'],localities[row['group'],row['arm']],document,profile)
        if checked['status']!='available':raise ValueError('training controlled intervention rejected: '+str(checked.get('reason')))
        controlled.append(dict(row,features=checked['features'],controlled_group=checked['controlled_group_sha256']))
        profile['dependencies'].append(dict(path=str(path),sha256=sha(path)))
        for name,value in document['provenance']['dependencies'].items():profile['dependencies'].append(dict(path=str(Path(row['path'])/name),sha256=value))
    for name in ('joint_cost_model.py','joint_cost_calibration.py','address_intervention.py','access_reuse.py','execution_cost_summary.py'):
        path=Path(__file__).parent/name;profile['dependencies'].append(dict(path=str(path),sha256=sha(path),role='source'))
    for name,value in locality.SOURCE_HASHES.items():profile['dependencies'].append(dict(path=str(Path(locality.__file__).parent/name),sha256=value,role='source'))
    cone,audit=build_ordering_cone(controlled,names,controlled_only=True);profile['ordering_cone']=cone
    profile['scope']=dict(pair_eligibility='same validated controlled_group inside same program and paired measurement population',
        common_offset_invariant=True,feature_weight_field='nonnegative real numbers, not GF2',
        timing_population='observed R3 development mixture; no phase-specific/native cache/TLB or unseen-workload guarantee')
    out=Path(out);out.mkdir();write(out/'profile.json',dict(payload=profile,payload_sha256=model.digest(profile)))
    model.load_profile(out/'profile.json',verify_training_dependencies=True);write(out/'ordering-cone-audit.json',audit)
    write(out/'training-ledger.json',[{k:r[k] for k in ('group','arm','execution_identity','features','controlled_group','median_us','batch_medians_us')} for r in controlled])
    return profile,documents,dict(path=str(out/'profile.json'),sha256=sha(out/'profile.json'),
        occurrences=len(controlled),columns=len(names),risk=cone.get('leave_program_out'),status=cone['status'])


def evaluate_bundle(rows,pools,quality,profiles,old_decisions,summary_rows,locality_documents=None,intervention_documents=None):
    by={(r['group'],r['arm']):r for r in rows};summaries={(r['group'],r['arm']):r['summary'] for family in summary_rows.values() for r in family}
    result={};counts=defaultdict(int)
    for group,pool in pools.items():
        medians={i:float(np.median(v)) for i,v in quality[group]['image_batch_medians_us'].items()};union=min(medians.values());entry={}
        for mode,info in pool['modes'].items():
            arms=info['arms'];items={a:by[group,a] for a in arms};baseline=old_decisions['groups'][group]['choices'][mode+'-affine-frame']['selected_source']
            order=[baseline]+[a for a in info['static_order'] if a!=baseline];natives={a:r['native'] for a,r in items.items()};evidences={a:r['evidence'] for a,r in items.items()}
            ss={a:summaries.get((group,a)) for a in arms}
            ls={a:locality_documents.get((group,a)) for a in arms} if locality_documents else None
            ins={a:intervention_documents.get((group,a)) for a in arms} if intervention_documents else None
            receipt=model.dispatch(natives,evidences,profiles,baseline,order,summaries=ss,localities=ls,interventions=ins)
            model.verify_dispatch(receipt,natives,evidences,profiles,baseline,order,summaries=ss,localities=ls,interventions=ins)
            which=receipt['payload']['cost_family']
            selected=receipt['payload']['selected_source'];latencies={a:medians[items[a]['execution_identity']] for a in arms};best=min(latencies.values());value=latencies[selected]
            entry[mode]=dict(cost_family=which,selected=selected,old_affine_choice=baseline,
                pool_coverage_ratio=best/union,within_pool_selection_ratio=value/best,union_regret_ratio=value/union,
                relative_to_static=value/latencies[info['fallback']],relative_to_old_policy=value/latencies[baseline],
                relative_to_legacy=value/quality[group]['selections']['legacy-static']['median_us'],latencies_us=latencies,receipt=receipt)
            counts[receipt['payload']['status']]+=1
        result[group]=entry
    aggregate={}
    for mode in ('joint','local'):
        aggregate[mode]={}
        for key in ('pool_coverage_ratio','within_pool_selection_ratio','union_regret_ratio','relative_to_static','relative_to_old_policy','relative_to_legacy'):
            values=[r[mode][key] for r in result.values()];aggregate[mode][key]=dict(geometric_mean=math.exp(sum(map(math.log,values))/len(values)),maximum=max(values))
    return dict(dataset_role='R3 observed development only, not fresh qualification',groups=result,summary=aggregate,status_counts=dict(counts),qualification_pass_claim=False)


def _run_bundle(run,out,summary_directory,native_micro_labels=None,locality_directory=None,intervention_directory=None):
    run=Path(run);out=Path(out)
    if out.exists() or run.resolve() in out.resolve().parents:raise ValueError('fresh non-archive output required')
    out.mkdir(parents=True);rows,contract,dependencies,pools,quality=collect(run);old=read(run/'partial-decisions.json')
    micro=load_native_micro_pairs(native_micro_labels,contract) if native_micro_labels else None
    if micro is not None:write(out/'native-micro-constraints.json',dict(verification=micro,pairs=_NATIVE_MICRO_PAIRS))
    summary_rows,summary_dependencies=load_summaries(rows,summary_directory);profiles={};outputs={}
    localities,locality_dependencies,locality_contract=load_localities(rows,locality_directory) if locality_directory else ({},[],None)
    families=dict(summary_rows);families['native15']=[r for r in rows if r['features'] is not None]
    for family,items in families.items():
        items,reduction=augment_locality_family(items,family,localities,locality_contract)
        key=family.replace('/','_');folder=out/key;folder.mkdir();native=family=='native15';analysis=contract if native else items[0]['summary_contract']
        if not native and any(r['summary_contract']!=analysis for r in items):raise ValueError('summary analysis contracts differ')
        profile,fit,graph,margin=fit_profile(items,analysis,dependencies+summary_dependencies+locality_dependencies,pools,old,family=None if native else family)
        write(folder/'profile.json',dict(payload=profile,payload_sha256=model.digest(profile)));model.load_profile(folder/'profile.json',verify_training_dependencies=True)
        if reduction:write(folder/'locality-axis-reduction.json',reduction)
        write(folder/'ordering-cone-audit.json',fit.pop('ordering_cone_audit'))
        write(folder/'fit.json',fit);write(folder/'graph-family-stress.json',graph);write(folder/'selection-margin-calibration.json',margin)
        write(folder/'training-ledger.json',[{k:r[k] for k in ('group','graph','arm','execution_identity','features','median_us','batch_medians_us')} for r in items])
        if native:write(folder/'algebraic-diagnostics.json',algebraic_diagnostics(items))
        profiles[family]=profile;outputs[family]=dict(path=str(folder/'profile.json'),sha256=sha(folder/'profile.json'),
            occurrences=len(items),unique_programs=len(deduplicate(items)),rank=profile['identifiability']['rank'],columns=len(profile['features']),risk=profile['risk_calibration'])
    interventions={}
    if intervention_directory:
        if not locality_directory:raise ValueError('controlled margin requires locality proofs')
        address_profile,interventions,outputs['address-margin']=build_address_margin(rows,intervention_directory,localities,dependencies+summary_dependencies+locality_dependencies,out/'address-margin')
        profiles['address-margin']=address_profile
    regression=evaluate_bundle(rows,pools,quality,profiles,old,summary_rows,locality_documents=localities,intervention_documents=interventions);write(out/'development-regression.json',regression)
    bundle=dict(schema_version=1,contract=model.CONTRACT,profiles=outputs,dispatch='native15 if entire pool supported; else identical complete-summary family; otherwise old policy fallback',
        qualification_pass_claim=False,development_regression_sha256=sha(out/'development-regression.json'))
    write(out/'profile-bundle.json',bundle);write(out/'completion.json',dict(status='complete',bundle_sha256=sha(out/'profile-bundle.json'),no_gpu_or_compile=True,qualification_pass_claim=False))
    return dict(profiles=outputs,summary=regression['summary'],status_counts=regression['status_counts'])


def run_bundle(run,out,summary_directory,native_micro_labels=None,locality_directory=None,intervention_directory=None):
    """Keep optional auxiliary training state scoped to one offline run."""
    global _NATIVE_MICRO_PAIRS,_NATIVE_MICRO_MAXIMA,_NATIVE_MICRO_DEPENDENCIES
    saved=(_NATIVE_MICRO_PAIRS,_NATIVE_MICRO_MAXIMA,_NATIVE_MICRO_DEPENDENCIES)
    _NATIVE_MICRO_PAIRS=[];_NATIVE_MICRO_MAXIMA={};_NATIVE_MICRO_DEPENDENCIES=[]
    try:return _run_bundle(run,out,summary_directory,native_micro_labels,locality_directory,intervention_directory)
    finally:_NATIVE_MICRO_PAIRS,_NATIVE_MICRO_MAXIMA,_NATIVE_MICRO_DEPENDENCIES=saved


def _run_calibration(run,out):
    run=Path(run);out=Path(out)
    if out.exists():raise ValueError('fresh output directory required')
    if run.resolve()==out.resolve() or run.resolve() in out.resolve().parents:raise ValueError('never write under observed archive')
    out.mkdir(parents=True)
    rows,contract,dependencies,pools,quality=collect(run);supported=[r for r in rows if r['features'] is not None]
    weights,fit=fit_weights(supported);fit['objective']='equal program-mass pair differences normalized by max(abs paired delta, 5% program median), plus .05 relative absolute regularizer';risk,ensemble=graph_validation(supported);algebra=algebraic_diagnostics(supported)
    # The ensemble range is descriptive; nullspace and noise prevent calling
    # fitted active coefficients exact even when their training column is nonzero.
    ranges={n:dict(min=min(w[n] for w in ensemble+[weights]),max=max(w[n] for w in ensemble+[weights]),
        physically_identified=False,alias_involved=n in algebra['full_features']['alias_involved_features']) for n in model.FEATURES}
    source_dependencies=[dict(path=str(Path(__file__)),sha256=sha(__file__),role='source')]
    source_dependencies += [dict(path=str(Path(__file__).parent/n),sha256=v,role='source') for n,v in model.implementation().items()]
    domains=[r['evidence']['payload']['execution_domain'] for r in supported]
    if any(d!=domains[0] for d in domains) or not model.supported_execution_domain(domains[0]):raise ValueError('mixed or unsupported calibration execution domains')
    profile=dict(execution_domain=domains[0],schema_version=1,contract=model.CONTRACT,model=model.MODEL,features=list(model.FEATURES),feature_units=model.UNITS,
        implementation_sha256=model.implementation(),analysis_contract=contract,coefficients_us_per_unit=weights,
        training_maxima={n:max(r['features'][n] for r in supported) for n in model.FEATURES},
        identifiability=dict(**algebra['full_features'],coefficient_graph_ensemble_ranges=ranges),
        risk_calibration={k:v for k,v in risk.items() if k!='folds'},
        uses_observed_development_timings=True,training_role='R3 all already-observed development/training data',
        hardware_time_bound=False,weight_field='nonnegative real numbers (not GF2)',intercept_us=0,
        interpretation=model.BASIS,local_semantics='local_transfer_bytes = sum static native LDL/STL operand bytes; CUDA driver LOCAL_SIZE_BYTES is allocation footprint and is not a traffic feature',
        dependencies=dependencies+source_dependencies+_NATIVE_MICRO_DEPENDENCIES)
    write(out/'profile.json',dict(payload=profile,payload_sha256=model.digest(profile)))
    model.load_profile(out/'profile.json',verify_training_dependencies=True)
    development=evaluate(rows,pools,quality,profile,read(run/'partial-decisions.json'))
    ledger=[{k:r[k] for k in ('group','graph','spec','arm','execution_identity','cubin_sha256','path','features','feature_status','feature_reason','static20','median_us','batch_medians_us')} for r in rows]
    write(out/'training-ledger.json',ledger);write(out/'fit.json',fit);write(out/'graph-validation.json',risk);write(out/'algebraic-diagnostics.json',algebra);write(out/'development-regression.json',development)
    completion=dict(status='complete',dataset_role=profile['training_role'],input_run=str(run),
        candidate_occurrences=len(rows),supported_occurrences=len(supported),unique_training_programs=len(deduplicate(supported)),
        profile_sha256=sha(out/'profile.json'),fit_sha256=sha(out/'fit.json'),
        development_regression_sha256=sha(out/'development-regression.json'),graph_validation_sha256=sha(out/'graph-validation.json'),
        algebraic_diagnostics_sha256=sha(out/'algebraic-diagnostics.json'),training_ledger_sha256=sha(out/'training-ledger.json'),
        no_gpu_or_compile=True,qualification_pass_claim=False)
    write(out/'completion.json',completion)
    return dict(completion=completion,summary=development['summary'],status_counts=development['status_counts'],
        rank=algebra['full_features']['rank'],gf2_rank_increment=algebra['gf2_rank_increment'],risk_envelope=risk['log_ratio_error_envelope'])


def run_calibration(run,out):
    """The no-summary CLI path has no implicit auxiliary training state."""
    global _NATIVE_MICRO_PAIRS,_NATIVE_MICRO_MAXIMA,_NATIVE_MICRO_DEPENDENCIES
    saved=(_NATIVE_MICRO_PAIRS,_NATIVE_MICRO_MAXIMA,_NATIVE_MICRO_DEPENDENCIES)
    _NATIVE_MICRO_PAIRS=[];_NATIVE_MICRO_MAXIMA={};_NATIVE_MICRO_DEPENDENCIES=[]
    try:return _run_calibration(run,out)
    finally:_NATIVE_MICRO_PAIRS,_NATIVE_MICRO_MAXIMA,_NATIVE_MICRO_DEPENDENCIES=saved


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('out',type=Path);parser.add_argument('--summary-directory',type=Path);parser.add_argument('--native-micro-labels',type=Path);parser.add_argument('--locality-directory',type=Path);parser.add_argument('--intervention-directory',type=Path);args=parser.parse_args()
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{24,25,26,27})
    print(json.dumps(run_bundle(args.run,args.out,args.summary_directory,args.native_micro_labels,args.locality_directory,args.intervention_directory) if args.summary_directory else run_calibration(args.run,args.out),sort_keys=True))


if __name__=='__main__':main()
