"""Separate archive coverage from ranking using the same paired observations.

No fitting, selection, filtering of candidates, or extra measurements occurs.
The batch-oracle factorization is descriptive and distinct from the registered
max-of-paired-median regret statistic used for acceptance.
"""
import numpy as np
from selection_quality_analysis import analyze_group


def _summary(values,draws):
    a=np.asarray(values,dtype=float)
    return dict(paired_median=float(np.median(a)),
        ci95=np.quantile(np.median(a[draws],axis=1),[.025,.975]).tolist(),
        batch_values=a.tolist())


def decompose(group,pool,aliases,choices):
    method=group['method'];n=group['paired_batches']
    images={k:np.asarray(v,dtype=float) for k,v in group['image_batch_medians_us'].items()}
    if any(v.shape!=(n,) or not np.all(np.isfinite(v)) or np.any(v<=0) for v in images.values()):
        raise ValueError('complete finite positive paired image times required')
    draws=np.random.default_rng(method['bootstrap_seed']).integers(0,n,(method['bootstrap_samples'],n))
    union=np.min(np.stack(list(images.values())),axis=0);result={}
    for mode in ('joint','local'):
        members=pool['modes'][mode]['arms']
        identities={aliases[a]['execution_identity'] for a in members}
        if not identities or not identities<=set(images): raise ValueError('pool image missing')
        own=np.min(np.stack([images[a] for a in sorted(identities)]),axis=0)
        coverage=own/union;selected={}
        for name,c in choices.items():
            if not name.startswith(mode+'-'):continue
            if c['selected_source'] not in members: raise ValueError('choice escaped own pool')
            t=images[aliases[c['selected_source']]['execution_identity']]
            ranking=t/own;total=t/union
            if not np.allclose(ranking*coverage,total,rtol=1e-13,atol=0):raise AssertionError('batch factorization failed')
            selected[name]=dict(within_pool_batch_oracle_loss=_summary(ranking,draws),
                finite_union_batch_oracle_loss=_summary(total,draws),
                max_factorization_absolute_error=float(np.max(np.abs(ranking*coverage-total))))
        result[mode]=dict(candidate_programs=len(members),unique_images=len(identities),
            pool_coverage_batch_oracle_loss=_summary(coverage,draws),selections=selected)
    return dict(modes=result,scope='registered finite union only',
        definition='per batch: selected/union_min = (selected/pool_min)*(pool_min/union_min)',
        warning='batch minima are descriptive optimistic oracles; medians/CIs cannot be multiplied; acceptance uses its separately registered regret estimator')


def ranking(raw,rows,pool,aliases,choices,*,seed,bootstrap_samples=10000):
    result={};byarm={r['arm']:r for r in rows}
    for mode in ('joint','local'):
        members=pool['modes'][mode]['arms'];images={};all_exact=True;native_states={}
        for arm in members:
            alias=aliases[arm];identity=alias['execution_identity'];r=byarm[arm]
            item=images.setdefault(identity,dict(arm=alias['arm'],cubin_sha256=r['cubin_sha256'],
                execution_identity=identity,model_scores={'ordered-network-static':r['cost']['cost']}))
            item['model_scores']['ordered-network-static']=min(item['model_scores']['ordered-network-static'],r['cost']['cost'])
            state=choices[mode+'-interval-only']['intervals'][arm];native_states[arm]=state['status']
            if state['status']!='exact':all_exact=False
            else:item['model_scores']['native14-point']=min(item['model_scores'].get('native14-point',[state['lower_ticks']]),[state['lower_ticks']])
        if not all_exact:
            for item in images.values():item['model_scores'].pop('native14-point',None)
        names={r['arm'] for r in images.values()}
        selections={name:aliases[c['selected_source']]['arm'] for name,c in choices.items() if name.startswith(mode+'-')}
        result[mode]=analyze_group([r for r in raw if r['arm'] in names],list(images.values()),selections,
            expected_batches=10,bootstrap_seed=seed,bootstrap_samples=bootstrap_samples)
        result[mode]['native_point_ranking_contract']=dict(available=all_exact,states=native_states,
            reason=None if all_exact else 'incomplete exact native point scores; no subset filtering or unknown-as-zero')
    return result
