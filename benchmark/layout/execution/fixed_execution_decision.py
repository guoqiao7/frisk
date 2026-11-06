"""Calibrate fixed native choices from matched interventions, with bounded transfer."""

import hashlib
import json
import math
import random
import statistics

from execution_plan_cost import compare_features, copy_publication_only, static_key
from execution_plan_gate import _facts

CONTRACT = 'frisk-fixed-intervention-calibration-v1'
MIN_SUPPORT = 2
MIN_GAIN = 0.005
POLICY_FIELDS = ('access', 'mma_group', 'reduction', 'shared', 'store', 'copy_batch')


def _policy_value(policy, field):
    return policy.get(field, False) if field == 'copy_batch' else policy[field]


def _finite_number(x):
    return type(x) in (int,float) and math.isfinite(x)


def _memory(f, which):
    return f['global_'+which+'_by_width_bytes']


def describe(candidate, incumbent):
    """Describe a native contrast using the emitted mechanism and its physical context."""
    if candidate['case'] != incumbent['case']:
        raise ValueError('fixed comparison requires the same workload')
    a,b=(row['preview_features'] for row in (candidate,incumbent))
    if a['scope']!=b['scope']:
        return None
    x,y=a['static'],b['static']
    old,new=_facts(incumbent),_facts(candidate)
    if old is None or new is None or old['device']!=new['device'] or old['limits']!=new['limits']:
        return None
    p,q=(row['policy'] for row in (candidate,incumbent))
    different=[k for k in POLICY_FIELDS if _policy_value(p,k)!=_policy_value(q,k)]
    same_layout=candidate['packet_sha256']==incumbent['packet_sha256']
    if not same_layout and not different:
        mechanism='layout'
    elif not same_layout or len(different)!=1:
        mechanism='compound'
    else:
        mechanism={'access':'address-only' if all(_memory(x,k)==_memory(y,k) for k in ('load','store'))
                              else 'access-width',
                   'mma_group':'wgmma-group','reduction':'ordered-reduction',
                   'shared':'shared-lifetime','store':'staged-epilogue','copy_batch':'copy-publication'}[different[0]]
    # Edge and compound contrasts are retained for diagnosis. A same-operator
    # aggregate does not isolate a transportable intervention.
    counters=('global_load_sites','global_store_sites','shared_load_sites','shared_store_sites',
              'conversion_shuffles','reduction_shuffles','cta_barriers','wgmma_waits',
              'global_sector_issues32_upper')
    if mechanism=='copy-publication':
        counters+=('async_proxy_fences',)
    if any(not _finite_number(f.get(k)) for f in (x,y) for k in counters):
        return None
    # All dimensions are dimensionless deltas or explicit baseline work
    # strata. The matcher does not extrapolate a fitted slope beyond them.
    coordinates=[y[k] for k in ('global_load_sites','global_store_sites','reduction_shuffles','wgmma_waits')]
    coordinates.extend((x[k]-y[k])/max(1,y[k]) for k in counters)
    coordinates.extend((new['registers']-old['registers'],new['waves']-old['waves']))
    spec=candidate['spec']
    family=spec['operator']
    # Store staging implemented before factorized Shared stores has a different
    # measured effect. Compare actual transfer widths, not the policy spelling.
    shared_widths=tuple(sorted(int(w) for w in x['shared_store_by_width_bytes'])) if mechanism=='staged-epilogue' else ()
    mma_forms=tuple(sorted(k.partition(':')[0] for k in a['scope']['wgmma_atoms']))
    key=[mechanism,*[list((k,_policy_value(q,k),_policy_value(p,k))) for k in different],
         list(a['scope']['block']),list(a['scope']['global_alignments']),list(mma_forms),list(shared_widths),
         old['waves']]
    return dict(mechanism=mechanism,key=key,coordinates=coordinates,family=family)


def _median_interval(values, strata, seed):
    rng=random.Random(seed)
    groups=[[v for v,o in zip(values,strata) if o==offset] for offset in sorted(set(strata))]
    if len(groups)<2 or any(len(g)<2 for g in groups):
        raise ValueError('independent alignment blocks required')
    sampled=sorted(statistics.median([rng.choice(g) for g in groups for _ in g])
                   for _ in range(4000))
    return [sampled[99],sampled[3900]]


def fit_profile(trials):
    """Fit shape-wise paired intervals, with explicit leave-one-shape/family auditing."""
    evidence=[]
    identifiers=set()
    for index,t in enumerate(trials):
        vals=t['ratios']
        offsets=t['offsets']
        if (t['id'] in identifiers or len(vals)!=len(offsets) or len(vals)<12
                or any(not _finite_number(v) or v<=0 for v in vals)):
            raise ValueError('invalid development pair or duplicate identity')
        identifiers.add(t['id'])
        if t['contrast'] is not None:
            c=t['contrast']['coordinates']
            if not c or any(not _finite_number(v) for v in c):
                raise ValueError('invalid development footprint')
        evidence.append(dict(id=t['id'],shape=t['shape'],family=t['family'],contrast=t['contrast'],
                             mechanism=t['mechanism'],pairs=len(vals),median=statistics.median(vals),
                             interval95=_median_interval(vals,offsets,43193+index),provenance=t['provenance']))
    profile=dict(contract=CONTRACT,evidence=evidence,minimum_independent_shapes=MIN_SUPPORT,
                 minimum_relative_gain=MIN_GAIN)
    for kind in ('shape','family'):
        folds=[]
        for entry in evidence:
            if entry['contrast'] is None:
                continue
            train=[e for e in evidence if e[kind]!=entry[kind]]
            p=_bound(train,entry['contrast'],entry['shape'],allow_other_families=(kind=='family'))
            folds.append(dict(id=entry['id'],prediction=p,observed=entry['interval95'],
                              covers=p is not None and p['lower']<=entry['median']<=p['upper']))
        profile['leave_one_'+kind+'_out']=folds
    return profile


def _bound(evidence, contrast, shape, *, allow_other_families=False):
    if contrast is None or contrast['mechanism'] in ('layout','compound'):
        return None
    records=[e for e in evidence if e['shape']!=shape and e['contrast'] is not None
             and e['contrast']['key']==contrast['key']
             and (allow_other_families or e['family']==contrast['family'])]
    if len({e['shape'] for e in records})<MIN_SUPPORT:
        return None
    if allow_other_families and len({e['family'] for e in records})<MIN_SUPPORT:
        return None
    axes=list(zip(*(e['contrast']['coordinates'] for e in records)))
    if len(axes)!=len(contrast['coordinates']) or any(not min(v)<=x<=max(v)
                                                     for x,v in zip(contrast['coordinates'],axes)):
        return None
    return dict(lower=min(e['interval95'][0] for e in records),upper=max(e['interval95'][1] for e in records),
                support=sorted(e['id'] for e in records),shapes=len({e['shape'] for e in records}))


def prediction(profile, contrast, shape):
    if profile['contract']!=CONTRACT:
        raise ValueError('unrecognized calibration contract')
    estimate=_bound(profile['evidence'],contrast,shape)
    if estimate is None:
        return None
    # A development fold must have exercised this mechanism, and its
    # predicted interval must have covered the omitted observation.
    folds=[f for f in profile['leave_one_shape_out'] if f['id'] in estimate['support'] and f['prediction'] is not None]
    if not folds or any(not f['covers'] for f in folds):
        return None
    return estimate


def _publication_control(row, candidates, facts):
    for control, resources in candidates:
        if (resources is None or any(row[k]!=control[k] for k in ('case','grid','block','packet_sha256'))
                or resources['device']!=facts['device'] or resources['limits']!=facts['limits']
                or _policy_value(control['policy'],'copy_batch') is not False
                or any(_policy_value(row['policy'],k)!=_policy_value(control['policy'],k)
                       for k in POLICY_FIELDS if k!='copy_batch')):
            continue
        if copy_publication_only(row['preview_features'],control['preview_features']):
            return control,resources
    return None


def fixed_choice(rows, fallback, profile, *, local=False):
    """Select before timing. Unknown trade-offs retain the admitted fallback."""
    base=_facts(fallback)
    if base is None or fallback['status']!='admitted':
        raise ValueError('fixed choice requires an admitted fallback')
    candidates=[(row,_facts(row)) for row in rows]
    eligible=[]
    audit={}
    for row,facts in candidates:
        arm=row['arm']
        if facts is None:
            audit[arm]=dict(reason='not-admitted')
            continue
        same_launch=all(row[k]==fallback[k] for k in ('case','grid','block'))
        if not same_launch or facts['limits']!=base['limits'] or facts['device']!=base['device']:
            raise ValueError('candidate does not match fallback workload/device')
        if row['program_sha256']==fallback['program_sha256']:
            audit[arm]=dict(reason='incumbent-program')
            continue
        if facts['waves']>base['waves'] or facts['local']>base['local']:
            audit[arm]=dict(reason='resource-regression')
            continue
        control_arm=None
        if _policy_value(row['policy'],'copy_batch') is True:
            match=_publication_control(row,(*candidates,(fallback,base)),facts)
            if match is None:
                audit[arm]=dict(reason='no-matched-publication-control')
                continue
            control,resources=match
            control_arm=control['arm']
            if (facts['blocks']<max(base['blocks'],resources['blocks'])
                    or facts['waves']>resources['waves'] or facts['local']>resources['local']):
                audit[arm]=dict(reason='publication-resource-regression',control_arm=control_arm)
                continue
        contrast=describe(row,fallback)
        estimate=prediction(profile,contrast,row['case'])
        compare=compare_features(row['preview_features'],fallback['preview_features'])
        if estimate is not None and estimate['upper']<=1/(1+MIN_GAIN):
            reason='calibrated-win'
            eligible.append(((0,estimate['upper'],arm),row))
        elif estimate is not None and estimate['lower']>1:
            reason='calibrated-regression'
        elif compare['structural_relation']=='dominates':
            reason='copy-publication-preference' if control_arm is not None else 'structural-preference'
            eligible.append(((1,static_key(row['preview_features'],local=local),arm),row))
        else:
            reason='unresolved-tradeoff'
        audit[arm]=dict(reason=reason,contrast=contrast,prediction=estimate,structural=compare['structural_relation'])
        if control_arm is not None:
            audit[arm]['control_arm']=control_arm
    best=min(eligible,key=lambda x:x[0])[1] if eligible else fallback
    reason=audit[best['arm']]['reason'] if eligible else 'pinned-fallback'
    return best,dict(reason=reason,fallback_arm=fallback['arm'],candidates=audit)


def profile_digest(profile):
    return hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
