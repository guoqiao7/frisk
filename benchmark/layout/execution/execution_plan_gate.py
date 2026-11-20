"""Pre-label native fallback and held-out promotion decisions."""

from hashlib import sha256
import json
import math

from execution_plan_cost import compare_features


def _facts(row):
    if row.get('status') != 'admitted':
        return None
    keys=('cubin_sha256','grid','block','entry','dynamic_shared_bytes')
    identity=sha256(json.dumps({k:row[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if row.get('image_identity')!=identity:
        raise ValueError('resource receipt is not tied to the admitted native image')
    resources=row['resources']
    residency=resources['residency']
    limits=resources['device_limits']
    blocks=residency['active_blocks_per_sm']
    if not residency['launchable_by_resources'] or type(blocks) is not int or blocks<=0:
        return None
    waves=math.ceil(math.prod(row['grid'])/(limits['multiprocessor_count']*blocks))
    return dict(device=resources['device_ordinal'],limits=limits,waves=waves,blocks=blocks,
                local=resources['local_bytes'],registers=resources['registers_per_thread'])


def native_guard(proposed, fallback):
    """Conserve the pinned native route when static tradeoffs are unresolved."""
    old=_facts(fallback)
    if old is None:
        raise ValueError('no admitted fallback')
    new=_facts(proposed)
    if new is None:
        return fallback['arm'],dict(reason='proposal-not-admitted',fallback=old)
    if (proposed['case'],proposed['grid'],proposed['block'])!=(fallback['case'],fallback['grid'],fallback['block']):
        raise ValueError('proposal and fallback do not share the launch problem')
    if new['device']!=old['device'] or new['limits']!=old['limits']:
        raise ValueError('proposal and fallback do not share native device resources')
    if proposed['program_sha256']==fallback['program_sha256']:
        return proposed['arm'],dict(reason='same-preview-program',candidate=new,fallback=old)
    comparison=compare_features(proposed['preview_features'],fallback['preview_features'])
    if new['local']>old['local'] or new['waves']>old['waves']:
        reason='local-or-wave-regression'
    elif comparison['structural_relation']!='dominates':
        reason='unresolved-structural-tradeoff'
    else:
        return proposed['arm'],dict(reason='structural-win-with-resource-guard',candidate=new,fallback=old,
                                     comparison=comparison)
    return fallback['arm'],dict(reason=reason,candidate=new,fallback=old,comparison=comparison)


def promotion_verdict(cases, *, expected_cases, complete_scope, relative_target=1/1.03,
                      max_case_regression=1.05, mean_regret_cap=1.03, worst_regret_cap=1.10):
    """Apply frozen limits to independent fixed-strategy evaluation intervals."""
    coverage=set(cases)==set(expected_cases) and bool(cases)
    regressions={}; regret={}; uppers=[]
    for name,item in cases.items():
        ratios=item['fixed_ratios']
        relative=ratios.get('frisk:J/legacy') or ratios.get('frisk:J/frisk:legacy')
        selection=item['descriptive_finite_pool_regret'].get('frisk:J')
        if relative is None or selection is None:
            coverage=False
            continue
        upper=relative['bonferroni_ci95'][1]
        uppers.append(upper)
        if upper>max_case_regression:
            regressions[name]=upper
        regret[name]=selection['ci95'][1]
    geo=lambda values: math.exp(sum(math.log(x) for x in values)/len(values)) if values else None
    g=geo(uppers); rg=geo(list(regret.values())); rw=max(regret.values(),default=None)
    requirements=dict(case_coverage=coverage,requested_operator_scope=complete_scope,
                      suite_relative=g is not None and g<relative_target,case_guard=not regressions,
                      mean_regret=rg is not None and rg<=mean_regret_cap,
                      worst_regret=rw is not None and rw<=worst_regret_cap)
    return dict(status='passed' if all(requirements.values()) else 'held',requirements=requirements,
                per_case_regression_upper=regressions,per_case_regret_upper=regret,
                suite_relative_upper=g,mean_regret_upper=rg,worst_regret_upper=rw)
