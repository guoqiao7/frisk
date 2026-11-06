"""Certified selection with partially identified nonnegative native costs.

No compiler, filesystem mutation or CUDA work is performed here.  A bounded
interval is a conditional linear-model family, never a hardware latency bound.
Only a zero training column AND zero coefficient may introduce an independent
unknown nonnegative coefficient. Other unsupported analyses remain unsupported.
"""
from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import native_cost_calibration as calibration

SCHEMA_VERSION = 1
CONTRACT = 'sm90-nonnegative-partial-cost-selection-v1'
BASIS = 'conditional nonnegative additive model family; not a hardware-time bound'
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ValueError('partial cost evidence must be finite JSON') from error


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _profile(profile):
    if not isinstance(profile, dict) or profile.get('model') != 'sm90-native-surviving-schedule-v1' or profile.get('features') != list(calibration.MODEL_FEATURES):
        raise ValueError('native profile feature contract differs')
    names = set(calibration.MODEL_FEATURES)
    for name in ('coefficients', 'training_maxima'):
        values = profile.get(name)
        if not isinstance(values, dict) or set(values) != names or any(not _nonnegative_int(v) for v in values.values()):
            raise ValueError('invalid native ' + name)
    if not isinstance(profile.get('analysis_contract'), dict):
        raise ValueError('native analysis contract missing')
    if any(profile['coefficients'][n] <= 0 for n in ('shared_transfer_bytes', 'barrier_instructions', 'shuffle_instructions')):
        raise ValueError('communication cannot be free')
    return profile['coefficients'], profile['training_maxima']


def cost_interval(native, profile):
    """Return exact, bounded or unsupported cost with complete input bindings.

    ``upper_ticks=None`` denotes positive infinity ONLY for ``bounded``. An
    unsupported record has no mathematical interval: both endpoints are None.
    The profile intercept is excluded, as in calibration.score().
    """
    weights, maxima = _profile(profile)
    if not isinstance(native, dict):
        raise ValueError('native analysis must be an object')
    base = dict(schema_version=SCHEMA_VERSION, contract=CONTRACT, implementation_sha256=IMPLEMENTATION_SHA256,
                native_sha256=_digest(native), profile_sha256=_digest(profile),
                uses_gpu_timing=False, bound_basis=BASIS)

    def unsupported(reason):
        return dict(base, status='unsupported', reason=reason, lower_ticks=None,
                    upper_ticks=None, upper_unbounded=False, unknown_features=[],
                    known_terms={}, unknown_terms={})

    # This narrower typed entry prevents native['supported']=True from turning
    # an explicitly unsupported/unavailable analyzer payload into a bound.
    if native.get('status') != 'available':
        return unsupported('native-analysis-' + str(native.get('status', 'missing-status')))
    support = native.get('support')
    identity = native.get('identity')
    if not isinstance(support, dict) or support.get('status') != 'supported' or support.get('reasons') != []:
        return unsupported('native-support-proof-unavailable')
    if not isinstance(identity, dict) or support.get('domain') != identity.get('domain'):
        return unsupported('native-support-domain-mismatch')
    if native.get('schema_version') != identity.get('schema_version'):
        return unsupported('native-schema-mismatch')
    if not isinstance(identity.get('entry'), str) or not identity['entry']:
        return unsupported('native-entry-unavailable')
    cubin = identity.get('cubin_sha256')
    if not isinstance(cubin, str) or len(cubin) != 64 or any(c not in '0123456789abcdef' for c in cubin):
        return unsupported('native-cubin-identity-unavailable')
    banks = native.get('symbolic_banks')
    if not isinstance(banks, dict) or banks.get('contract') != 'frisk-emitted-affine-shared-bank-v1':
        return unsupported('symbolic-bank-proof-unavailable')
    # Production affine proofs carry this binding. Synthetic micro proofs use
    # a separately validated stride derivation and do not assert the field.
    if 'native_cubin_sha256' in banks and banks['native_cubin_sha256'] != cubin:
        return unsupported('symbolic-bank-cubin-mismatch')
    try:
        actual_contract = calibration.analysis_contract(native)
    except (KeyError, TypeError):
        return unsupported('native-analysis-contract-unavailable')
    if actual_contract != profile['analysis_contract']:
        return unsupported('native-analysis-contract-mismatch')
    # Validate split features BEFORE summing widths: bool+int could otherwise
    # become a superficially valid int and bypass calibration.model_features.
    raw=native.get('features')
    raw_names=(set(calibration.MODEL_FEATURES)-{'shared_transfer_bytes','local_transfer_bytes','symbolic_bank_excess'}) | {
        'shared_load_bytes','shared_store_bytes','local_load_bytes','local_store_bytes'}
    if not isinstance(raw,dict) or set(raw)!=raw_names or any(not _nonnegative_int(v) for v in raw.values()):
        raise ValueError('invalid native feature units/schema')
    if any(not _nonnegative_int(banks.get(n)) for n in ('shared_read_bank_excess','shared_write_bank_excess')):
        raise ValueError('invalid symbolic bank feature units')
    # Malformed evidence is an integrity error, not free/unsupported work.
    try:
        features = calibration.model_features(native)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('invalid partial-cost native features') from error
    result = calibration.score(native, profile)
    unknown = sorted(n for n in calibration.MODEL_FEATURES if features[n] and maxima[n] == 0)
    known_terms = {n: dict(count=features[n], coefficient_ticks=weights[n],
                          contribution_ticks=features[n] * weights[n])
                   for n in calibration.MODEL_FEATURES if n not in unknown}
    lower = sum(v['contribution_ticks'] for v in known_terms.values())
    if result.get('status') == 'supported':
        if unknown or result.get('uses_gpu_timing') is not False or result.get('profile_sha256') != base['profile_sha256']:
            raise ValueError('native exact score contradicts its evidence')
        if not _nonnegative_int(result.get('cost')) or result['cost'] != lower:
            raise ValueError('native exact score differs from known contribution')
        return dict(base, status='exact', reason=None, lower_ticks=lower, upper_ticks=lower,
                    upper_unbounded=False, unknown_features=[], known_terms=known_terms,
                    unknown_terms={}, features=features, native_score=result)
    if result.get('reason') != 'training-zero-feature-activation' or not unknown:
        return unsupported(str(result.get('reason', 'native-score-unsupported')))
    if any(weights[n] != 0 for n in unknown):
        # A positive untrained engineering prior has no registered interval
        # family in v1. Do not silently erase it or invent a coefficient floor.
        return unsupported('unidentified-feature-has-nonzero-coefficient')
    unknown_terms = {n: dict(count=features[n], coefficient_lower_ticks=0,
                            coefficient_upper_ticks=None, point_cost_available=False,
                            interpretation='unidentified nonnegative unit coefficient')
                     for n in unknown}
    return dict(base, status='bounded', reason='training-zero-feature-activation',
                lower_ticks=lower, upper_ticks=None, upper_unbounded=True,
                unknown_features=unknown, known_terms=known_terms, unknown_terms=unknown_terms,
                features=features, native_score=result)


def verify_interval(native, profile, interval):
    expected = cost_interval(native, profile)
    if interval != expected:
        raise ValueError('partial cost interval differs from complete evidence recomputation')
    return expected


def _validate_interval(interval):
    if not isinstance(interval, dict) or interval.get('schema_version') != SCHEMA_VERSION or interval.get('contract') != CONTRACT:
        raise ValueError('wrong partial cost interval schema')
    if interval.get('implementation_sha256') != IMPLEMENTATION_SHA256:
        raise ValueError('partial cost implementation identity differs')
    if interval.get('uses_gpu_timing') is not False or interval.get('bound_basis') != BASIS:
        raise ValueError('partial cost lacks its no-timing/model-family statement')
    for key in ('native_sha256', 'profile_sha256'):
        value = interval.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('partial cost identity missing')
    status = interval.get('status');lower=interval.get('lower_ticks');upper=interval.get('upper_ticks')
    if status == 'unsupported':
        if lower is not None or upper is not None or interval.get('upper_unbounded') is not False:
            raise ValueError('unsupported analysis cannot assert an interval')
    elif status == 'exact':
        if not _nonnegative_int(lower) or upper != lower or type(upper) is not int or interval.get('upper_unbounded') is not False or interval.get('unknown_features') != []:
            raise ValueError('invalid exact interval')
    elif status == 'bounded':
        unknown = interval.get('unknown_features')
        if not _nonnegative_int(lower) or upper is not None or interval.get('upper_unbounded') is not True or not isinstance(unknown, list) or not unknown or unknown != sorted(set(unknown)):
            raise ValueError('invalid unidentified interval')
        terms = interval.get('unknown_terms')
        if not isinstance(terms, dict) or set(terms) != set(unknown):
            raise ValueError('unidentified interval lacks coefficient evidence')
        for value in terms.values():
            if (not isinstance(value,dict) or not _nonnegative_int(value.get('count')) or value['count']==0 or
                value.get('coefficient_lower_ticks') != 0 or type(value['coefficient_lower_ticks']) is not int or
                value.get('coefficient_upper_ticks') is not None or value.get('point_cost_available') is not False):
                raise ValueError('invalid unidentified coefficient family')
    else:
        raise ValueError('unknown partial cost status')
    # Interval selection itself trusts only receipts produced/verified above;
    # still detect accidental receipt edits before they reach decision logic.
    if status != 'unsupported':
        terms = interval.get('known_terms')
        if not isinstance(terms, dict):raise ValueError('known contribution missing')
        total=0
        for value in terms.values():
            if not isinstance(value,dict) or any(not _nonnegative_int(value.get(k)) for k in ('count','coefficient_ticks','contribution_ticks')):
                raise ValueError('invalid known contribution')
            if value['contribution_ticks'] != value['count'] * value['coefficient_ticks']:
                raise ValueError('known contribution product differs')
            total += value['contribution_ticks']
        if lower != total:raise ValueError('known contributions do not sum to bound')
    _canonical(interval)


def select_partial_cost(candidates, *, fallback_arm, priority=None, force_fallback_reason=None):
    """Choose an exact argmin if every unknown lower bound proves it safe.

    All candidate receipts remain in the certificate, including unsupported
    ones. If a bound is equal to the incumbent score, prefer the exact-score
    incumbent, proving argmin membership rather than a unique optimum.
    """
    if not isinstance(candidates, Mapping) or not candidates or any(not isinstance(k,str) or not k for k in candidates):
        raise ValueError('nonempty named candidate pool required')
    if fallback_arm not in candidates:raise ValueError('fallback candidate absent')
    order=sorted(candidates) if priority is None else list(priority)
    if len(order)!=len(candidates) or len(set(order))!=len(order) or set(order)!=set(candidates):
        raise ValueError('priority must contain every candidate exactly once')
    if force_fallback_reason is not None and (not isinstance(force_fallback_reason,str) or not force_fallback_reason):
        raise ValueError('forced fallback reason must be a nonempty string')
    for interval in candidates.values():_validate_interval(interval)
    profiles={v['profile_sha256'] for v in candidates.values()}
    if len(profiles)!=1:raise ValueError('candidate pool mixes cost profiles')
    binding=dict(contract=CONTRACT, implementation_sha256=IMPLEMENTATION_SHA256, candidates={k:_digest(candidates[k]) for k in sorted(candidates)},
                 fallback_arm=fallback_arm, priority=order, force_fallback_reason=force_fallback_reason,
                 profile_sha256=next(iter(profiles)))
    exact=[k for k in order if candidates[k]['status']=='exact']
    hard=[k for k in order if candidates[k]['status']=='unsupported']
    bounded=[k for k in order if candidates[k]['status']=='bounded']
    incumbent=min(exact,key=lambda k:(candidates[k]['lower_ticks'],order.index(k))) if exact else None
    score=candidates[incumbent]['lower_ticks'] if incumbent else None
    blockers=[k for k in bounded if incumbent is not None and candidates[k]['lower_ticks']<score]
    reasons=[]
    if force_fallback_reason is not None:reasons.append('forced:'+force_fallback_reason)
    reasons += ['unsupported:'+k+':'+str(candidates[k].get('reason')) for k in hard]
    if incumbent is None:reasons.append('no-exact-incumbent')
    reasons += ['unknown-lower-bound-can-be-better:'+k for k in blockers]
    selected=fallback_arm if reasons else incumbent
    comparisons=[dict(candidate=k, lower_ticks=candidates[k]['lower_ticks'],
                      incumbent_ticks=score, dominated=score is not None and score<=candidates[k]['lower_ticks'],
                      gap_ticks=candidates[k]['lower_ticks']-score if score is not None else None)
                 for k in bounded]
    certificate=dict(binding=binding,binding_sha256=_digest(binding),
                     status='uncertified-fallback' if reasons else'argmin-member-proven',
                     incumbent=incumbent,incumbent_ticks=score,
                     exact_candidates=exact,bounded_candidates=bounded,unsupported_candidates=hard,
                     unknown_lower_bound_comparisons=comparisons,
                     argmin_membership_only=True,unique_optimum_proven=False,
                     tie_rule='declared-priority-within-exact; exact-incumbent-preferred-at-unknown-bound-equality',
                     interpretation=BASIS)
    return dict(schema_version=SCHEMA_VERSION,contract=CONTRACT,implementation_sha256=IMPLEMENTATION_SHA256,
                status='fallback' if reasons else'interval-certified' if bounded else'exact-ranked',
                selected_source=selected,reasons=reasons,candidates=list(order),
                uses_gpu_timing=False,certificate=certificate)


def verify_selection(candidates, decision, *, fallback_arm, priority=None, force_fallback_reason=None):
    expected=select_partial_cost(candidates,fallback_arm=fallback_arm,priority=priority,
                                 force_fallback_reason=force_fallback_reason)
    if decision != expected:raise ValueError('partial decision differs from complete pool recomputation')
    return expected


def select_from_native(candidates, profile, *, fallback_arm, priority=None, force_fallback_reason=None):
    """Convenience entry that computes every receipt before pool selection."""
    intervals={name:cost_interval(native,profile) for name,native in candidates.items()}
    decision=select_partial_cost(intervals,fallback_arm=fallback_arm,priority=priority,
                                 force_fallback_reason=force_fallback_reason)
    return dict(decision,intervals=intervals)


def verify_from_native(candidates, profile, decision, *, fallback_arm, priority=None, force_fallback_reason=None):
    expected=select_from_native(candidates,profile,fallback_arm=fallback_arm,priority=priority,
                                force_fallback_reason=force_fallback_reason)
    if decision != expected:raise ValueError('partial decision differs from native/profile/pool recomputation')
    return expected
