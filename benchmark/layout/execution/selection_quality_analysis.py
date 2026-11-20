"""Pure paired-block analysis for a preregistered finite executable domain.

Repeated measurements of an identical execution identity are collapsed inside
one batch, so aliases never increase the number of independent blocks. Cost
scores order assignments lexicographically; identical images use their minimum
score, the score which matters when choosing from the complete finite domain.
"""
import hashlib
import json
import math
import numpy as np


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _score(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("nonempty lexicographic model score required")
    if any(type(x) is not int or x < 0 for x in value):
        raise ValueError("model scores must contain nonnegative integers")
    return tuple(value)


def _ratio_summary(a, b, resample, alpha):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratios = a / b
    if not np.all(np.isfinite(ratios)):
        raise ValueError("nonfinite paired ratio")
    replicates = np.median(ratios[resample], axis=1)
    return dict(ratio=float(np.median(ratios)),
                ratio_ci95=np.quantile(replicates, [alpha / 2, 1 - alpha / 2]).tolist(),
                batch_ratios=ratios.tolist())


def analyze_group(records, candidates, selections, *, expected_batches=10,
                  bootstrap_seed=20261102, bootstrap_samples=10000,
                  practical_tolerance=0.03, alpha=0.05):
    """Validate complete paired records and analyze one fixed algorithm/ABI.

    candidates: unique arm dictionaries with cubin_sha256, model_scores mapping
    model names to integer cost tuples (optional for a natural selected-image
    union without complete cross-scoring), and mandatory execution_identity which
    binds bytes, entry, launch config and physical ABI. selections maps each
    selected policy name to one of those candidate arms. Every candidate arm is
    measured in every batch. An invalid or missing batch fails closed. A model
    with any missing candidate scores is explicitly unavailable for ranking;
    the measured regret of all selections still applies to the measured domain.

    Percentile bootstrap samples *whole paired batches*, never event replays.
    Pairwise familywise intervals use Bonferroni over unique-image pairs in this
    group. The max regret bootstrap accounts for choosing the best measured
    image, but is a descriptive finite-set estimate, not a hardware optimum.
    """
    if type(expected_batches) is not int or expected_batches < 2:
        raise ValueError("at least two declared paired batches required")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1000:
        raise ValueError("at least 1000 bootstrap samples required")
    if not 0 < alpha < 1 or not 0 <= practical_tolerance < 1:
        raise ValueError("invalid confidence or practical tolerance")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("nonempty candidate set required")
    if not isinstance(selections, dict) or not selections:
        raise ValueError("selected policies required")
    byarm = {}
    identity_cubins = {}
    model_names = set()
    score_lengths = {}
    for c in candidates:
        arm = c.get("arm")
        identity = c.get("execution_identity")
        cubin = c.get("cubin_sha256")
        if not isinstance(arm, str) or not arm or arm in byarm:
            raise ValueError("duplicate or invalid candidate arm")
        if not isinstance(identity, str) or not identity or not isinstance(cubin, str) or not cubin:
            raise ValueError("full execution identity and cubin hash required")
        if identity in identity_cubins and identity_cubins[identity] != cubin:
            raise ValueError("one execution identity cannot identify multiple images")
        identity_cubins[identity] = cubin
        scores = c.get("model_scores")
        if scores is None:
            scores = {}
        if not isinstance(scores, dict):
            raise ValueError("model score map required when supplied")
        model_names.update(scores)
        normalized = {}
        for model, value in scores.items():
            if not isinstance(model, str) or not model:
                raise ValueError("invalid model name")
            normalized[model] = _score(value)
            if model in score_lengths and len(normalized[model]) != score_lengths[model]:
                raise ValueError("inconsistent lexicographic score dimensions")
            score_lengths[model] = len(normalized[model])
        byarm[arm] = dict(c, model_scores=normalized)
    unavailable_models = {model:dict(status="unavailable-incomplete-domain-scores",
                                    missing_arms=sorted(arm for arm,c in byarm.items() if model not in c["model_scores"]))
                          for model in sorted(model_names) if any(model not in c["model_scores"] for c in byarm.values())}
    model_names -= set(unavailable_models)
    if any(not isinstance(k, str) or not k or v not in byarm for k, v in selections.items()):
        raise ValueError("selection refers to an unmeasured candidate")
    expected_arms = set(byarm)
    batches = {}
    for row in records:
        batch = row.get("batch")
        arm = row.get("arm")
        if type(batch) is not int or batch not in range(expected_batches) or arm not in byarm:
            raise ValueError("undeclared batch or candidate")
        if row.get("valid") is not True:
            raise ValueError("invalid paired batch; no automatic replacement")
        values = batches.setdefault(batch, {})
        if arm in values:
            raise ValueError("duplicate candidate in paired batch")
        if row.get("cubin_sha256") != byarm[arm]["cubin_sha256"]:
            raise ValueError("measured candidate image differs from declaration")
        nodes = row.get("launch_count")
        if type(nodes) is not int or nodes < 1 or type(row.get("actual_graph_nodes")) is not int or row.get("actual_graph_nodes") != nodes:
            raise ValueError("graph launch count mismatch")
        samples = row.get("samples_ms")
        elapsed = row.get("graph_elapsed_ms")
        if not isinstance(samples, list) or not isinstance(elapsed, list) or len(samples) < 2 or len(samples) != len(elapsed):
            raise ValueError("event replay counts differ")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples + elapsed):
            raise ValueError("nonpositive or nonfinite event duration")
        if not np.allclose(samples, np.asarray(elapsed) / nodes, rtol=1e-12, atol=0):
            raise ValueError("event duration equation mismatch")
        content = row.get("pair_content")
        if not isinstance(content, dict) or content.get("batch") != batch:
            raise ValueError("paired content must bind its declared batch")
        expected_pair = hashlib.sha256(_canonical(content).encode()).hexdigest()
        if row.get("pair_id") != expected_pair:
            raise ValueError("paired input identity changed")
        values[arm] = (float(np.median(samples)) * 1000, expected_pair)
    if set(batches) != set(range(expected_batches)):
        raise ValueError("incomplete paired batch set")
    for values in batches.values():
        if set(values) != expected_arms or len({x[1] for x in values.values()}) != 1:
            raise ValueError("incomplete or unmatched paired batch")
    identities = sorted(identity_cubins)
    aliases = {identity: sorted(a for a, c in byarm.items() if c["execution_identity"] == identity)
               for identity in identities}
    times = np.asarray([[np.median([batches[b][a][0] for a in aliases[identity]])
                         for identity in identities] for b in range(expected_batches)])
    if not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("nonfinite derived image duration")
    index = {identity: i for i, identity in enumerate(identities)}
    resample = np.random.default_rng(bootstrap_seed).integers(0, expected_batches,
                                                           (bootstrap_samples, expected_batches))
    pair_count = len(identities) * (len(identities) - 1) // 2
    pair_alpha = alpha / max(1, pair_count)
    comparisons = []
    for i, first in enumerate(identities):
        for j in range(i + 1, len(identities)):
            second = identities[j]
            result = _ratio_summary(times[:, i], times[:, j], resample, alpha)
            simultaneous = _ratio_summary(times[:, i], times[:, j], resample, pair_alpha)
            low, high = simultaneous["ratio_ci95"]
            relation = "first-faster" if high < 1 / (1 + practical_tolerance) else "second-faster" if low > 1 + practical_tolerance else "unresolved-or-practical-tie"
            comparisons.append(dict(first=first, second=second, observed_relation=relation,
                                    ratio=result["ratio"], ratio_ci95=result["ratio_ci95"],
                                    familywise_ratio_ci=simultaneous["ratio_ci95"],
                                    batch_ratios=result["batch_ratios"]))
    models = {}
    for model in sorted(model_names):
        effective = {identity: min(byarm[a]["model_scores"][model] for a in aliases[identity])
                     for identity in identities}
        minimum = min(effective.values())
        best_ids = [identity for identity in identities if effective[identity] == minimum]
        pairs = []
        for pair in comparisons:
            first, second = pair["first"], pair["second"]
            relation = "first-faster" if effective[first] < effective[second] else "second-faster" if effective[first] > effective[second] else "tie"
            observed = pair["observed_relation"]
            status = "unresolved" if observed == "unresolved-or-practical-tie" else "correct" if relation == observed else "model-tie" if relation == "tie" else "inversion"
            pairs.append(dict(first=first, second=second, predicted_relation=relation,
                              observed_relation=observed, status=status))
        models[model] = dict(best_execution_identities=best_ids, effective_scores={k:list(v) for k,v in effective.items()},
                             distinguishable_pairs=sum(p["status"] != "unresolved" for p in pairs),
                             correct_pairs=sum(p["status"] == "correct" for p in pairs),
                             inversions=sum(p["status"] == "inversion" for p in pairs),
                             model_ties_on_distinguishable_pairs=sum(p["status"] == "model-tie" for p in pairs),
                             assignment_score_disagreements_within_image=[identity for identity in identities
                                 if len({byarm[a]["model_scores"][model] for a in aliases[identity]}) > 1], pairs=pairs)
    chosen = {}
    for policy, arm in sorted(selections.items()):
        identity = byarm[arm]["execution_identity"]
        i = index[identity]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratios = times[:, i, None] / times
        if not np.all(np.isfinite(ratios)):
            raise ValueError("nonfinite regret ratio")
        paired_ratios = np.median(ratios, axis=0)
        regret = float(np.max(paired_ratios))
        regret_replicates = np.max(np.median(ratios[resample], axis=1), axis=1)
        chosen[policy] = dict(arm=arm, execution_identity=identity,
                             median_us=float(np.median(times[:, i])),
                             observed_regret_ratio=max(1., regret),
                             regret_ci95=np.quantile(regret_replicates, [alpha/2,1-alpha/2]).tolist(),
                             ratio_of_aggregate_medians_to_best=float(np.median(times[:,i])/np.min(np.median(times,axis=0))),
                             model_optimum_membership={model: identity in models[model]["best_execution_identities"] for model in models})
    policy_comparisons = {}
    policies = sorted(selections)
    for i, a in enumerate(policies):
        for b in policies[i+1:]:
            ia=index[chosen[a]["execution_identity"]];ib=index[chosen[b]["execution_identity"]]
            policy_comparisons[a+"/"+b] = _ratio_summary(times[:,ia],times[:,ib],resample,alpha)
    return dict(schema_version=1, status="complete", paired_batches=expected_batches,
                candidate_assignments=len(candidates), unique_execution_identities=len(identities),
                identity_aliases=aliases, image_batch_medians_us={identity:times[:,index[identity]].tolist() for identity in identities},
                models=models, unavailable_models=unavailable_models, selections=chosen, policy_comparisons=policy_comparisons, image_comparisons=comparisons,
                empirical_best_identities=[identities[i] for i,x in enumerate(np.median(times,axis=0)) if x==np.min(np.median(times,axis=0))],
                method=dict(bootstrap_seed=bootstrap_seed,bootstrap_samples=bootstrap_samples,alpha=alpha,
                            practical_tolerance=practical_tolerance,pairwise_familywise_alpha=pair_alpha,
                            unit="whole paired batch; image aliases collapsed by within-batch median",
                            regret="max over measured execution identities of paired-median selected/peer; paired block bootstrap",
                            scope="registered finite executable domain, not unrestricted hardware optimum"))


def policy_ratio(group, numerator, denominator):
    """Retrieve a comparison in either orientation without ratio-of-medians."""
    if numerator not in group["selections"] or denominator not in group["selections"]:
        raise ValueError("unknown selected policy")
    if numerator == denominator:
        return dict(ratio=1.,ratio_ci95=[1.,1.],batch_ratios=[1.]*group["paired_batches"])
    forward=numerator+"/"+denominator
    if forward in group["policy_comparisons"]:
        return dict(group["policy_comparisons"][forward])
    reverse=denominator+"/"+numerator
    value=group["policy_comparisons"][reverse]
    ratios=np.asarray([1/x for x in value["batch_ratios"]])
    method=group["method"]
    resample=np.random.default_rng(method["bootstrap_seed"]).integers(0,len(ratios),(method["bootstrap_samples"],len(ratios)))
    replicates=np.median(ratios[resample],axis=1)
    return dict(ratio=float(np.median(ratios)),ratio_ci95=np.quantile(replicates,[method["alpha"]/2,1-method["alpha"]/2]).tolist(),
                batch_ratios=ratios.tolist())



def summarize_suite(groups, policies=None, *, bootstrap_seed=20261102,
                    bootstrap_samples=10000, alpha=0.05):
    """Fixed-suite uncertainty with independent paired-block resampling per case.

    `groups` maps registered program keys to analyze_group results from ONE
    domain/cohort (e.g. six new natural holdouts). Do not mix observed anchors or
    duplicate controlled/natural variants as independent programs. This function
    never resamples programs. Every resample recomputes each policy's maximum
    paired-median regret over all measured images, including its own image.
    """
    if not isinstance(groups, dict) or not groups:
        raise ValueError("nonempty fixed suite required")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1000 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap configuration")
    keys=sorted(groups)
    if any(not isinstance(k,str) or not k for k in keys):
        raise ValueError("invalid program key")
    if policies is None:
        policies=sorted(groups[keys[0]]["selections"])
    if not isinstance(policies,(list,tuple)) or not policies or len(set(policies))!=len(policies):
        raise ValueError("unique selected policies required")
    if any(not isinstance(p,str) or not p for p in policies):
        raise ValueError("invalid policy name")
    rng=np.random.default_rng(bootstrap_seed)
    points={p:[] for p in policies};draws={p:[] for p in policies}
    pairs={(a,b):dict(points=[],draws=[]) for a in policies for b in policies if a!=b}
    per_case={}
    for key in keys:
        group=groups[key]
        if group.get("status")!="complete" or any(p not in group.get("selections",{}) for p in policies):
            raise ValueError("incomplete fixed-suite selected policies")
        values=group.get("image_batch_medians_us")
        n=group.get("paired_batches")
        if not isinstance(values,dict) or not values or type(n) is not int or n<2:
            raise ValueError("invalid paired image observations")
        identities=sorted(values)
        index={identity:i for i,identity in enumerate(identities)}
        matrix=np.asarray([values[identity] for identity in identities],dtype=float).T
        if matrix.shape!=(n,len(identities)) or not np.all(np.isfinite(matrix)) or np.any(matrix<=0):
            raise ValueError("invalid fixed-suite image durations")
        resample=rng.integers(0,n,(bootstrap_samples,n))
        selected={}
        per_case[key]={}
        for policy in policies:
            identity=group["selections"][policy]["execution_identity"]
            if identity not in index:raise ValueError("selected image absent from finite reference")
            selected[policy]=matrix[:,index[identity]]
            with np.errstate(over="ignore",divide="ignore",invalid="ignore"):
                ratios=selected[policy][:,None]/matrix
            if not np.all(np.isfinite(ratios)):raise ValueError("nonfinite fixed-suite regret")
            point=float(np.max(np.median(ratios,axis=0)))
            bootstrap=np.max(np.median(ratios[resample],axis=1),axis=1)
            points[policy].append(point);draws[policy].append(bootstrap)
            per_case[key][policy]=dict(observed_regret_ratio=point,
                regret_ci95=np.quantile(bootstrap,[alpha/2,1-alpha/2]).tolist())
        for (a,b),ledger in pairs.items():
            with np.errstate(over="ignore",divide="ignore",invalid="ignore"):
                ratios=selected[a]/selected[b]
            if not np.all(np.isfinite(ratios)):raise ValueError("nonfinite fixed-suite policy ratio")
            ledger["points"].append(float(np.median(ratios)))
            ledger["draws"].append(np.median(ratios[resample],axis=1))
    policy_output={}
    for policy in policies:
        matrix=np.asarray(draws[policy])
        geometric=np.exp(np.mean(np.log(matrix),axis=0));maximum=np.max(matrix,axis=0)
        policy_output[policy]=dict(geomean_regret_ratio=float(np.exp(np.mean(np.log(points[policy])))),
            geomean_regret_ci95=np.quantile(geometric,[alpha/2,1-alpha/2]).tolist(),
            max_case_regret_ratio=float(np.max(points[policy])),
            max_case_regret_ci95=np.quantile(maximum,[alpha/2,1-alpha/2]).tolist(),
            case_regret_ratios=dict(zip(keys,points[policy])))
    comparison_output={}
    for (a,b),ledger in pairs.items():
        geometric=np.exp(np.mean(np.log(np.asarray(ledger["draws"])),axis=0))
        comparison_output[a+"/"+b]=dict(ratio=float(np.exp(np.mean(np.log(ledger["points"])))),
            ratio_ci95=np.quantile(geometric,[alpha/2,1-alpha/2]).tolist(),
            case_paired_median_ratios=dict(zip(keys,ledger["points"])))
    return dict(schema_version=1,status="complete",programs=keys,program_count=len(keys),
        policies=policy_output,policy_comparisons=comparison_output,per_case=per_case,
        method=dict(bootstrap_seed=bootstrap_seed,bootstrap_samples=bootstrap_samples,alpha=alpha,
            unit="independent whole paired-batch bootstrap within each fixed program; programs not resampled",
            reference="max over all measured images recomputed in each resample",
            scope="one registered cohort and domain; no population generalization or unrestricted hardware optimum"))



def paired_policy_interval(group, numerator, denominator, *, alpha=0.05):
    """Recompute a declared policy interval, e.g. a familywise case guard.

    The caller must predeclare the family and its alpha allocation. Stored raw
    paired ratios are resampled; a reciprocal median is never approximated by
    the reciprocal of a median (those differ for an even number of blocks).
    """
    if not 0 < alpha < 1:raise ValueError("invalid interval alpha")
    ratios=np.asarray(policy_ratio(group,numerator,denominator)["batch_ratios"])
    method=group["method"]
    resample=np.random.default_rng(method["bootstrap_seed"]).integers(0,len(ratios),
        (method["bootstrap_samples"],len(ratios)))
    draws=np.median(ratios[resample],axis=1)
    return dict(ratio=float(np.median(ratios)),ratio_ci=np.quantile(draws,[alpha/2,1-alpha/2]).tolist(),
        alpha=alpha,confidence=1-alpha,paired_batches=len(ratios),
        batch_ratios=ratios.tolist(),method="paired whole-block percentile bootstrap")
