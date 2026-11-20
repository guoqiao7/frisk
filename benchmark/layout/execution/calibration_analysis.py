"""Deterministic train-only calibration and held-out ranking/resource analysis."""
import hashlib
import json
import numpy as np

FEATURES = ("access_warp_issues","global_sectors32","shared_bank_excess","copy_setup_units","mma_instructions",
            "reduction_element_ops","alu_warp_issues","div_warp_issues","exp_warp_issues","cast_warp_issues",
            "conversion_register_bytes","conversion_shuffle_bytes","conversion_shared_bytes",
            "conversion_conservative_bytes","synchronization_barriers")

def features(record):
    value=record["cost"].get("features")
    if not isinstance(value,dict) or set(value)-set(FEATURES):
        raise ValueError("missing/unrecognized v2 feature ledger")
    result={name:value.get(name,0) for name in FEATURES}
    if any(type(x) is not int or x<0 for x in result.values()):
        raise ValueError("features must be nonnegative integers")
    return result

def coordinate_fit(x,y,ridge=1e-6,max_iterations=20000,tolerance=1e-10,*,signed=False,fit_intercept=True):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if x.ndim!=2 or y.shape!=(len(x),) or len(x)<2 or np.any(~np.isfinite(x)) or (not signed and np.any(x<0)) or np.any(~np.isfinite(y)) or (not signed and np.any(y<0)):
        raise ValueError("invalid training matrix")
    maxima=np.max(np.abs(x),axis=0);active=maxima>0
    a=x[:,active]/maxima[active]
    if fit_intercept:a=np.column_stack([np.ones(len(x)),a])
    # Lawson-Hanson active-set NNLS on the ridge-augmented design.
    penalty=np.sqrt(ridge)*np.eye(a.shape[1])
    if fit_intercept:penalty[0,0]=0
    augmented=np.vstack([a,penalty]);target=np.r_[y,np.zeros(a.shape[1])]
    weights=np.zeros(a.shape[1]);passive=np.zeros(a.shape[1],dtype=bool);converged=False
    iteration=0
    while iteration<max_iterations:
        gradient=augmented.T@(target-augmented@weights)
        if not np.any(~passive) or np.max(gradient[~passive])<=tolerance:
            converged=True;break
        eligible=np.where(~passive,gradient,-np.inf);passive[int(np.argmax(eligible))]=True
        while True:
            iteration+=1
            proposed=np.zeros_like(weights)
            proposed[passive]=np.linalg.lstsq(augmented[:,passive],target,rcond=None)[0]
            if np.all(proposed[passive]>0):
                weights=proposed;break
            negative=passive&(proposed<=0)
            alpha=np.min(weights[negative]/(weights[negative]-proposed[negative]))
            weights+=alpha*(proposed-weights)
            passive[passive&(weights<=tolerance)]=False;weights[~passive]=0
            if iteration>=max_iterations:break
    residual=y-a@weights
    coefficients=np.zeros(x.shape[1]);coefficients[active]=weights[1 if fit_intercept else 0:]/maxima[active]
    gradient=augmented.T@(target-augmented@weights)
    kkt=max(float(np.max(np.abs(gradient[passive]))) if np.any(passive) else 0.,
            max(0.,float(np.max(gradient[~passive]))) if np.any(~passive) else 0.)
    return dict(coefficients=coefficients.tolist(),intercept=float(weights[0]) if fit_intercept else 0.,iterations=iteration,converged=converged,kkt_violation=kkt,
                design_rank=int(np.linalg.matrix_rank(a)),design_columns=int(a.shape[1]),active_columns=active.tolist(),
                training_rmse=float(np.sqrt(np.mean(residual**2))),prediction=(y-residual).tolist())

def fit_profile(rows,protocol):
    # Input contains train and holdout rows; holdout data are deliberately never read.
    train=[r for r in rows if r["spec"]["split"]=="train" and r.get("valid",True)]
    if len({r["spec"]["key"] for r in train})<2:raise ValueError("at least two complete training specifications required")
    x=[[r["features"][k] for k in FEATURES] for r in train]
    y=[r["median_us"] for r in train]
    if any(not np.isfinite(v) or v<=0 for v in y):raise ValueError("invalid kernel time")
    fit=coordinate_fit(x,y,**{k:protocol["fit"][k] for k in ("ridge","max_iterations","tolerance")})
    if not fit["converged"]:raise ValueError("training optimizer did not converge")
    scale=protocol["fit"]["ticks_per_microsecond"];inactive=protocol["fit"]["inactive_coefficient_ticks"]
    coefficients={k:(int(round(v*scale)) if active else inactive) for k,v,active in zip(FEATURES,fit["coefficients"],fit["active_columns"])}
    identity=[dict(key=r["spec"]["key"],candidate=r["candidate"],features=r["features"],median_us=r["median_us"],
                   cubin_sha256=r["cubin_sha256"],measurement_sha256=r["measurement_sha256"]) for r in train]
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    profile=dict(schema_version=1,model="sm90-layout-calibrated-v2",coefficients=coefficients,
                 provenance="train-only nonnegative-ridge-v1; training-ledger-sha256="+digest)
    fit.update(training_ledger_sha256=digest,training_specs=sorted({r["spec"]["key"] for r in train}),
               coefficient_units="selection ticks per static model feature; fitted to training kernel time, not architectural latency",
               inactive_features=[k for k,a in zip(FEATURES,fit["active_columns"]) if not a],
               identifiability="individual coefficients not identifiable when rank < columns; ridge selects one solution")
    return profile,fit

def prediction_us(feature,profile,fit,scale=1000000):
    return fit["intercept"]+sum(feature[k]*profile["coefficients"][k] for k in FEATURES)/scale

def assess(rows,profile,fit):
    output=[]
    for r in rows:
        if not r.get("valid",True):continue
        predicted=prediction_us(r["features"],profile,fit)
        output.append(dict(key=r["spec"]["key"],split=r["spec"]["split"],candidate=r["candidate"],
                           observed_us=r["median_us"],predicted_us=predicted,error_us=predicted-r["median_us"],
                           relative_error=(predicted-r["median_us"])/r["median_us"]))
    summaries={}
    for split in ("train","holdout"):
        subset=[x for x in output if x["split"]==split]
        summaries[split]=dict(n=len(subset),mae_us=float(np.mean([abs(x["error_us"]) for x in subset])) if subset else None,
                              max_absolute_relative_error=max([abs(x["relative_error"]) for x in subset],default=None))
    return dict(rows=output,summaries=summaries,warning="time fit is bounded empirical proxy; no universal latency/throughput guarantee")

def ranking(candidates,selected):
    if not candidates or any(r.get("valid") is False for r in candidates):return dict(status="incomplete")
    byid={r["cubin_sha256"]:r for r in candidates}
    result=[]
    for row in selected:
        peer=byid.get(row["cubin_sha256"])
        if peer is None:
            result.append(dict(arm=row["arm"],status="new-unmeasured-image"));continue
        best=min(r["median_us"] for r in candidates)
        result.append(dict(arm=row["arm"],status="measured",median_us=peer["median_us"],
                           observed_regret_ratio=peer["median_us"]/best,
                           empirical_best_us=best,domain="measured executable candidate set, not unrestricted GPU optimum"))
    return result

def paired_analysis(records,arms,seed=20261031):
    complete=[];invalid=[]
    for batch in sorted({r["batch"] for r in records}):
        rows=[r for r in records if r["batch"]==batch]
        if len(rows)!=len(arms) or {r["arm"] for r in rows}!=set(arms) or any(not r["valid"] for r in rows):
            invalid.append(batch);continue
        ids={r["pair_id"] for r in rows}
        if len(ids)!=1:raise ValueError("pair identity mismatch")
        values={}
        for r in rows:
            expected=hashlib.sha256(json.dumps(r["pair_content"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
            if expected!=r["pair_id"]:raise ValueError("pair content changed")
            if r["launch_count"]!=r["actual_graph_nodes"] or r["launch_count"]<1:raise ValueError("invalid graph count")
            if len(r["samples_ms"])<2 or len(r["samples_ms"])!=len(r["graph_elapsed_ms"]):raise ValueError("invalid events")
            if any(not np.isfinite(x) or x<=0 for x in r["samples_ms"]):raise ValueError("invalid timing")
            if not np.allclose(r["samples_ms"],np.array(r["graph_elapsed_ms"])/r["launch_count"],rtol=1e-12,atol=0):raise ValueError("event equation mismatch")
            values[r["arm"]]=float(np.median(r["samples_ms"]))*1000
        complete.append([values[a] for a in arms])
    result=dict(n=len(complete),formal=len(complete)==10,invalid_batches=invalid,arms={},comparisons={})
    if not complete:return result
    m=np.asarray(complete);rng=np.random.default_rng(seed);index=rng.integers(0,len(m),(10000,len(m)))
    for i,a in enumerate(arms):result["arms"][a]=dict(median_us=float(np.median(m[:,i])),batch_medians_us=m[:,i].tolist())
    for i in range(len(arms)):
        for j in range(i+1,len(arms)):
            ratios=m[:,i]/m[:,j]
            result["comparisons"][arms[i]+"/"+arms[j]]=dict(ratio=float(np.median(ratios)),ratio_ci95=np.percentile(np.median(ratios[index],axis=1),[2.5,97.5]).tolist())
    return result


def legacy_ssa_resource_calibration(rows,frozen_models=None):
    """Train an empirical proxy conversion; keep unknown/multi-scope rows explicit."""
    samples=[];unavailable=[]
    for row in rows:
        scopes=[x for x in row["resource_comparison"]["scopes"] if x.get("status")=="available" and not x.get("saturated",True)]
        if len(scopes)!=1:
            unavailable.append(dict(key=row["spec"]["key"],candidate=row["candidate"],reason="requires exactly one available unsaturated execution scope"));continue
        scope=scopes[0];actual=row["physical_resources"]
        samples.append(dict(key=row["spec"]["key"],split=row["spec"]["split"],candidate=row["candidate"],
             register_features=[scope["logical_live_carrier_bytes_per_thread"]/4,scope["logical_live_carrier_slots_per_thread"]],
             shared_features=[scope["logical_live_shared_root_bytes"],scope["declared_shared_root_bytes"]],
             registers_per_thread=actual["registers_per_thread"],shared_bytes=actual["shared_bytes"],local_bytes=actual["local_bytes"],
             theoretical_residency=actual["residency"],backend_allocation_components=row["resource_comparison"].get("backend_allocation_components",{})))
    output=dict(models={},rows=[],unavailable=unavailable,meaning="empirical mapping of logical proxies to backend allocation, not a replacement for register allocation or measured occupancy")
    for target,feature in (("registers_per_thread","register_features"),("shared_bytes","shared_features")):
        if frozen_models is not None:
            output["models"][target]=frozen_models[target];continue
        train=[r for r in samples if r["split"]=="train"]
        if len(train)<2:
            output["models"][target]=dict(status="unavailable",reason="insufficient train-only resource samples");continue
        model=coordinate_fit([r[feature] for r in train],[r[target] for r in train])
        if not model["converged"] or model["kkt_violation"]>1e-8*max(1.,max(r[target] for r in train)):
            output["models"][target]=dict(status="unavailable",reason="resource fit did not converge or violates KKT",diagnostic=model);continue
        model.update(training_specs=sorted({r["key"] for r in train}),status="fitted")
        output["models"][target]=model
    for row in samples:
        result={k:v for k,v in row.items() if not k.endswith("_features")}
        for target,feature in (("registers_per_thread","register_features"),("shared_bytes","shared_features")):
            model=output["models"][target]
            if model["status"]=="fitted":
                prediction=model["intercept"]+float(np.dot(model["coefficients"],row[feature]))
                result[target+"_predicted"]=prediction;result[target+"_error"]=prediction-row[target]
        output["rows"].append(result)
    output["errors"]={}
    for split in ("train","holdout"):
        output["errors"][split]={}
        for target in ("registers_per_thread","shared_bytes"):
            values=[r[target+"_error"] for r in output["rows"] if r["split"]==split and target+"_error" in r]
            output["errors"][split][target]=dict(n=len(values),mae=float(np.mean(np.abs(values))) if values else None,max_abs=max(map(abs,values),default=None))
    return output


def resource_calibration(rows,frozen_models=None):
    """Versioned physical resource report. Legacy fit is explicit and unqualified.

    Default reports never infer physical registers from logical SSA slots. If
    export/PTX/cubin evidence is absent, exact counts remain unavailable.
    """
    if frozen_models is not None and "registers_per_thread" in frozen_models:
        result=legacy_ssa_resource_calibration(rows,frozen_models)
        result.update(resource_model_version="legacy-ssa-regression",physical_prediction_qualified=False,
                      warning="historical compatibility only; use lowering/compiler resources for physical counts")
        return result
    if frozen_models is not None and frozen_models.get("resource_model_version")!="lowering-compiler-v2":
        raise ValueError("unknown resource model version")
    from pathlib import Path
    from resource_prediction import resolve_resources, confirm_driver_resources
    result=dict(models=dict(resource_model_version="lowering-compiler-v2"),resource_model_version="lowering-compiler-v2",
                meaning="structural Shared allocation plus exact postcompile feedback; PTX pressure is not a physical allocation",
                physical_prediction_qualified=False,rows=[],unavailable=[],errors={})
    for row in rows:
        record=row.get("record_path")
        directory=Path(record).parent if record else None
        needed=("execution.json","kernel.ptx","kernel.cubin")
        if directory is None or any(not (directory/name).is_file() for name in needed):
            result["unavailable"].append(dict(key=row["spec"]["key"],candidate=row["candidate"],
                reason="lowering/native evidence required; logical SSA fit is not physical resource evidence"))
            continue
        data=json.loads((directory/"execution.json").read_text())
        resolved=resolve_resources(data,(directory/"kernel.ptx").read_text(),directory/"kernel.cubin",
                                   expected_cubin_sha256=row.get("cubin_sha256"))
        actual=row["physical_resources"]
        identity=resolved["compiler_exact"]["identity"]
        if "reserved_shared_bytes_per_block" in actual.get("device_limits",{}):
            confirmation=confirm_driver_resources(resolved,actual,
                loaded_cubin_sha256=row.get("cubin_sha256"),loaded_entry=identity["entry"])
        else:
            confirmation=dict(status="unconfirmed-missing-target-reserved-property",
                              reason="historical driver record lacks reserved Shared; raw native is not user allocation")
        result["rows"].append(dict(key=row["spec"]["key"],split=row["spec"]["split"],candidate=row["candidate"],
             registers_per_thread=actual["registers_per_thread"],shared_bytes=actual["shared_bytes"],local_bytes=actual["local_bytes"],
             shared_bytes_predicted=resolved["prediction"]["shared"]["physical_shared_bytes"],
             shared_bytes_error=resolved["prediction"]["shared"]["physical_shared_bytes"]-actual["shared_bytes"],
             registers_per_thread_predicted=None,register_prediction_status="requires-compilation",
             registers_per_thread_compiler_exact=resolved["registers_per_thread"],
             compiler_register_error=resolved["registers_per_thread"]-actual["registers_per_thread"],
             raw_compiler_shared_bytes=resolved["raw_native_shared_bytes"],
             native_resource_confirmation=confirmation,
             compiler_shared_error=(confirmation["shared_bytes"]-actual["shared_bytes"]
                                    if confirmation["status"]=="confirmed" else None),
             prediction_stage="execution-export / PTX-only",physical_counts_source="compiled-cubin",
             virtual_pressure=resolved["prediction"]["registers"],backend_allocation_plan=resolved["prediction"]["shared"]))
    for split in ("train","holdout","known","prospective"):
        selected=[r for r in result["rows"] if r["split"]==split]
        shared=[r["shared_bytes_error"] for r in selected]
        result["errors"][split]=dict(registers_per_thread=dict(n=0,mae=None,max_abs=None,status="not-claimed-as-prediction"),
                                    shared_bytes=dict(n=len(shared),mae=float(np.mean(np.abs(shared))) if shared else None,
                                                      max_abs=max(map(abs,shared),default=None)))
    return result


def assess_selected(rows,profile,fit,resource_models):
    nonlegacy=[r for r in rows if r.get("features") is not None]
    report=dict(timing=assess(nonlegacy,profile,fit),resources=resource_calibration(rows,resource_models),selection_groups=[])
    for key in sorted({r["spec"]["key"] for r in rows}):
        group=[r for r in rows if r["spec"]["key"]==key]
        best=min(r["median_us"] for r in group)
        report["selection_groups"].append(dict(key=key,split=group[0]["spec"]["split"],measured_arms=len(group),
            fastest_observed_us=best,fastest_observed_arms=[r["arm"] for r in group if r["median_us"]==best],
            rows=[dict(arm=r["arm"],median_us=r["median_us"],empirical_regret=r["median_us"]/best,
                       cubin_sha256=r["cubin_sha256"]) for r in group],
            interpretation="descriptive regret against fastest observed executable in this paired candidate set, not an unrestricted optimum"))
    return report
