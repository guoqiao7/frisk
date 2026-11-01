"""Prospective whole-program resource validation, separate from sealed calibration."""
import hashlib
import json
from calibration_fixture import specifications as previous_specifications
from calibration_fixture import fixture, inputs, reference


def specifications():
    known=[]
    for original in previous_specifications():
        item=dict(original)
        item.update(key="known-"+original["key"], split="known",
                    previous_key=original["key"], previous_split=original["split"],
                    role="previously-observed-regression-and-development")
        known.append(item)
    def new(key,case,shape,tail="none"):
        return dict(key="prospective-"+key,split="prospective",case_id=case,
                    shape=list(shape),tail=tail,abi_stratum="weak-vector",dtype="float32",threads=128,
                    role="never-used-for-fitting-or-model-selection")
    return known+[
        new("diamond-16x64","diamond",(16,64)),
        new("remat-32x64","remat",(32,64)),
        new("mixed-128x32","no-conflict",(128,32),"mixed"),
        new("div-64x64","no-conflict",(64,64),"div6"),
        new("conflict-256x8","conflict",(256,8)),
        new("exp-16x128","no-conflict",(16,128),"exp1"),
    ]


def program_identity(spec):
    return tuple([spec["case_id"],tuple(spec["shape"]),spec["tail"],spec["abi_stratum"],spec["dtype"],spec["threads"]])


def arms(spec):
    if spec["case_id"]=="attention":
        return [dict(name="J-"+policy, selection="J", conversion=policy,force_producer=None) for policy in ("C","O")]
    return [dict(name=layout+"-"+policy,selection="J",conversion=policy,force_producer=layout)
            for layout in ("A","B") for policy in ("C","O")]


def protocol():
    specs=specifications()
    if len({x["key"] for x in specs})!=len(specs):raise ValueError("duplicate spec key")
    if len({program_identity(x) for x in specs})!=len(specs):raise ValueError("whole-program split overlaps")
    return dict(schema_version=1,stage="resource-prediction-repair",seed=20261031,
        specifications=specs,
        split_rule="complete program/shape/ABI; every layout and conversion arm remains in the same split",
        previous_holdout_rule="all twelve old specifications, including their three holdouts, are now known development/regression cases",
        candidate_arms={x["key"]:arms(x) for x in specs},
        acceptance=dict(structural_shared=dict(supported_error_bytes=0,unsupported="explicit, never zero-filled"),
                        compiler_feedback=dict(register_error=0,shared_error_bytes=0,local_semantics="independent driver comparison; no assumed stack/local sum"),
                        physical_register_estimate=dict(prospective_mae_max=8.,prospective_max_underprediction=16.,
                            unsupported_or_threshold_failure="not-qualified; compilation feedback required; do not silently clip")),
        estimation_levels=["execution-only structural shared plan", "PTX-only pre-register-allocation pressure",
                           "optional frozen physical register estimator", "compiler-exact resource feedback", "independent loaded-function CUDA driver attributes"],
        fitting_rule="fit/select parameters on known cases only; freeze complete model and source/tool identities before compiling or inspecting prospective native resources",
        prospective_rule="no model, hyperparameter, formula, or specification updates from prospective results; failures remain in the denominator",
        metrics=dict(primary="deduplicate by complete assignment plus algorithm/ABI/spec; do not count identical arms as independent programs",
                     also_report="all arm observations and exact native-image/config duplicates; MAE/max signed underprediction and unsupported coverage"),
        numerics=dict(float32="bit-exact per-operation unless exp tail",exp="finite abs(error)<=1e-5+2e-6*abs(reference)",
                      attention="frozen f16 QKV / f32 QK+stable-softmax / RNE f16 P / f32 PV; abs(error)<=.002+.002*abs(reference)",
                      offsets="ordinary weak1/2/3; strong inputs0/128/256 crossed with weak outputs1/2/3"),
        safety=dict(tools=["memcheck","racecheck","synccheck","initcheck"],reuse="exact raw cubin plus complete emission config and physical launch ABI only"),
        no_kernel_timing=True, target="static SM90 single CTA / 128 threads; supported backend subset only")


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
