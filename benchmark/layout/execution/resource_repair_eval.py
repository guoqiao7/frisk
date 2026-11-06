"""Resource repair validation with prospective specs and explicit evidence levels.

No import-time CUDA work. Old sealed datasets are never rewritten. Kernel timing
is deliberately absent: these experiments test allocations, prediction coverage
and exact compiler feedback against an independent loaded-function query.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from resource_repair_fixture import protocol,canonical_digest,fixture,arms

ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/"guoqiao/optimize_plan/resource_prediction_repair_artifacts"
BENCH=ROOT/"build-m6-release/benchmark/layout/selection/frisk_joint_ablation_bench"
REGISTRATION=ART/"prospective-registration.json"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(ART.resolve()):raise ValueError("refuse output outside new resource-repair evidence root")
    path.write_text(json.dumps(value,indent=2)+"\n")


def checked_run(path):
    run=Path(path).resolve()
    if not run.is_relative_to(ART.resolve()) or run==ART.resolve():
        raise ValueError("a dedicated run inside resource_prediction_repair_artifacts is required")
    return run


def source_identity():
    from cost_calibration import source_identity as source
    return source()


def tools():
    from cost_calibration import freeze_tools
    return freeze_tools()


def check_tools(identity):
    from cost_calibration import check_tools as check
    check(identity)


def command(args,log,timeout=600):
    from cost_calibration import invoke
    return invoke(args,log,timeout=timeout)


def init(path):
    run=checked_run(path);registered=read(REGISTRATION)
    if canonical_digest(registered["protocol"])!=registered["protocol_digest"]:
        raise ValueError("registration checksum mismatch")
    if protocol()!=registered["protocol"]:raise ValueError("live specs/thresholds differ from preregistration")
    run.mkdir(exist_ok=False)
    write(run/"protocol.json",registered["protocol"])
    write(run/"registration.json",dict(registration_sha256=sha(REGISTRATION),protocol_sha256=sha(run/"protocol.json"),
        registered_protocol_digest=registered["protocol_digest"],created_ns=time.time_ns(),
        prospective_native_observations_before_registration=False))
    write(run/"validation-semantics.json",dict(candidate_key="whole spec + complete assignment + conversion policy + post_opt; only duplicate arms with all equal may collapse",
        exact_native_reuse="raw cubin SHA + whole emission config + physical ABI",
        precompile_register_point="None or unsupported is unavailable, never a zero estimate",
        nonzero_local="compiler STACK and raw LOCAL stay separate; CUDA driver local bytes resolve the authoritative allocation"))
    from calibration_analysis import FEATURES
    write(run/"unit-profile.json",dict(schema_version=1,model="sm90-layout-calibrated-v2",coefficients={k:1 for k in FEATURES},
        provenance="resource-repair witness selection only; unit weights; no timing fit"))
    print(str(run),flush=True)


def verify_run(path):
    run=checked_run(path);registered=read(run/"registration.json");p=read(run/"protocol.json")
    if sha(REGISTRATION)!=registered["registration_sha256"] or sha(run/"protocol.json")!=registered["protocol_sha256"] or canonical_digest(p)!=registered["registered_protocol_digest"]:
        raise ValueError("registered protocol identity changed")
    return run,p


def inputs_guard(run):
    paths=[REGISTRATION,run/"registration.json",run/"protocol.json",run/"validation-semantics.json",run/"unit-profile.json"]
    if (run/"model-freeze.json").exists():paths += [run/"model-freeze.json",run/"frozen-model.json"]
    return {str(p):sha(p) for p in paths}


def check_inputs(guard):
    if any(sha(p)!=h for p,h in guard.items()):raise ValueError("registered input/model changed")


def model_for(run,split):
    if split=="known":return None
    frozen=read(run/"model-freeze.json")
    if (sha(run/"frozen-model.json")!=frozen["model_sha256"] or
        sha(run/"known-manifest.json")!=frozen["known_manifest_sha256"] or
        source_identity()!=frozen["source_identity"]):raise ValueError("frozen model/training lineage/source changed")
    return read(run/"frozen-model.json")


def generate(path,split):
    run,p=verify_run(path)
    if split=="prospective":model_for(run,split)
    out=run/split;out.mkdir(exist_ok=False)
    identity=source_identity();native=tools();guard=inputs_guard(run)
    manifest=dict(schema_version=1,split=split,status="generating",root=str(run),identity=identity,tools=native,input_hashes=guard,rows=[])
    try:
        for spec in p["specifications"]:
            if spec["split"]!=split:continue
            base=out/spec["key"];base.mkdir();raw=base/"raw.mlir"
            if spec["case_id"]=="attention":
                suffix={"weak-vector":"fixed","strong-cpasync":"cp-async","strong-tma":"tma"}[spec["abi_stratum"]]
                raw.write_bytes((ROOT/("test/Transforms/attention-layout-"+suffix+"-abi.mlir")).read_bytes())
            else:raw.write_text(fixture(spec))
            prepared=base/"prepared"
            command([BENCH,"--prepare",raw,spec["case_id"],prepared,"--cost-profile="+str(run/"unit-profile.json")],base/"prepare.log")
            for arm in p["candidate_arms"][spec["key"]]:
                d=base/arm["name"];row=dict(spec=spec,arm=arm["name"],conversion=arm["conversion"],path=str(d),status="generating")
                manifest["rows"].append(row);write(run/(split+"-manifest.json"),manifest)
                args=[BENCH,"--prepared="+str(prepared),"--case="+spec["case_id"],"--selection=J","--conversion="+arm["conversion"],
                      "--output-dir="+str(d),"--cost-profile="+str(run/"unit-profile.json")]
                if arm["force_producer"]:args.append("--force-producer="+arm["force_producer"])
                command(args,base/(arm["name"]+".log"))
                record=read(d/"record.json")
                if record.get("status")!="success":raise ValueError("selection did not succeed")
                row.update(status="generated",assignment_sha256=record["complete_assignment_sha256"],post_opt=record["post_opt"],
                           execution_sha256=sha(d/"execution.json"),record_sha256=sha(d/"record.json"))
                write(d/"calibration_spec.json",spec)
        if source_identity()!=identity:raise ValueError("source changed during generation")
        check_tools(native);check_inputs(guard);manifest["status"]="generated"
    except Exception as e:manifest.update(status="failed",error=type(e).__name__+": "+str(e));raise
    finally:write(run/(split+"-manifest.json"),manifest)


def verify_manifest(run,p,split,required_status):
    manifest=read(run/(split+"-manifest.json"))
    if manifest.get("root")!=str(run) or manifest.get("split")!=split or manifest.get("status")!=required_status:
        raise ValueError("same-run phase and status required")
    expected={(s["key"],a["name"]):(s,a) for s in p["specifications"] if s["split"]==split for a in p["candidate_arms"][s["key"]]}
    rows=manifest.get("rows")
    if type(rows) is not list or len(rows)!=len(expected):raise ValueError("incomplete registered candidate matrix")
    seen=set()
    for row in rows:
        key=(row["spec"]["key"],row["arm"])
        if key not in expected or key in seen:raise ValueError("duplicate/unregistered candidate")
        seen.add(key);spec,arm=expected[key]
        if row["spec"]!=spec or row["conversion"]!=arm["conversion"] or Path(row["path"]).resolve()!=run/split/spec["key"]/arm["name"]:
            raise ValueError("candidate specification/path mismatch")
    if source_identity()!=manifest["identity"]:raise ValueError("source changed since generation")
    check_inputs(manifest["input_hashes"])
    return manifest


VALIDATION_GPU_UUID="GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618"


class GpuPreflightError(RuntimeError):
    def __init__(self,state):
        self.state=state
        super().__init__(state["error"])


def check_idle_state(state):
    """An empty successful compute-app query is valid; all other ambiguity fails closed."""
    import csv,io
    for name in ("device","processes"):
        query=state[name]
        if query["exit_code"]!=0 or query["stderr"].strip():
            raise ValueError("untrusted GPU "+name+" query")
    if state["device"]["stdout"].strip()!=VALIDATION_GPU_UUID:
        raise ValueError("GPU UUID mismatch")
    processes=[]
    for fields in csv.reader(io.StringIO(state["processes"]["stdout"]),strict=True):
        if len(fields)!=4:raise ValueError("malformed GPU process row")
        gpu,pid,name,memory=(value.strip() for value in fields)
        if gpu!=VALIDATION_GPU_UUID:raise ValueError("process GPU UUID mismatch")
        if not pid.isascii() or not pid.isdecimal() or int(pid)<=0 or not name:
            raise ValueError("malformed GPU process identity")
        if not memory.isascii() or not memory.isdecimal():raise ValueError("untrusted GPU process memory")
        processes.append(dict(gpu_uuid=gpu,pid=int(pid),process_name=name,used_gpu_memory_mib=int(memory)))
    if processes:raise RuntimeError("foreign GPU workload; do not launch validation")
    return processes


def idle_preflight():
    state=dict(gpu_uuid=VALIDATION_GPU_UUID,created_ns=time.time_ns())
    try:
        for name,fields in (("device","uuid"),("processes","gpu_uuid,pid,process_name,used_gpu_memory")):
            command=["nvidia-smi","-i",VALIDATION_GPU_UUID,"--query-"+("gpu=" if name=="device" else "compute-apps=")+fields,
                     "--format=csv,noheader,nounits"]
            result=subprocess.run(command,text=True,capture_output=True,timeout=30)
            state[name]=dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
        state["compute_processes"]=check_idle_state(state)
        state["status"]="idle"
        return state
    except Exception as error:
        state.update(status="failed",error=type(error).__name__+": "+str(error))
        raise GpuPreflightError(state) from error


def worker(directory,model_path=None,safety=False):
    import cost_calibration as old
    from resource_prediction import predict_resources
    directory=Path(directory).resolve()
    if not directory.is_relative_to(ART.resolve()):raise ValueError("worker path outside resource-repair run")
    model=read(model_path) if model_path else None
    if safety:
        return old.execute(directory,load_only=True,safety=True)
    original_compile=old.compile_exact
    def compile_and_estimate(d):
        execution=read(d/"execution.json")
        before=predict_resources(execution,model=model)
        write(d/"estimate-before-native.json",dict(created_ns=time.time_ns(),input_execution_sha256=sha(d/"execution.json"),
            level="execution-only",prediction=before,compiler_feedback_read=False,driver_read=False))
        original_compile(d)
        # Normal compilation now writes its first pure PTX estimate before
        # resolve_resources performs the first native resource query. Reuse
        # that event, rather than manufacturing an after-the-fact timestamp.
        first=read(d/"resource-pressure-ptx-only.json")
        if first.get("reads_native_resources") is not False:
            raise ValueError("normal compiler lacks PTX-only information provenance")
        write(d/"estimate-ptx-only.json",dict(first,input_execution_sha256=sha(d/"execution.json"),
            input_ptx_sha256=sha(d/"kernel.ptx"),reused_first_pre_feedback_record=True,
            original_record_sha256=sha(d/"resource-pressure-ptx-only.json"),
            snapshot_recorded_after_combined_compile=True))
        resolved=read(d/"resolved-compiler-resources.json")
        write(d/"compiler-resource-feedback.json",dict(recorded_ns=time.time_ns(),resolved=resolved,
            normal_compiler_record_sha256=sha(d/"resolved-compiler-resources.json"),
            process_native_feedback_already_available=True,driver_read=False))
    old.compile_exact=compile_and_estimate
    try:actual=old.execute(directory)
    finally:old.compile_exact=original_compile
    before=read(directory/"estimate-before-native.json");after=read(directory/"estimate-ptx-only.json");feedback=read(directory/"compiler-resource-feedback.json")
    if not before["created_ns"]<after["created_ns"]<feedback["recorded_ns"]:
        raise ValueError("prediction event provenance ordering failure")
    physical=actual["physical_resources"];resolved=feedback["resolved"]
    confirmation=actual.get("compiler_resource_confirmation")
    if not isinstance(confirmation,dict) or confirmation.get("status")!="confirmed":
        raise ValueError("normal runner did not independently confirm compiler/driver resources")
    if (confirmation["loaded_cubin_sha256"]!=sha(directory/"kernel.cubin") or
        confirmation["loaded_cubin_sha256"]!=resolved["compiler_exact"]["identity"]["cubin_sha256"]):
        raise ValueError("actual loaded image and compiler feedback differ")
    if before["prediction"]["shared"]["physical_shared_bytes"]!=physical["shared_bytes"]:
        raise ValueError("structural Shared estimate differs from independent driver")
    for key in ("registers_per_thread","shared_bytes"):
        if confirmation[key]!=physical[key]:raise ValueError("compiler/driver mismatch: "+key)
    if confirmation["local_bytes"]!=physical["local_bytes"]:raise ValueError("driver local resolution mismatch")
    write(directory/"resource-repair-evidence.json",dict(status="verified",estimate_before_native=before,estimate_ptx_only=after,
        compiler_feedback=feedback,driver_resources=physical,driver_confirmation=confirmation,
        authoritative=dict(registers_per_thread=confirmation["registers_per_thread"],shared_bytes=confirmation["shared_bytes"],local_bytes_per_thread=confirmation["local_bytes"]),
        raw_native_shared_bytes=resolved["raw_native_shared_bytes"],
        exact_feedback_is_not_a_prediction=True))
    return actual


def validate(path,split):
    run,p=verify_run(path);manifest=verify_manifest(run,p,split,"generated")
    check_tools(manifest["tools"]);check_inputs(manifest["input_hashes"])
    model=model_for(run,split);model_path=run/"frozen-model.json" if split=="prospective" else None
    accepted={}
    try:
        try:preflight=idle_preflight()
        except GpuPreflightError as error:
            write(run/(split+"-gpu-preflight.json"),error.state)
            raise
        write(run/(split+"-gpu-preflight.json"),preflight)
        for row in manifest["rows"]:
            d=Path(row["path"])
            args=[sys.executable,__file__,"worker",str(d)]
            if model_path:args += ["--model",str(model_path)]
            command(args,d/"numerics.log")
            actual=read(d/"correctness.json");row["resource_evidence"]=str(d/"resource-repair-evidence.json")
            key=(actual["cubin_sha256"],sha(d/"execution_config.json"),canonical_digest(actual["physical_params"]))
            if key in accepted:
                row["safety"]=dict(reused_from=accepted[key],cubin_sha256=key[0],execution_config_sha256=key[1],physical_abi_digest=key[2],rule="exact image/config/ABI")
            else:
                row["safety"]=[]
                for tool in p["safety"]["tools"]:
                    command(["/usr/local/cuda/bin/compute-sanitizer","--tool",tool,"--error-exitcode","91",sys.executable,__file__,"worker",str(d),"--safety"],d/("safety-"+tool+".log"))
                    row["safety"].append(dict(tool=tool,status="passed",cubin_sha256=key[0],log_sha256=sha(d/("safety-"+tool+".log"))))
                accepted[key]=str(d)
            row.update(status="validated",cubin_sha256=key[0],artifact_hashes={str(f.relative_to(d)):sha(f) for f in d.rglob("*") if f.is_file()})
            write(run/(split+"-manifest.json"),manifest)
        if source_identity()!=manifest["identity"]:raise ValueError("source changed during validation")
        check_tools(manifest["tools"]);check_inputs(manifest["input_hashes"]);manifest["status"]="validated"
    except Exception as e:manifest.update(status="failed",error=type(e).__name__+": "+str(e));raise
    finally:write(run/(split+"-manifest.json"),manifest)


def freeze_model(path,model_path=None):
    run,p=verify_run(path);known=read(run/"known-manifest.json")
    if known["status"]!="validated" or source_identity()!=known["identity"]:raise ValueError("current validated known data required before model freeze")
    if (run/"prospective").exists() or (run/"model-freeze.json").exists():raise ValueError("freeze once before prospective generation")
    model=read(model_path) if model_path else None
    write(run/"frozen-model.json",model)
    write(run/"model-freeze.json",dict(model_sha256=sha(run/"frozen-model.json"),known_manifest_sha256=sha(run/"known-manifest.json"),source_identity=source_identity(),
        created_ns=time.time_ns(),prospective_resources_used=False,physical_point_model_configured=model is not None,
        interpretation="None explicitly requires compiler feedback; no missing point estimate is reported as zero error"))


def point_value(prediction):
    value=prediction.get("physical_registers")
    if value is None:return None
    if type(value) in (int,float):number=value
    elif type(value) is dict and value.get("status") in ("estimated","available"):
        number=value.get("registers_per_thread")
    elif type(value) is dict and value.get("status") in ("unavailable","requires-compilation"):
        return None
    else:raise ValueError("invalid physical register estimate schema")
    if type(number) not in (int,float) or not 0<=number<2**31:raise ValueError("invalid physical register point estimate")
    return float(number)


def summarize_evidence(records,thresholds):
    groups={}
    for row in records:
        key=(row["spec"]["key"],row["assignment_sha256"],row["conversion"],row["post_opt"])
        if key in groups:
            fields=("shared_error_bytes","register_prediction","actual_registers","register_error","compiler_driver_verified","local_bytes")
            if any(row[k]!=groups[key][k] for k in fields):raise ValueError("duplicate configuration has inconsistent resource evidence")
        else:groups[key]=row
    values=list(groups.values());prospective=[r for r in values if r["spec"]["split"]=="prospective"]
    all_shared=all(r["shared_error_bytes"]==0 for r in values);exact=all(r["compiler_driver_verified"] for r in values)
    errors=[r["register_error"] for r in prospective if r["register_error"] is not None]
    absent=len(prospective)-len(errors);mae=sum(abs(e) for e in errors)/len(errors) if errors else None
    under=max([max(0.,-e) for e in errors],default=None)
    qualified=bool(prospective) and absent==0 and mae<=thresholds["prospective_mae_max"] and under<=thresholds["prospective_max_underprediction"]
    return dict(observations=len(records),unique_configurations=len(values),prospective_configurations=len(prospective),
        structural_shared_qualified=all_shared,compiler_driver_feedback_verified=exact,
        physical_register_estimator=dict(predicted=len(errors),unavailable=absent,mae=mae,max_underprediction=under,qualified=qualified),
        compilation_feedback_required_for_unqualified=not qualified,
        interpretation="Structural exact allocation and compiler feedback are not a fitted register prediction. Unavailable points remain unavailable.")


def assess(path):
    run,p=verify_run(path);records=[]
    for split in ("known","prospective"):
        m=verify_manifest(run,p,split,"validated")
        check_tools(m["tools"])
        if split=="prospective":model_for(run,split)
        for row in m["rows"]:
            d=Path(row["path"])
            if any(sha(d/k)!=v for k,v in row["artifact_hashes"].items()):raise ValueError("accepted artifact changed")
            evidence=read(d/"resource-repair-evidence.json");point=point_value(evidence["estimate_ptx_only"]["prediction"])
            actual=evidence["driver_resources"];shared=evidence["estimate_before_native"]["prediction"]["shared"]["physical_shared_bytes"]
            records.append(dict(spec=row["spec"],arm=row["arm"],assignment_sha256=row["assignment_sha256"],conversion=row["conversion"],post_opt=row["post_opt"],
                shared_error_bytes=shared-actual["shared_bytes"],register_prediction=point,actual_registers=actual["registers_per_thread"],
                register_error=point-actual["registers_per_thread"] if point is not None else None,
                compiler_driver_verified=evidence["status"]=="verified",local_bytes=actual["local_bytes"],path=str(d)))
    result=dict(protocol_sha256=sha(run/"protocol.json"),model_freeze_sha256=sha(run/"model-freeze.json"),
                summary=summarize_evidence(records,p["acceptance"]["physical_register_estimate"]),rows=records)
    write(run/"assessment.json",result);print(json.dumps(result["summary"]),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("init","generate","validate","worker","freeze-model","assess"));parser.add_argument("path",type=Path)
    parser.add_argument("--split",choices=("known","prospective"),default="known");parser.add_argument("--model",type=Path);parser.add_argument("--safety",action="store_true");args=parser.parse_args()
    if args.command=="init":init(args.path)
    elif args.command=="generate":generate(args.path,args.split)
    elif args.command=="validate":validate(args.path,args.split)
    elif args.command=="worker":worker(args.path,args.model,args.safety)
    elif args.command=="freeze-model":freeze_model(args.path,args.model)
    else:assess(args.path)


if __name__=="__main__":main()
