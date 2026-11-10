"""One-stage reproducible calibration: frozen specifications, exact execution, fit, real reselection.
No import-time CUDA or subprocess work. All output directories are append-only/fresh.
"""
import argparse
import ctypes
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
from calibration_fixture import specifications, fixture, inputs, reference, protocol, protocol_digest, SEED
from calibration_analysis import FEATURES, features, fit_profile, assess, paired_analysis, resource_calibration, assess_selected

ROOT=Path(__file__).resolve().parents[3]
BENCH=ROOT/"build-m6-release/benchmark/layout/selection/frisk_joint_ablation_bench"
UUID="GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618"

FEATURE_SEMANTICS=dict(schema_version=1,model="sm90-layout-calibrated-v2",features=list(FEATURES),
    synchronization_barriers="modeled Copy plus conversion CTA barrier instances, multiplied by known boundary frequency",
    copy_global_to_shared=dict(vector_cta_barriers=1,cp_async_cta_barriers=1,tma_cta_barriers=2),
    conversion_cta_barriers="route proof count; ordinary Shared exchange has two, Register/Shuffle have none",
    excluded_sync="mbarrier polling, proxy fences, MMA waits and cp.async wait instructions are not CTA barrier feature instances",
    coverage="partial static features; cache, overlap, dependency latency, MMA Shared read bank behavior and scalar control remain unmodeled",
    resource_policy="physical allocation and theoretical residency are observed separately, not hidden in selection ticks",
    timing_units="nonnegative fitted selection ticks; empirical time mapping includes a separate fitted intercept, not an architectural latency guarantee",
    registration_note="pre-data feature semantics clarification; does not change frozen specification split, hyperparameters, ABI or numerical thresholds")

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2))
def fresh(path):
    path=Path(path)
    path.mkdir(parents=True,exist_ok=False)
    return path

def source_paths(*,root=None):
    source_root=ROOT if root is None else Path(root)
    suffixes={".cpp",".cc",".c",".h",".hpp",".td",".py",".txt",".cmake",".mlir"}
    paths={}
    for folder in ("include","lib","tools","benchmark","unittests","test","cmake"):
        base=source_root/folder
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file() and p.suffix in suffixes:paths[str(p.relative_to(source_root))]=p
    for p in source_root.glob("CMakeLists.txt"):paths[str(p.relative_to(source_root))]=p
    return paths


def source_identity():
    from integrity_session import active, verified_sha
    paths=source_paths()
    if active() is not None:
        root=ROOT
        active().inventory("source:"+str(root),paths,lambda:source_paths(root=root))
    files={name:(verified_sha(path) if active() is not None else sha(path)) for name,path in paths.items()}
    return dict(schema_version=1,source_manifest=files,source_digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest())


def input_hashes(run,profile=None):
    paths=[run/"protocol.json",run/"registration.json",run/"unit-profile.json",run/"feature-semantics.json"]
    if profile:paths.append(Path(profile))
    return {str(p):sha(p) for p in paths}

def check_inputs(expected):
    for p,h in expected.items():
        if sha(p)!=h:raise ValueError("profile/protocol identity changed: "+p)

def freeze_tools():
    from triton.backends.nvidia.compiler import get_ptxas
    import triton,torch
    ctypes.CDLL("libcuda.so.1")
    paths=[BENCH,Path(sys.executable),Path(get_ptxas().path),Path(triton.__file__).parent/"_C/libtriton.so",Path(torch._C.__file__),
           Path("/usr/local/cuda/bin/compute-sanitizer"),Path("/usr/local/cuda/bin/cuobjdump")]
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields=line.split()
        if len(fields)>=6 and fields[-1].startswith("/") and any(k in fields[-1] for k in ("libcuda.so","libtorch_cuda.so","libc10_cuda.so","libcudart.so")):
            paths.append(Path(fields[-1]))
    return dict(python=sys.version,torch=torch.__version__,triton=triton.__version__,
                hashes={str(p.resolve()):sha(p) for p in paths})

def check_tools(identity):
    for p,h in identity["hashes"].items():
        if sha(p)!=h:raise ValueError("tool identity changed: "+p)

def documented_outcome(code,directory):
    """Only the public code-3 failure protocol may become a retained unavailable row."""
    if code!=3:raise RuntimeError("non-documented worker exit "+str(code))
    value=read(Path(directory)/"failure.json")
    allowed={"no-feasible","unknown","search-budget","unsupported","backend-unsupported","timeout"}
    if (type(value) is not dict or set(value)!={"status","failure_kind","fatal","reason","limits"} or
            value.get("status")!="failed" or value.get("fatal") is not False or
            type(value.get("failure_kind")) is not str or value.get("failure_kind") not in allowed or type(value.get("reason")) is not str or not value["reason"] or
            type(value.get("limits")) is not dict or set(value["limits"])!={"probes","refinement"} or
            any(type(n) is not int or n<0 or n>2**63-1 for n in value["limits"].values())):
        raise ValueError("malformed or fatal failure cannot become unavailable")
    return dict(value,exit_code=3,status="documented-failure")

def invoke(args,log,timeout=600,allow_documented_directory=None):
    from run_joint_ablation import command
    code=command([str(x) for x in args],Path(log),timeout_seconds=timeout)
    if code:
        if allow_documented_directory is not None:return documented_outcome(code,allow_documented_directory)
        raise RuntimeError("worker failed code="+str(code)+" log="+str(log))
    return dict(exit_code=0,status="success")

def init(out):
    out=fresh(out);p=protocol()
    p["numerics"]=dict(ordinary_f32="bit-exact per-operation float32",exp_tail="finite abs(error)<=1e-5+2e-6*abs(reference)",
                       attention="existing independent FP32 accum / stable softmax / RNE f16 P reference, abs(error)<=.002+.002*abs(ref)")
    write(out/"protocol.json",p)
    write(out/"feature-semantics.json",FEATURE_SEMANTICS)
    write(out/"unit-profile.json",dict(schema_version=1,model="sm90-layout-calibrated-v2",
            coefficients={k:1 for k in FEATURES},provenance="uncalibrated unit features for exhaustive controlled producer sampling; no latency claim"))
    write(out/"registration.json",dict(protocol_sha256=sha(out/"protocol.json"),protocol_digest=protocol_digest(p),time_ns=time.time_ns(),
                                      rule="freeze all specifications and optimizer hyperparameters before any measured data"))
    print(str(out));return out

def verify_protocol(run):
    p=read(run/"protocol.json");registered=read(run/"registration.json")
    if sha(run/"protocol.json")!=registered["protocol_sha256"] or protocol_digest(p)!=registered["protocol_digest"]:
        raise ValueError("preregistered protocol changed")
    return p

def generate(run,phase,profile=None):
    registered=verify_protocol(run);out=fresh(run/phase)
    identity=source_identity();tools=freeze_tools();inputs_guard=input_hashes(run,profile)
    manifest=dict(schema_version=1,phase=phase,status="generating",identity=identity,tools=tools,
                  protocol_sha256=sha(run/"protocol.json"),input_hashes=inputs_guard,rows=[])
    write(run/(phase+"-manifest.json"),manifest)
    try:
        for spec in registered["specifications"]:
            case=spec["case_id"];base=fresh(out/spec["key"]);raw=base/"raw.mlir"
            if case=="attention":
                suffix={"weak-vector":"fixed","strong-cpasync":"cp-async","strong-tma":"tma"}[spec["abi_stratum"]]
                raw.write_text((ROOT/("test/Transforms/attention-layout-"+suffix+"-abi.mlir")).read_text())
            else:raw.write_text(fixture(spec))
            profiles=[("unit",run/"unit-profile.json")] if phase=="initial" else [("legacy",None),("calibrated",Path(profile))]
            for model,pfile in profiles:
                prepared=base/("prepared-"+model)
                args=[BENCH,"--prepare",raw,case,prepared]+(["--cost-profile="+str(pfile)] if pfile else [])
                invoke(args,base/("prepare-"+model+".log"))
                prepared_meta=read(prepared/"prepared.json")
                if phase=="selected":
                    reference_meta=read(run/"initial"/spec["key"]/"prepared-unit"/"prepared.json")
                    for digest_name in ("domain_digest","coverage_digest"):
                        if prepared_meta[digest_name]!=reference_meta[digest_name]:
                            raise ValueError("model changed common candidate/legal-proof domain: "+digest_name)
                if phase=="initial":
                    arms=[(a+"-"+c,"J",c,a) for a in ("A","B") for c in ("C","O")] if case!="attention" else [(s+"-"+c,s,c,None) for s in ("L","LI","J") for c in ("C","O")]
                else:
                    selectors=("L","LI","J") if model=="legacy" and case=="attention" and spec["abi_stratum"]=="weak-vector" else ("LI","J")
                    arms=[(model+"-"+s,s,"C",None) for s in selectors]
                for name,selection,conversion,force in arms:
                    d=base/name
                    args=[BENCH,"--prepared="+str(prepared),"--case="+case,"--selection="+selection,
                          "--conversion="+conversion,"--output-dir="+str(d),"--oracle"]
                    if pfile:args+=["--cost-profile="+str(pfile)]
                    if force:args+=["--force-producer="+force]
                    row=dict(spec=spec,arm=name,candidate=name,selection=selection,conversion=conversion,path=str(d),
                             source_sha256=sha(raw),profile_sha256=sha(pfile) if pfile else None,status="generating",
                              prepared_metadata_sha256=sha(prepared/"prepared.json"),domain_digest=prepared_meta["domain_digest"],
                              coverage_digest=prepared_meta["coverage_digest"],prepared_domain_equal_to_unit=True if phase=="selected" else None)
                    manifest["rows"].append(row);write(run/(phase+"-manifest.json"),manifest)
                    outcome=invoke(args,base/(name+".log"),allow_documented_directory=d)
                    if outcome["exit_code"]==3:
                        row.update(outcome)
                        row["artifact_hashes"]={str(p.relative_to(d)):sha(p) for p in d.rglob("*") if p.is_file()}
                        write(run/(phase+"-manifest.json"),manifest);continue
                    record=read(d/"record.json")
                    if record.get("status")!="success":raise ValueError("worker success without success record")
                    if record["cost"].get("saturated") is not False:raise ValueError("saturated/unknown model cost is not calibration data")
                    row.update(status="generated",record_sha256=sha(d/"record.json"),
                        assignment_sha256=record["complete_assignment_sha256"],features=features(record) if model!="legacy" else None)
                    write(d/"calibration_spec.json",spec);write(run/(phase+"-manifest.json"),manifest)
        if source_identity()!=identity:raise ValueError("source changed during generation")
        check_tools(tools);check_inputs(inputs_guard);manifest["status"]="generated"
    except Exception as e:
        manifest.update(status="failed",error=type(e).__name__+": "+str(e));raise
    finally:write(run/(phase+"-manifest.json"),manifest)
    return manifest

def compile_exact(directory):
    import triton
    from triton.backends.compiler import GPUTarget
    from backend import emit
    from resource_prediction import predict_resources, resolve_resources
    if any((directory/name).exists() for name in ("kernel.cubin","kernel.ptx","kernel.ttgir","correctness.json")):
        raise FileExistsError("fresh compilation refuses overwrite")
    start=time.perf_counter_ns();execution=read(directory/"execution.json");tt,info=emit(execution)
    from native_static_support import check_static_shared
    check_static_shared(info)  # reject unsupported static allocations before native compilation
    write(directory/"resource-plan-before-native.json",dict(prediction=predict_resources(execution),
          information_stage="execution-export-before-native-compile",created_ns=time.time_ns()))
    (directory/"kernel.ttgir").write_text(tt)
    k=triton.compile(str(directory/"kernel.ttgir"),target=GPUTarget("cuda",90,32),options={"num_warps":4,"enable_fp_fusion":False})
    (directory/"kernel.cubin").write_bytes(k.asm["cubin"]);(directory/"kernel.ptx").write_text(k.asm["ptx"])
    if "llir" in k.asm:(directory/"kernel.llir").write_text(k.asm["llir"])
    write(directory/"execution_config.json",info)
    write(directory/"resource-pressure-ptx-only.json",dict(prediction=predict_resources(execution,k.asm["ptx"]),
          information_stage="PTX-only-pre-RA-information",wallclock_stage="after-combined-Triton-compile",
          created_ns=time.time_ns(),reads_native_resources=False))
    resolved=resolve_resources(execution,k.asm["ptx"],directory/"kernel.cubin",
                               cache_dir=directory.parents[1]/"resource-feedback-cache",
                               expected_cubin_sha256=sha(directory/"kernel.cubin"))
    write(directory/"resolved-compiler-resources.json",resolved)
    phase_manifest=directory.parents[2]/(directory.parents[1].name+"-manifest.json")
    phase_tools=read(phase_manifest)["tools"] if phase_manifest.exists() else freeze_tools()
    write(directory/"compile.json",dict(wall_ns=time.perf_counter_ns()-start,cubin_sha256=sha(directory/"kernel.cubin"),
                                      execution_sha256=sha(directory/"execution.json"),source=source_identity(),tools=phase_tools,
                                      tool_identity_boundary="full native/tool snapshot checked before and after complete validation phase",
                                      cache="normal compiler cache; descriptive compile wall, not cold compilation"))

def parse_sass(text):
    from collections import Counter
    opcodes=Counter()
    for line in text.splitlines():
        match=re.search(r"/\*[0-9a-fA-F]+\*/\s+(?:@!?U?P(?:[0-9]+|T)\s+)?([A-Z][A-Z0-9_.]*)",line)
        if match:opcodes[match.group(1).split(".")[0]]+=1
    if not opcodes:raise ValueError("SASS has no parsed instructions")
    return opcodes

def machine_artifacts(directory):
    """Static disassembly of this raw cubin, not dynamic instruction counts."""
    from collections import Counter
    executable="/usr/local/cuda/bin/cuobjdump"
    for flag,name in (("--dump-sass","kernel.sass"),("--dump-resource-usage","machine-resources.txt")):
        result=subprocess.run([executable,flag,str(directory/"kernel.cubin")],capture_output=True,text=True)
        (directory/name).write_text(result.stdout)
        (directory/(name+".stderr")).write_text(result.stderr)
        if result.returncode:raise RuntimeError("cuobjdump failed "+flag)
    text=(directory/"kernel.sass").read_text()
    opcodes=parse_sass(text)
    result=dict(schema_version=1,cubin_sha256=sha(directory/"kernel.cubin"),cuobjdump_sha256=sha(executable),
                instruction_count=sum(opcodes.values()),opcodes=dict(sorted(opcodes.items())),
                interpretation="static disassembly counts, not dynamic executed instructions or a latency model",
                local_load_instructions=opcodes["LDL"],local_store_instructions=opcodes["STL"],spill_count=None,
                raw_resource_sha256=sha(directory/"machine-resources.txt"),sass_sha256=sha(directory/"kernel.sass"))
    write(directory/"machine-evidence.json",result)
    return result

def execute(directory,load_only=False,safety=False):
    import torch
    from run_joint_ablation import load_exact,attention_inputs,attention_reference,attention_offsets
    from physical_resources import query_physical_resources,compare_logical_resources
    from async_host import physical_abi,validate_launch,TensorMaps
    spec=read(directory/"calibration_spec.json")
    if not load_only:compile_exact(directory)
    elif sha(directory/"kernel.cubin")!=read(directory/"correctness.json")["cubin_sha256"]:raise ValueError("exact accepted cubin changed")
    torch.cuda.set_device(1);torch.empty(1,device="cuda")
    cuda,module,function,binary=load_exact(directory);physical=query_physical_resources(cuda,function)
    from resource_prediction import resolve_resources, confirm_driver_resources
    resolved_path=directory/"resolved-compiler-resources.json"
    resolved=resolve_resources(read(directory/"execution.json"),(directory/"kernel.ptx").read_text(),
        directory/"kernel.cubin",cache_dir=directory.parents[1]/"resource-feedback-cache",
        expected_cubin_sha256=hashlib.sha256(binary).hexdigest())
    driver_confirmation=confirm_driver_resources(resolved,physical,
        loaded_cubin_sha256=hashlib.sha256(binary).hexdigest(),loaded_entry="frisk_execution")
    device=ctypes.c_int();raw_uuid=(ctypes.c_ubyte*16)()
    if cuda.cuCtxGetDevice(ctypes.byref(device)) or cuda.cuDeviceGetUuid(ctypes.byref(raw_uuid),device):raise ValueError("CUDA identity query")
    actual_uuid="GPU-"+str(uuid.UUID(bytes=bytes(raw_uuid)))
    if actual_uuid!=UUID:raise ValueError("unexpected device UUID")
    info=read(directory/"execution_config.json");abi=physical_abi((directory/"kernel.ptx").read_text(),len(info["tma_plans"]))
    attn=spec["case_id"]=="attention"
    if not attn and (len(abi)!=5 or info["copy_path"]!="vector"):raise ValueError("unexpected controlled ABI")
    source=attention_inputs() if attn else [("seed"+str(b),*inputs(spec,b)) for b in range(3)]
    if safety:source=source[:1]
    offsets=attention_offsets(info["copy_path"]) if attn else [(x,x) for x in (1,2,3)]
    records=[]
    for batch,(case_name,q,k,v) in enumerate(source):
        ref=attention_reference(q,k,v) if attn else reference(spec,q,k,v)
        for offset,output_offset in offsets:
            buffers=[]
            for x in (q,k,v):
                buf=torch.full((x.nbytes+272,),165,dtype=torch.uint8,device="cuda")
                buf[offset:offset+x.nbytes].copy_(torch.from_numpy(x.view(np.uint8).reshape(-1)));buffers.append(buf)
            output=torch.full((ref.nbytes+272,),165,dtype=torch.uint8,device="cuda");saved=[x.cpu().numpy().copy() for x in buffers]
            pointers=[x.data_ptr()+offset for x in buffers]+[output.data_ptr()+output_offset];validate_launch(info,pointers)
            maps=TensorMaps(info,pointers) if info["tma_plans"] else None
            args=[ctypes.c_uint64(x) for x in pointers+([d.data_ptr() for d in maps.device] if maps else [])+[0]]
            if len(args)!=len(abi):raise ValueError("physical launch ABI")
            params=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in args])
            def launch():
                status=cuda.cuLaunchKernel(function,1,1,1,128,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),params,None)
                if status:raise RuntimeError("launch "+str(status))
            def check(path):
                torch.cuda.synchronize();raw=output.cpu().numpy();got=raw[output_offset:output_offset+ref.nbytes].copy().view(np.float32).reshape(ref.shape)
                if attn:valid=np.all(np.abs(got-ref)<=.002+.002*np.abs(ref))
                elif spec["tail"] in ("exp1","mixed"):valid=np.all(np.abs(got-ref)<=1e-5+2e-6*np.abs(ref))
                else:valid=np.array_equal(got,ref)
                if not np.all(np.isfinite(got)) or not valid:raise ValueError("independent numerical reference failure")
                if not np.all(raw[:output_offset]==165) or not np.all(raw[output_offset+ref.nbytes:]==165):raise ValueError("output guard")
                if not all(np.array_equal(x.cpu().numpy(),before) for x,before in zip(buffers,saved)):raise ValueError("input bytes/guards changed")
                records.append(dict(case=case_name,input_offset=offset,output_offset=output_offset,path=path,numerics=True,guards=True,inputs_unchanged=True))
            for repeat in range(1 if safety else 3):output.fill_(165);launch();check("ordinary")
            if not safety and (not attn or batch==0):
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):output.fill_(165);launch()
                check("nondefault-stream")
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):
                    for _ in range(3):launch()
                count=ctypes.c_size_t();cuda.cuGraphGetNodes.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)]
                if cuda.cuGraphGetNodes(ctypes.c_void_p(graph.raw_cuda_graph()),None,ctypes.byref(count)) or count.value!=3:raise ValueError("capture node count")
                output.fill_(165);graph.replay();check("capture-3-nodes")
    record=read(directory/"record.json")
    result=dict(cubin_sha256=sha(directory/"kernel.cubin"),gpu_uuid=actual_uuid,physical_resources=physical,
                compiler_resource_confirmation=driver_confirmation,physical_params=abi,records=records,safety_subset=safety,
                resource_comparison=compare_logical_resources(record["output_diagnostics"]["resources"],physical,info))
    result["resource_comparison"]["backend_allocation_components"]={k:info[k] for k in ("static_shared_bytes","shared_root_bytes","scratch_offset","scratch_max_bytes","tma_barrier_bytes","physical_shared_bytes")}
    result["resource_comparison"]["backend_components_are_post_materialization_explanations_not_SSA_prediction_inputs"]=True
    if result["resource_comparison"].get("driver_shared_matches_emitter") is not True:raise ValueError("driver static Shared differs from emitted allocation")
    write(directory/("safety-correctness.json" if safety else "correctness.json"),result)
    return result

def validate(run,phase):
    manifest=read(run/(phase+"-manifest.json"))
    check_inputs(manifest["input_hashes"]);check_tools(manifest["tools"])
    if source_identity()!=manifest["identity"]:raise ValueError("source changed since generation")
    if manifest["status"]!="generated":raise ValueError("fresh generated manifest required")
    accepted_safety={}
    try:
        for row in manifest["rows"]:
            if row["status"]=="documented-failure":
                failure=Path(row["path"])/"failure.json"
                if row.get("artifact_hashes",{}).get("failure.json")!=sha(failure):raise ValueError("retained failure evidence changed")
                outcome=documented_outcome(row.get("exit_code"),failure.parent)
                if any(row.get(k)!=v for k,v in outcome.items()):raise ValueError("retained failure schema changed")
                continue
            if row["status"]!="generated":raise ValueError("unexpected matrix status before validation")
            d=Path(row["path"]);invoke([sys.executable,__file__,"execute",d],d/"compile-numerics.log")
            row["machine_evidence"]=machine_artifacts(d)
            accepted=read(d/"correctness.json");row["cubin_sha256"]=accepted["cubin_sha256"]
            row["physical_resources"]=accepted["physical_resources"];row["resource_comparison"]=accepted["resource_comparison"]
            row["safety"]=[]
            safety_key=(row["cubin_sha256"],sha(d/"execution_config.json"),json.dumps(accepted["physical_params"],sort_keys=True))
            if safety_key in accepted_safety:
                row["safety"]=dict(reused_from=accepted_safety[safety_key],cubin_sha256=safety_key[0],config_sha256=safety_key[1],
                                   rule="exact binary bytes plus complete execution config and physical ABI; current numerical validation still executed")
            for tool in ("memcheck","racecheck","synccheck","initcheck"):
                if isinstance(row["safety"],dict):break
                log=d/("safety-"+tool+".log")
                invoke(["/usr/local/cuda/bin/compute-sanitizer","--tool",tool,"--error-exitcode","91",sys.executable,__file__,"execute",d,"--load-only","--safety"],log)
                row["safety"].append(dict(tool=tool,cubin_sha256=row["cubin_sha256"],log_sha256=sha(log),status="passed"))
            accepted_safety.setdefault(safety_key,str(d))
            row["status"]="validated"
            row["artifact_hashes"]={str(p.relative_to(d)):sha(p) for p in d.rglob("*") if p.is_file()}
            write(run/(phase+"-manifest.json"),manifest)
        check_inputs(manifest["input_hashes"]);check_tools(manifest["tools"])
        if source_identity()!=manifest["identity"]:raise ValueError("source changed during validation")
        manifest["status"]="validated"
    except Exception as e:
        manifest.update(status="failed",error=type(e).__name__+": "+str(e));raise
    finally:write(run/(phase+"-manifest.json"),manifest)

def measurement_groups(run,phase):
    registered={s["key"]:s for s in verify_protocol(run)["specifications"]}
    manifest=read(run/(phase+"-manifest.json"))
    if manifest["status"]!="validated":raise ValueError("exact-cubin numerical and sanitizer validation required")
    if source_identity()!=manifest["identity"]:raise ValueError("source changed since exact-cubin validation")
    check_inputs(manifest["input_hashes"]);check_tools(manifest["tools"])
    rows=[dict(r) for r in manifest["rows"] if r["status"]=="validated" and (phase!="initial" or r["spec"]["split"]=="train")]
    if phase=="selected":
        initial=read(run/"initial-manifest.json")
        if initial["status"]!="validated":raise ValueError("validated candidate set required")
        if source_identity()!=initial["identity"]:raise ValueError("source changed since candidate generation")
        check_inputs(initial["input_hashes"]);check_tools(initial["tools"])
        for r in initial["rows"]:
            if r["status"]=="validated" and r["conversion"]=="C":
                new=dict(r);new["arm"]="candidate-"+r["arm"];rows.append(new)
    groups={}
    for row in rows:
        if registered.get(row["spec"]["key"])!=row["spec"]:raise ValueError("measurement specification differs from preregistration")
        groups.setdefault(row["spec"]["key"],[]).append(row)
    wanted={s["key"] for s in registered.values() if phase!="initial" or s["split"]=="train"}
    if set(groups)!=wanted:raise ValueError("no executable candidates for part of the registered program set")
    if phase=="selected":
        for key,group in groups.items():
            required={"legacy-LI","legacy-J","calibrated-LI","calibrated-J"}
            if not required<={r["arm"] for r in group}:raise ValueError("required final C comparison arm unavailable: "+key)
    for key,group in groups.items():
        if len({r["arm"] for r in group})!=len(group):raise ValueError("duplicate arm")
    return groups

def verify_measurement(run,measurement,phase,expected_profile=None):
    """Bind every accepted sample/analysis to this run, phase and frozen profile."""
    run=Path(run).resolve();measurement=Path(measurement).resolve();verify_protocol(run)
    completion=read(measurement/"completion.json");frozen=read(measurement/"freeze.json")
    if completion.get("status")!="complete" or completion.get("phase")!=phase or frozen.get("phase")!=phase:
        raise ValueError("measurement is incomplete or belongs to another phase")
    if completion.get("run")!=str(run) or frozen.get("run")!=str(run):raise ValueError("measurement belongs to another run")
    manifest_path=run/(phase+"-manifest.json");manifest=read(manifest_path)
    if completion.get("freeze_sha256")!=sha(measurement/"freeze.json"):raise ValueError("measurement freeze changed")
    for value in (completion,frozen):
        if value.get("protocol_sha256")!=sha(run/"protocol.json"):raise ValueError("measurement protocol mismatch")
        if value.get("manifest_sha256")!=sha(manifest_path):raise ValueError("measurement manifest mismatch")
    if manifest.get("status")!="validated" or frozen.get("source")!=manifest.get("identity") or frozen.get("source")!=source_identity():
        raise ValueError("measurement source/generation identity mismatch")
    if frozen.get("tools")!=manifest.get("tools"):raise ValueError("measurement tool snapshot mismatch")
    check_tools(frozen["tools"])
    if frozen.get("input_hashes")!=manifest.get("input_hashes"):raise ValueError("measurement input snapshot mismatch")
    check_inputs(frozen["input_hashes"])
    evidence=completion.get("evidence_hashes")
    if type(evidence) is not dict or not evidence:raise ValueError("missing measurement evidence hashes")
    actual={str(p.relative_to(measurement)) for p in measurement.rglob("*") if p.is_file() and p!=measurement/"completion.json"}
    if set(evidence)!=actual:raise ValueError("measurement evidence file set changed")
    for relative,h in evidence.items():
        target=(measurement/relative).resolve()
        if not target.is_relative_to(measurement) or sha(target)!=h:raise ValueError("measurement evidence changed: "+relative)
    observations=read(measurement/"observations.json")
    registered={s["key"]:s for s in read(run/"protocol.json")["specifications"]}
    for row in observations:
        if registered.get(row["spec"]["key"])!=row["spec"]:raise ValueError("measurement observation specification changed")
        if phase=="initial" and row["spec"]["split"]!="train":raise ValueError("holdout data in training measurement")
    if phase=="selected":
        if expected_profile is None:raise ValueError("selected assessment requires a frozen training profile")
        profile_hash=sha(expected_profile)
        calibrated=[r for r in manifest["rows"] if r["arm"].startswith("calibrated-")]
        if not calibrated or any(r.get("profile_sha256")!=profile_hash for r in calibrated):
            raise ValueError("selected measurement used another profile")
        if profile_hash not in frozen["input_hashes"].values():raise ValueError("training profile absent from measurement freeze")
    return observations

def measure(run,phase,out):
    import torch
    from run_joint_ablation import load_exact,attention_inputs,attention_reference
    from joint_measure import gpu_state,foreign_pids,measured_tensormaps
    from async_host import physical_abi,validate_launch,TensorMaps
    p=verify_protocol(run);groups=measurement_groups(run,phase);out=fresh(out);identity=source_identity();tools=freeze_tools();inputs_guard=read(run/(phase+"-manifest.json"))["input_hashes"];check_inputs(inputs_guard)
    write(out/"freeze.json",dict(run=str(run.resolve()),protocol_sha256=sha(run/"protocol.json"),phase=phase,source=identity,tools=tools,
                                 manifest_sha256=sha(run/(phase+"-manifest.json")),input_hashes=inputs_guard,groups=groups))
    torch.cuda.set_device(1);torch.empty(1,device="cuda")
    if foreign_pids(gpu_state()):raise RuntimeError("foreign GPU workload before timing")
    rng=np.random.default_rng(SEED);summaries=[];observations=[]
    try:
        for key,rows in groups.items():
            entries={};setup=[];spec=rows[0]["spec"];arms=[r["arm"] for r in rows]
            for row in rows:
                d=Path(row["path"])
                for name,h in row["artifact_hashes"].items():
                    if sha(d/name)!=h:raise ValueError("validated artifact changed: "+str(d/name))
                start=time.perf_counter_ns();cuda,module,function,binary=load_exact(d);load_ns=time.perf_counter_ns()-start
                info=read(d/"execution_config.json");abi=physical_abi((d/"kernel.ptx").read_text(),len(info["tma_plans"]))
                phases=[]
                for b in range(10):
                    if spec["case_id"]=="attention":_,q,k,v=attention_inputs()[b];ref=attention_reference(q,k,v)
                    else:q,k,v=inputs(spec,b);ref=reference(spec,q,k,v)
                    io=1+b%3 if spec["abi_stratum"]=="weak-vector" else 128*(b%3);oo=1+b%3
                    start=time.perf_counter_ns();buffers=[]
                    for x in (q,k,v):
                        buf=torch.full((x.nbytes+272,),165,dtype=torch.uint8,device="cuda");buf[io:io+x.nbytes].copy_(torch.from_numpy(x.view(np.uint8).reshape(-1)));buffers.append(buf)
                    output=torch.full((ref.nbytes+272,),165,dtype=torch.uint8,device="cuda");pointers=[x.data_ptr()+io for x in buffers]+[output.data_ptr()+oo];validate_launch(info,pointers)
                    maps=measured_tensormaps(info,pointers) if info["tma_plans"] else None
                    torch.cuda.synchronize()
                    if maps:
                        accepted=TensorMaps(info,pointers);torch.cuda.synchronize()
                        if not all(torch.equal(x,y) for x,y in zip(maps.device,accepted.device)):raise ValueError("TMA encoder mismatch")
                    args=[ctypes.c_uint64(x) for x in pointers+([x.data_ptr() for x in maps.device] if maps else [])+[0]]
                    if len(args)!=len(abi):raise ValueError("physical launch ABI")
                    params=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in args])
                    def launch(cuda=cuda,function=function,params=params):
                        status=cuda.cuLaunchKernel(function,1,1,1,128,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),params,None)
                        if status:raise RuntimeError("cuLaunchKernel "+str(status))
                    content=dict(key=key,batch=b,input_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in (q,k,v)],
                                 reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),input_offset=io,output_offset=oo)
                    phases.append(dict(launch=launch,buffers=buffers,output=output,maps=maps,args=args,params=params,ref=ref,
                                       saved=[x.cpu().numpy().copy() for x in buffers],content=content,
                                       pair_id=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(",",":")).encode()).hexdigest()))
                    setup.append(dict(arm=row["arm"],batch=b,module_load_ns=load_ns,allocation_descriptor_upload_sync_ns=time.perf_counter_ns()-start,
                                      descriptor_encode_ns=maps.setup_ns if maps else [],descriptor_upload_ms=maps.upload_ms if maps else []))
                entries[row["arm"]]=dict(phases=phases,cuda=cuda,module=module,function=function,binary=binary)
            write(out/(key+"-setup.json"),setup)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            def check(ph):
                stream.synchronize();raw=ph["output"].cpu().numpy();oo=ph["content"]["output_offset"];ref=ph["ref"];got=raw[oo:oo+ref.nbytes].copy().view(ref.dtype).reshape(ref.shape)
                exp=spec["tail"] in ("exp1","mixed");attn=spec["case_id"]=="attention"
                valid=np.all(np.isfinite(got)) and (np.all(np.abs(got-ref)<=((.002+.002*np.abs(ref)) if attn else (1e-5+2e-6*np.abs(ref)))) if (exp or attn) else np.array_equal(got,ref))
                if not valid or not np.all(raw[:oo]==165) or not np.all(raw[oo+ref.nbytes:]==165):raise ValueError("measurement numerical/guard preflight")
                if not all(np.array_equal(x.cpu().numpy(),saved) for x,saved in zip(ph["buffers"],ph["saved"])):raise ValueError("input guard preflight")
            def events(fn,count=5):
                result=[]
                for _ in range(count):
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();fn();end.record();end.synchronize();result.append(begin.elapsed_time(end))
                return result
            pilots=[];preflight=[]
            with torch.cuda.stream(stream):
                for arm in arms:
                    for b,ph in enumerate(entries[arm]["phases"]):
                        ph["output"].fill_(165);ph["launch"]();check(ph);graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream):ph["launch"]()
                        ph["output"].fill_(165);graph.replay();check(ph);preflight.append(dict(arm=arm,batch=b,pair_id=ph["pair_id"],ordinary=True,capture=True,guards=True,inputs_unchanged=True))
                    launch=entries[arm]["phases"][0]["launch"]
                    for _ in range(100):launch()
                    stream.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        for _ in range(100):launch()
                    pilots.append(dict(arm=arm,graph_ms=events(graph.replay),nodes=100))
            count=max(1,int(p["measurement"]["target_graph_ms"]/max(np.median(x["graph_ms"])/100 for x in pilots)))
            write(out/(key+"-preflight.json"),preflight);write(out/(key+"-pilot.json"),dict(nodes=count,records=pilots))
            records=[]
            for b in range(10):
                before=gpu_state()
                if foreign_pids(before):raise RuntimeError("foreign GPU workload before paired batch; no batch sampled")
                order=rng.permutation(arms).tolist();batch=[]
                with torch.cuda.stream(stream):
                    for arm in order:
                        ph=entries[arm]["phases"][b];launch=ph["launch"]
                        for _ in range(100):launch()
                        stream.synchronize();start=time.perf_counter_ns();graph=torch.cuda.CUDAGraph(keep_graph=True)
                        with torch.cuda.graph(graph,stream=stream):
                            for _ in range(count):launch()
                        stream.synchronize();capture_ns=time.perf_counter_ns()-start
                        cuda=entries[arm]["cuda"];cuda.cuGraphGetNodes.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)];actual=ctypes.c_size_t()
                        if cuda.cuGraphGetNodes(ctypes.c_void_p(graph.raw_cuda_graph()),None,ctypes.byref(actual)) or actual.value!=count:raise ValueError("actual graph nodes mismatch")
                        start=time.perf_counter_ns();graph.instantiate();instantiate_ns=time.perf_counter_ns()-start;elapsed=events(graph.replay)
                        batch.append(dict(batch=b,arm=arm,pair_id=ph["pair_id"],pair_content=ph["content"],order=order,valid=True,
                                          launch_count=count,actual_graph_nodes=actual.value,graph_elapsed_ms=elapsed,samples_ms=[x/count for x in elapsed],
                                          capture_sync_ns=capture_ns,instantiate_ns=instantiate_ns,cubin_sha256=next(r["cubin_sha256"] for r in rows if r["arm"]==arm)))
                after=gpu_state();foreign=foreign_pids(before)+foreign_pids(after)
                for r in batch:r.update(valid=not foreign,foreign_pids=foreign,state_before=before,state_after=after)
                records+=batch;write(out/(key+"-raw.json"),records)
                print(json.dumps(dict(key=key,batch=b,foreign=foreign)),flush=True)
            analysis=paired_analysis(records,arms);summaries.append(dict(key=key,spec=spec,analysis=analysis));write(out/"analysis.json",summaries)
            if not analysis["formal"]:raise RuntimeError("insufficient valid paired batches; retained, no replacement")
            for row in rows:
                observations.append(dict(spec=spec,candidate=row["arm"],arm=row["arm"],features=row["features"],median_us=analysis["arms"][row["arm"]]["median_us"],
                                         cubin_sha256=row["cubin_sha256"],measurement_sha256=sha(out/(key+"-raw.json")),physical_resources=row["physical_resources"],
                                         resource_comparison=row["resource_comparison"],record_path=str(Path(row["path"])/"record.json"),valid=True))
            write(out/"observations.json",observations)
        if source_identity()!=identity:raise ValueError("source changed during formal timing")
        check_tools(tools);check_inputs(inputs_guard)
        evidence={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()}
        write(out/"completion.json",dict(status="complete",run=str(run.resolve()),phase=phase,protocol_sha256=sha(run/"protocol.json"),
              manifest_sha256=sha(run/(phase+"-manifest.json")),freeze_sha256=sha(out/"freeze.json"),
              evidence_hashes=evidence,source_unchanged=True,tools_unchanged=True,groups=len(groups)))
    except Exception as e:
        write(out/"failure.json",dict(status="failed",error=type(e).__name__+": "+str(e),partial_samples_retained=True));raise
    return out

def fit(run,measurement,out):
    p=verify_protocol(run);rows=verify_measurement(run,measurement,"initial");out=fresh(out)
    profile,result=fit_profile(rows,p)
    write(out/"profile.json",profile);write(out/"fit.json",result)
    write(out/"training-assessment.json",assess(rows,profile,result))
    write(out/"resource-fit.json",resource_calibration(rows))
    write(out/"inputs.json",dict(run=str(run.resolve()),training_measurement=str(measurement.resolve()),protocol_sha256=sha(run/"protocol.json"),observations_sha256=sha(measurement/"observations.json"),
                                measurement_completion_sha256=sha(measurement/"completion.json"),holdout_used=False,
                                profile_sha256=sha(out/"profile.json"),fit_sha256=sha(out/"fit.json"),resource_fit_sha256=sha(out/"resource-fit.json")))
    return out/"profile.json"

def assessment(run,measurement,fit_directory,out):
    verify_protocol(run)
    frozen=read(fit_directory/"inputs.json")
    if frozen["protocol_sha256"]!=sha(run/"protocol.json") or frozen["profile_sha256"]!=sha(fit_directory/"profile.json"):
        raise ValueError("frozen training profile changed")
    if frozen["fit_sha256"]!=sha(fit_directory/"fit.json") or frozen["resource_fit_sha256"]!=sha(fit_directory/"resource-fit.json"):
        raise ValueError("frozen training fit or resource mapping changed")
    if frozen.get("run")!=str(run.resolve()):raise ValueError("training fit belongs to another run")
    training=Path(frozen["training_measurement"])
    verify_measurement(run,training,"initial")
    if frozen["observations_sha256"]!=sha(training/"observations.json") or frozen["measurement_completion_sha256"]!=sha(training/"completion.json"):
        raise ValueError("training fit points to another measurement")
    rows=verify_measurement(run,measurement,"selected",fit_directory/"profile.json")
    result=assess_selected(rows,read(fit_directory/"profile.json"),read(fit_directory/"fit.json"),read(fit_directory/"resource-fit.json")["models"])
    result["identity"]=dict(training_inputs_sha256=sha(fit_directory/"inputs.json"),measurement_observations_sha256=sha(measurement/"observations.json"),
                            profile_sha256=sha(fit_directory/"profile.json"),holdout_refit=False,protocol_sha256=sha(run/"protocol.json"))
    if out.exists():raise FileExistsError("assessment refuses overwrite")
    write(out,result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("init","generate","validate","execute","measure","fit","assess"));parser.add_argument("directory",type=Path)
    parser.add_argument("--phase",choices=("initial","selected"),default="initial");parser.add_argument("--profile",type=Path)
    parser.add_argument("--out",type=Path);parser.add_argument("--measurement",type=Path);parser.add_argument("--fit-directory",type=Path)
    parser.add_argument("--load-only",action="store_true");parser.add_argument("--safety",action="store_true");a=parser.parse_args()
    if a.command=="init":return init(a.directory)
    if a.command=="generate":return generate(a.directory,a.phase,a.profile)
    if a.command=="validate":return validate(a.directory,a.phase)
    if a.command=="execute":return execute(a.directory,a.load_only,a.safety)
    if a.command=="measure":return measure(a.directory,a.phase,a.out)
    if a.command=="fit":return fit(a.directory,a.measurement,a.out)
    if a.command=="assess":return assessment(a.directory,a.measurement,a.fit_directory,a.out)
if __name__=="__main__":main()
