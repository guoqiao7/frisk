"""Current natural-selection GPU closure, reusing the checked Frisk backend.
No CUDA context or subprocess work at import time. Fresh run directories only.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import support_closure_fixture as fixture

class ClosureFailure(RuntimeError):
    def __init__(self,stage,code,reason):
        super().__init__(reason);self.stage=stage;self.code=code


ROOT=Path(__file__).resolve().parents[3]
OPT=ROOT/"build-m6-release/bin/frisk-opt"


def sha(path):
    from integrity_session import active
    session=active()
    return session.sha(path) if session is not None and session.enrolled(path) else hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
def fresh(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False);return path


def invoke(command,log,timeout=600):
    from run_joint_ablation import command as existing_command
    code=existing_command([str(x) for x in command],Path(log),timeout_seconds=timeout)
    return dict(exit_code=code,log=str(log),command_metadata=str(log)+".command.json")


def execution_environment():
    result={key:os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES","CUDA_MODULE_LOADING","TRITON_ALWAYS_COMPILE","TRITON_CACHE_DIR","LD_LIBRARY_PATH")}
    # torch.cuda._lazy_init itself applies this documented effective default.
    # Treat unset and its eventual LAZY value identically without creating CUDA.
    if result["CUDA_MODULE_LOADING"] is None:result["CUDA_MODULE_LOADING"]="LAZY"
    return result


def frozen_tools(opt):
    from cost_calibration import freeze_tools
    tools=freeze_tools();tools["hashes"][str(Path(opt).resolve())]=sha(opt)
    q=subprocess.run(["ldd",str(opt)],capture_output=True,text=True,check=True)
    for line in q.stdout.splitlines():
        fields=line.split()
        for word in fields:
            if word.startswith("/") and Path(word).is_file():tools["hashes"][str(Path(word).resolve())]=sha(word)
    runtime=(Path("/usr/local/cuda/bin/compute-sanitizer").resolve().parent.parent/"compute-sanitizer").resolve()
    payloads=sorted(x for x in runtime.iterdir() if x.is_file())
    if not (runtime/"compute-sanitizer").is_file() or not any(x.suffix==".so" for x in payloads):raise ValueError("sanitizer runtime payload directory is unavailable")
    tools["sanitizer_runtime"]=dict(directory=str(runtime),payload_hashes={str(x.resolve()):sha(x) for x in payloads})
    tools["hashes"].update(tools["sanitizer_runtime"]["payload_hashes"])
    tools["environment"]=execution_environment()
    tools["environment_defaults"]={"CUDA_MODULE_LOADING":"LAZY when unset, matching torch.cuda._lazy_init"}
    return tools


def validate_runtime_regions(pointers,lengths):
    """The declared ABI is out-of-place; read-only inputs may overlap."""
    maximum=(1<<64)-1
    if len(pointers)!=4 or len(lengths)!=4:raise ValueError("four logical pointer ranges required")
    ranges=[]
    for pointer,length in zip(pointers,lengths):
        if type(pointer) is not int or type(length) is not int or pointer<=0 or length<=0 or pointer>maximum or length>maximum-pointer:
            raise ValueError("invalid or overflowing uint64 logical pointer range")
        ranges.append((pointer,pointer+length))
    begin,end=ranges[3]
    if any(begin<other_end and other_begin<end for other_begin,other_end in ranges[:3]):
        raise ValueError("output must not overlap read-only input ranges")
    return dict(runtime_abi_checked=True,pointer_range_bits=64,output_disjoint_from_inputs=True,read_only_input_overlap_allowed=True)


def verify_environment(expected,*,sanitizer_prefix=None):
    raw={key:os.environ.get(key) for key in expected}
    effective=execution_environment();canonical=dict(effective)
    if sanitizer_prefix is not None:
        if not Path(sanitizer_prefix).is_absolute():raise ValueError("sanitizer prefix must be absolute")
        original=expected.get("LD_LIBRARY_PATH")
        allowed=sanitizer_prefix+(":"+original if original is not None else "")
        if effective.get("LD_LIBRARY_PATH")!=allowed:raise ValueError("unexpected sanitizer LD_LIBRARY_PATH injection")
        canonical["LD_LIBRARY_PATH"]=original
    if canonical!=expected:raise ValueError("execution environment changed outside the frozen sanitizer prefix")
    return dict(raw=raw,effective=effective,canonical=canonical,sanitizer_prefix=sanitizer_prefix)


def parse_gpu_state(device,processes):
    if device.returncode or device.stderr.strip() or fixture.GPU_UUID not in device.stdout:
        raise ValueError("invalid target GPU identity query")
    if processes.returncode or processes.stderr.strip():raise ValueError("invalid GPU process query")
    pids=[]
    for line in processes.stdout.splitlines():
        if not line.strip():continue
        fields=[x.strip() for x in line.split(",")]
        if len(fields)!=3 or fields[0]!=fixture.GPU_UUID or not fields[1].isdigit():
            raise ValueError("malformed target GPU process row")
        if int(fields[1])!=os.getpid():pids.append(int(fields[1]))
    return dict(device=device.stdout.strip(),foreign_pids=pids,empty_process_stdout_valid=True)


def gpu_state():
    device=subprocess.run(["nvidia-smi","-i",fixture.GPU_UUID,"--query-gpu=uuid,name,driver_version","--format=csv,noheader"],capture_output=True,text=True)
    processes=subprocess.run(["nvidia-smi","-i",fixture.GPU_UUID,"--query-compute-apps=gpu_uuid,pid,process_name","--format=csv,noheader"],capture_output=True,text=True)
    return parse_gpu_state(device,processes)


def init(run,opt=OPT):
    from cost_calibration import source_identity
    run=fresh(run);protocol=fixture.protocol();profile=ROOT/fixture.PROFILE_RELATIVE
    if sha(profile)!=fixture.PROFILE_SHA256:raise ValueError("frozen coefficient file changed")
    write(run/"protocol.json",protocol)
    registration=dict(created_ns=time.time_ns(),protocol_sha256=sha(run/"protocol.json"),
        source=source_identity(),tools=frozen_tools(opt),optimizer=str(Path(opt).resolve()),profile_sha256=sha(profile))
    write(run/"registration.json",registration)
    write(run/"manifest.json",dict(schema_version=1,status="registered",registration_sha256=sha(run/"registration.json"),rows=[]))
    return run


def verify_run(run,*,sanitizer=False):
    from cost_calibration import source_identity,check_tools
    from integrity_session import active, verified_sha
    immutable_sha = verified_sha if active() is not None else sha
    run=Path(run);manifest=read(run/"manifest.json");registration=read(run/"registration.json")
    if immutable_sha(run/"registration.json")!=manifest["registration_sha256"]:raise ValueError("registration changed")
    if immutable_sha(run/"protocol.json")!=registration["protocol_sha256"]:raise ValueError("protocol changed")
    if "immutable_inputs" in registration:
        expected=registration["immutable_inputs"]
        if type(expected) is not dict or not expected:raise ValueError("registered immutable inputs must be a nonempty map")
        for name,digest in expected.items():
            path=Path(name)
            if not path.is_absolute() or path.is_symlink() or not path.is_file() or type(digest) is not str or len(digest)!=64 or immutable_sha(path)!=digest:
                raise ValueError("registered immutable input changed: "+str(name))
    elif immutable_sha(ROOT/fixture.PROFILE_RELATIVE)!=registration["profile_sha256"]:raise ValueError("frozen coefficient file changed")
    if source_identity()!=registration["source"]:raise ValueError("source changed since registration")
    if active() is None:check_tools(registration["tools"])
    else:
        for name,digest in registration["tools"]["hashes"].items():
            verified_sha(name,digest)
    prefix=registration["tools"]["sanitizer_runtime"]["directory"] if sanitizer else None
    verify_environment(registration["tools"]["environment"],sanitizer_prefix=prefix)
    return manifest,registration


def row_for(directory,manifest):
    rows=[x for x in manifest["rows"] if Path(x["path"]).resolve()==Path(directory).resolve()]
    if len(rows)!=1:raise ValueError("configuration missing or duplicated in run manifest")
    return rows[0]


def verify_files(directory,expected):
    from integrity_session import active, verified_sha
    immutable_sha = verified_sha if active() is not None else sha
    for name,digest in expected.items():
        path=Path(directory)/name
        if path.is_symlink() or not path.is_file() or immutable_sha(path)!=digest:
            raise ValueError("execution artifact identity changed: "+name)


def generate(run):
    from execution_support import assess_file
    manifest,registration=verify_run(run)
    if manifest["status"]!="registered":raise ValueError("fresh registered run required")
    manifest["status"]="generating";write(run/"manifest.json",manifest)
    for spec in read(run/"protocol.json")["specifications"]:
        for model in ("legacy","calibrated"):
            directory=fresh(run/"cases"/(spec["key"]+"-"+model))
            row=dict(spec=spec,model=model,path=str(directory.resolve()),status="generating",phases=[])
            manifest["rows"].append(row);write(run/"manifest.json",manifest)
            (directory/"input.mlir").write_text(fixture.source(spec,ROOT));write(directory/"calibration_spec.json",spec)
            write(directory/"configuration.json",dict(spec=spec,model=model,entry="frisk_execution",grid=[1,1,1],threads=128))
            options="ir-diagnostics=true stats-file="+str(directory/"stats.json")+" trace-file="+str(directory/"trace.json")
            if model=="calibrated":options+=" cost-profile="+str(ROOT/fixture.PROFILE_RELATIVE)
            command=[registration["optimizer"],directory/"input.mlir","--frisk-layout-pipeline="+options,"-verify-each","-o",directory/"selected.mlir"]
            outcome=invoke(command,directory/"inference.log");row["phases"].append(dict(stage="inference",**outcome))
            if outcome["exit_code"]:
                row.update(status="failed",stage="inference",code="inference-failed",reason="see frozen stats/command diagnostics")
                write(run/"manifest.json",manifest);continue
            stats=read(directory/"stats.json");model_id=stats.get("cost_model_id","")
            if (model=="legacy" and model_id!="sm90-layout-lex9-legacy-v2") or (model=="calibrated" and not model_id.startswith("sm90-layout-calibrated-v3-")):
                raise ValueError("unexpected complete model identity")
            row["model_id"]=model_id
            row["selected_ir_sha256"]=sha(directory/"selected.mlir")
            diagnostics=stats.get("actual_ir_diagnostics")
            if not isinstance(diagnostics,dict) or not isinstance(diagnostics.get("resources"),dict):raise ValueError("actual post-materialization IR resources required")
            write(directory/"record.json",dict(output_diagnostics=diagnostics,source="normal pipeline actual_ir_diagnostics"))
            outcome=invoke([registration["optimizer"],directory/"selected.mlir","--frisk-export-execution=output="+str(directory/"execution.json"),"-verify-each","-o","/dev/null"],directory/"export.log")
            row["phases"].append(dict(stage="checked-export",**outcome))
            if outcome["exit_code"]:
                row.update(status="failed",stage="checked-export",code="export-rejected",reason="see exact export diagnostics")
                write(run/"manifest.json",manifest);continue
            assessment=assess_file(directory/"execution.json",include_emission=True)
            tt=assessment.pop("ttgir",None);info=assessment.pop("execution_config",None);write(directory/"support.json",assessment)
            if not assessment.get("supported"):
                row.update(status="failed",stage=assessment.get("stage"),code=assessment.get("code"),reason=assessment.get("reason"))
                write(run/"manifest.json",manifest);continue
            if not isinstance(tt,str) or not isinstance(info,dict):raise ValueError("supported assessment missing emission")
            if sha(directory/"selected.mlir")!=row["selected_ir_sha256"]:raise ValueError("emission changed selected Frisk IR")
            packet=read(directory/"execution.json")
            consumption=dict(execution_packet_schema_version=packet.get("schema_version"),
                export_representation="compact-symbolic" if packet.get("schema_version")==2 else "enumerated",
                selected_ir_sha256=row["selected_ir_sha256"],selected_ir_unchanged=True,
                layout_decision_origin="normal Frisk pipeline selected IR; backend consumes mappings without layout reselection",
                demand_lowering=info.get("demand_lowering"),claim="backend consumption and checked emission, GPU acceptance recorded separately")
            write(directory/"backend-consumption.json",consumption)
            row["backend_consumption"]=consumption
            (directory/"emitted.ttgir").write_text(tt);write(directory/"emitted-config.json",info)
            names=("input.mlir","selected.mlir","stats.json","trace.json","execution.json","support.json","emitted.ttgir","emitted-config.json","calibration_spec.json","configuration.json","record.json","backend-consumption.json")
            guard={name:sha(directory/name) for name in names};write(directory/"generation-inputs.json",guard)
            row.update(status="generated",generation_inputs_sha256=sha(directory/"generation-inputs.json"))
            write(run/"manifest.json",manifest)
    verify_run(run)
    manifest["status"]="generated" if len(manifest["rows"])==22 and all(x["status"]=="generated" for x in manifest["rows"]) else "generation-failed"
    write(run/"manifest.json",manifest)
    if manifest["status"]!="generated":raise RuntimeError("predeclared support matrix did not fully generate")
    return manifest


def activate_target_device():
    """Select and verify the intended device before Triton can initialize CUDA."""
    import torch
    torch.cuda.set_device(1);torch.empty(1,device="cuda")
    cuda=ctypes.CDLL("libcuda.so.1");device=ctypes.c_int();raw_uuid=(ctypes.c_ubyte*16)()
    if cuda.cuCtxGetDevice(ctypes.byref(device)) or cuda.cuDeviceGetUuid(ctypes.byref(raw_uuid),device):
        raise ClosureFailure("device-selection","gpu-identity-query-failed","CUDA identity query failed before native compilation")
    actual="GPU-"+str(uuid.UUID(bytes=bytes(raw_uuid)))
    if actual!=fixture.GPU_UUID:raise ClosureFailure("device-selection","wrong-gpu","unexpected target GPU before native compilation")
    return dict(gpu_uuid=actual,device_ordinal=device.value,selected_before_native_compilation=True)


def execute(directory,load_only=False,safety=False,safety_tool=None,*,numerical_fixture=None,compiler_provider=None):
    # Explicit provider permits separately preregistered numerical suites.
    # The default closure contract and device/profile identity remain unchanged.
    numerical = fixture if numerical_fixture is None else numerical_fixture
    from cost_calibration import compile_exact
    from run_joint_ablation import load_exact
    from physical_resources import query_physical_resources,compare_logical_resources
    from resource_prediction import resolve_resources,confirm_driver_resources
    from async_host import physical_abi,validate_launch,TensorMaps
    if safety and safety_tool not in ("memcheck","racecheck","synccheck","initcheck"):raise ValueError("named sanitizer tool required for safety worker")
    if safety and not load_only:raise ValueError("safety worker requires accepted load-only image")
    directory=Path(directory);run=directory.parents[1];manifest,registration=verify_run(run,sanitizer=safety);row=row_for(directory,manifest)
    if sha(directory/"generation-inputs.json")!=row.get("generation_inputs_sha256"):raise ValueError("generation identity changed")
    verify_files(directory,read(directory/"generation-inputs.json"))
    if load_only:
        if row.get("image_identity_sha256")!=sha(directory/"image-identity.json") or row.get("correctness_sha256")!=sha(directory/"correctness.json"):
            raise ValueError("load-only requires exact accepted image and correctness identity")
        verify_files(directory,read(directory/"image-identity.json"))
    target_identity=activate_target_device()
    if load_only:
        if read(directory/"target-device-before-native.json")!=target_identity:raise ValueError("accepted target device identity changed")
    else:
        write(directory/"target-device-before-native.json",target_identity)
        try:
            if compiler_provider is None:compile_exact(directory)
            else:compiler_provider(directory,expected_source=registration["source"],expected_tools=registration["tools"])
        except Exception as error:raise ClosureFailure("native-compile","native-compilation-failed",str(error)) from error
        if (directory/"kernel.ttgir").read_text()!=(directory/"emitted.ttgir").read_text() or read(directory/"execution_config.json")!=read(directory/"emitted-config.json"):
            raise ValueError("checked emission differs from compiled input")
        image_files=("execution.json","kernel.ttgir","kernel.ptx","kernel.cubin","execution_config.json","generation-inputs.json","target-device-before-native.json")
        if compiler_provider is not None:image_files+=("compile-cache-receipt.json",)
        write(directory/"image-identity.json",{name:sha(directory/name) for name in image_files})
    import torch
    try:cuda,module,function,binary=load_exact(directory)
    except Exception as error:raise ClosureFailure("native-load","module-load-failed",str(error)) from error
    try:
        if hashlib.sha256(binary).hexdigest()!=read(directory/"image-identity.json")["kernel.cubin"]:raise ValueError("loaded bytes mismatch")
        device=ctypes.c_int();raw_uuid=(ctypes.c_ubyte*16)()
        if cuda.cuCtxGetDevice(ctypes.byref(device)) or cuda.cuDeviceGetUuid(ctypes.byref(raw_uuid),device):raise ValueError("CUDA identity query failed")
        actual_uuid="GPU-"+str(uuid.UUID(bytes=bytes(raw_uuid)))
        if actual_uuid!=fixture.GPU_UUID:raise ValueError("unexpected loaded GPU UUID")
        info=read(directory/"execution_config.json");spec=read(directory/"calibration_spec.json")
        abi=physical_abi((directory/"kernel.ptx").read_text(),len(info["tma_plans"]))
        physical=query_physical_resources(cuda,function)
        resolved=resolve_resources(read(directory/"execution.json"),(directory/"kernel.ptx").read_text(),directory/"kernel.cubin",cache_dir=run/"resource-feedback-cache",expected_cubin_sha256=hashlib.sha256(binary).hexdigest())
        confirmation=confirm_driver_resources(resolved,physical,loaded_cubin_sha256=hashlib.sha256(binary).hexdigest(),loaded_entry="frisk_execution")
        comparison=compare_logical_resources(read(directory/"record.json")["output_diagnostics"]["resources"],physical,info)
        if comparison.get("driver_shared_matches_emitter") is not True or not physical["residency"]["launchable_by_resources"]:
            raise ValueError("loaded physical resources do not match executable allocation")
        records=[];input_records=[];host=[]
        cases=numerical.numerical_cases(spec)
        if safety:cases=cases[:1]
        for case_name,q,k,v in cases:
            ref=numerical.reference(spec,q,k,v)
            for offset,output_offset in numerical.offsets(spec,info["copy_path"]):
                buffers=[]
                for value in (q,k,v):
                    buffer=torch.full((value.nbytes+272,),165,dtype=torch.uint8,device="cuda")
                    buffer[offset:offset+value.nbytes].copy_(torch.from_numpy(value.view(np.uint8).reshape(-1)));buffers.append(buffer)
                output=torch.full((ref.nbytes+272,),165,dtype=torch.uint8,device="cuda");saved=[x.cpu().numpy().copy() for x in buffers]
                pointers=[x.data_ptr()+offset for x in buffers]+[output.data_ptr()+output_offset];validate_launch(info,pointers)
                maps=TensorMaps(info,pointers) if info["tma_plans"] else None
                args=[ctypes.c_uint64(x) for x in pointers+([d.data_ptr() for d in maps.device] if maps else [])+[0]]
                if len(args)!=len(abi):raise ValueError("physical launch ABI mismatch")
                params=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in args])
                host.append(dict(case=case_name,input_offset=offset,output_offset=output_offset,descriptor_setup_ns=maps.setup_ns if maps else []))
                input_records.append(dict(case=case_name,input_offset=offset,output_offset=output_offset,input_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in (q,k,v)],reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest()))
                def launch():
                    validate_runtime_regions(pointers,[q.nbytes,k.nbytes,v.nbytes,ref.nbytes])
                    status=cuda.cuLaunchKernel(function,1,1,1,128,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),params,None)
                    if status:raise ClosureFailure("gpu-launch","launch-failed","CUDA launch status "+str(status))
                def check(path):
                    torch.cuda.synchronize();raw=output.cpu().numpy();got=raw[output_offset:output_offset+ref.nbytes].copy().view(np.float32).reshape(ref.shape)
                    result=numerical.check_numerics(spec,got,ref)
                    guards=bool(np.all(raw[:output_offset]==165) and np.all(raw[output_offset+ref.nbytes:]==165))
                    unchanged=all(np.array_equal(x.cpu().numpy(),before) for x,before in zip(buffers,saved))
                    rec=dict(case=case_name,input_offset=offset,output_offset=output_offset,path=path,guards=guards,inputs_unchanged=unchanged,runtime_abi_checked=True,**result)
                    records.append(rec)
                    if not result["passed"] or not guards or not unchanged:
                        np.save(directory/"failure-got.npy",got);np.save(directory/"failure-reference.npy",ref)
                        write(directory/"numerical-failure.json",rec);raise ClosureFailure("gpu-numerics","numerical-or-guard-failure","independent numerical or byte guard verification failed")
                for repeat in range(1 if safety else 3):output.fill_(165);launch();check("ordinary-"+str(repeat))
                if not safety and (spec["case_id"]!="attention" or case_name=="seed0"):
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):output.fill_(165);launch()
                    check("nondefault-stream")
                    graph=torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph,stream=stream):
                        for _ in range(3):launch()
                    count=ctypes.c_size_t();cuda.cuGraphGetNodes.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)]
                    if cuda.cuGraphGetNodes(ctypes.c_void_p(graph.raw_cuda_graph()),None,ctypes.byref(count)) or count.value!=3:raise ValueError("capture node count differs from three launches")
                    output.fill_(165);graph.replay();check("capture-3-nodes")
        result=dict(schema_version=1,status="passed",cubin_sha256=hashlib.sha256(binary).hexdigest(),image_identity_sha256=sha(directory/"image-identity.json"),
            gpu_uuid=actual_uuid,physical_params=abi,physical_resources=physical,compiler_resource_confirmation=confirmation,
            resource_comparison=comparison,spill_count=None,spill_count_is_not_inferred_from_local_bytes=True,safety_subset=safety,runtime_abi_checked=True,output_input_aliasing="disjoint-required",inputs=input_records,records=records,host_setup=host,
            safety_input_identity=numerical.safety_input_identity(spec,info,abi),
            execution_environment=verify_environment(registration["tools"]["environment"],sanitizer_prefix=registration["tools"]["sanitizer_runtime"]["directory"] if safety else None),no_kernel_timing=True)
        write(directory/(("safety-"+safety_tool+"-correctness.json") if safety else "correctness.json"),result)
        verify_files(directory,read(directory/"image-identity.json"));verify_files(directory,read(directory/"generation-inputs.json"));verify_run(run,sanitizer=safety)
        return result
    finally:
        torch.cuda.synchronize();cuda.cuModuleUnload.argtypes=[ctypes.c_void_p]
        if cuda.cuModuleUnload(module):raise RuntimeError("CUDA module unload failed")


def validate(run):
    manifest,registration=verify_run(run)
    if manifest["status"]!="generated":raise ValueError("fresh generated matrix required")
    before=gpu_state();write(run/"gpu-before.json",before)
    if before["foreign_pids"]:raise RuntimeError("target GPU has foreign process; no validation started")
    manifest["status"]="validating";write(run/"manifest.json",manifest);accepted={}
    try:
        for row in manifest["rows"]:
            directory=Path(row["path"])
            verify_run(run)
            state=gpu_state();write(directory/"gpu-before-configuration.json",state)
            if state["foreign_pids"]:raise RuntimeError("foreign process before configuration; no worker started")
            outcome=invoke([sys.executable,__file__,"execute",directory],directory/"compile-numerics.log")
            row["phases"].append(dict(stage="native-and-numerics",**outcome))
            if outcome["exit_code"]:
                row["failure"]=read(directory/"worker-failure.json") if (directory/"worker-failure.json").exists() else dict(stage="worker-process",code="worker-crash-or-timeout",reason="see command metadata")
                row["status"]="failed";raise RuntimeError("native compilation or correctness worker failed: "+str(directory))
            from cost_calibration import machine_artifacts
            row["machine_evidence"]=machine_artifacts(directory)
            correctness=read(directory/"correctness.json");row.update(image_identity_sha256=sha(directory/"image-identity.json"),correctness_sha256=sha(directory/"correctness.json"),cubin_sha256=correctness["cubin_sha256"],physical_resources=correctness["physical_resources"])
            write(run/"manifest.json",manifest)
            key=safety_key(directory,correctness)
            if key in accepted:
                row["safety"]=dict(status="exact-reuse",from_path=accepted[key],identity=key)
            else:
                row["safety"]=[]
                for tool in read(run/"protocol.json")["safety"]["tools"]:
                    outcome=invoke(["/usr/local/cuda/bin/compute-sanitizer","--tool",tool,"--error-exitcode","91",sys.executable,__file__,"execute",directory,"--load-only","--safety","--safety-tool",tool],directory/("safety-"+tool+".log"))
                    row["safety"].append(dict(tool=tool,identity=key,**outcome))
                    if outcome["exit_code"]:
                        row["status"]="failed";row["failure"]=dict(stage="sanitizer",code="sanitizer-failed",tool=tool,reason="see exact process output")
                        raise RuntimeError("sanitizer failed: "+tool+" "+str(directory))
                    row["safety"][-1]["correctness_sha256"]=sha(directory/("safety-"+tool+"-correctness.json"))
                accepted[key]=str(directory)
            row["status"]="validated";row["artifact_hashes"]={str(x.relative_to(directory)):sha(x) for x in directory.rglob("*") if x.is_file()}
            write(run/"manifest.json",manifest)
        verify_run(run);after=gpu_state();write(run/"gpu-after.json",after)
        if after["foreign_pids"]:raise RuntimeError("target GPU acquired foreign process during validation")
        manifest["status"]="validated"
    except Exception as error:
        manifest.update(status="failed",error=type(error).__name__+": "+str(error));raise
    finally:write(run/"manifest.json",manifest)
    return manifest


def verify_accepted_row(directory,row,*,sanitizer=False):
    directory=Path(directory)
    if row.get("status")!="validated":raise ValueError("nonvalidated configuration")
    verify_files(directory,row["artifact_hashes"])
    for name,key in (("generation-inputs.json","generation_inputs_sha256"),("image-identity.json","image_identity_sha256"),("correctness.json","correctness_sha256")):
        if sha(directory/name)!=row.get(key):raise ValueError("accepted identity changed: "+name)
    verify_files(directory,read(directory/"generation-inputs.json"))
    verify_files(directory,read(directory/"image-identity.json"))
    config=read(directory/"configuration.json")
    if config.get("spec")!=row.get("spec") or config.get("model")!=row.get("model") or read(directory/"calibration_spec.json")!=row.get("spec"):
        raise ValueError("manifest configuration differs from checked execution specification")
    correctness=read(directory/"correctness.json")
    if correctness.get("status")!="passed" or correctness.get("cubin_sha256")!=sha(directory/"kernel.cubin") or correctness.get("gpu_uuid")!=fixture.GPU_UUID:
        raise ValueError("accepted execution/cubin/GPU mismatch")
    if correctness.get("image_identity_sha256")!=sha(directory/"image-identity.json"):
        raise ValueError("correctness references another execution image")
    if row.get('validation_execution_contract')=='frisk-five-process-validation-batch-v1' or isinstance(row.get('safety'),dict) and row['safety'].get('status')=='verified-batch-safety-v1':
        from validation_batch import verify_row_safety
        verify_row_safety(directory,row,sanitizer=sanitizer)
    return correctness


def safety_key(directory,correctness):
    return fixture.digest(dict(cubin=correctness["cubin_sha256"],config=read(Path(directory)/"execution_config.json"),physical_abi=correctness["physical_params"],input_protocol=correctness["safety_input_identity"]))


def verify_declared_rows(rows,protocol):
    expected={(spec["key"],model):spec for spec in protocol["specifications"] for model in protocol["models"]}
    seen=set()
    for row in rows:
        key=(row["spec"]["key"],row["model"])
        if key in seen or key not in expected or row["spec"]!=expected[key]:raise ValueError("matrix configuration missing, duplicated or substituted")
        seen.add(key)
    if seen!=set(expected):raise ValueError("matrix does not cover the complete preregistered domain")


def report(run):
    manifest,registration=verify_run(run)
    if manifest["status"]!="validated":raise ValueError("fully validated matrix required")
    rows=manifest["rows"];verify_declared_rows(rows,read(run/"protocol.json"))
    if len(rows)!=22 or any(x["status"]!="validated" for x in rows):raise ValueError("all 22 predeclared configurations required")
    prior={}
    for row in rows:
        directory=Path(row["path"])
        if directory.is_symlink() or directory.resolve().parent!=(run/"cases").resolve():raise ValueError("configuration path is outside run cases")
        correctness=verify_accepted_row(directory,row);key=safety_key(directory,correctness)
        safety=row["safety"]
        if isinstance(safety,dict):
            if safety.get("status")!="exact-reuse" or safety.get("identity")!=key or safety.get("from_path")!=prior.get(key):
                raise ValueError("safety reuse source or exact identity mismatch")
        else:
            if len(safety)!=4 or {x["tool"] for x in safety}!=set(fixture.protocol()["safety"]["tools"]):raise ValueError("all four sanitizer processes required")
            for outcome in safety:
                if outcome.get("exit_code")!=0 or outcome.get("identity")!=key:raise ValueError("unaccepted sanitizer outcome")
                if sha(directory/("safety-"+outcome["tool"]+"-correctness.json"))!=outcome.get("correctness_sha256"):
                    raise ValueError("sanitizer result identity changed")
        prior.setdefault(key,str(directory))
    summary=dict(schema_version=1,status="PASS_CURRENT_DECLARED_GPU_CLOSURE",rows=len(rows),specifications=len(read(run/"protocol.json")["specifications"]),
        numerical_records=sum(len(read(Path(x["path"])/"correctness.json")["records"]) for x in rows),
        sanitizer_processes=sum(len(x["safety"]) for x in rows if isinstance(x["safety"],list)),
        exact_safety_reused_configurations=sum(isinstance(x["safety"],dict) for x in rows),
        source_digest=registration["source"]["source_digest"],tools=registration["tools"],no_kernel_timing=True,no_performance_claim=True,
        protocol_sha256=registration["protocol_sha256"],registration_sha256=sha(run/"registration.json"))
    write(run/"summary.json",summary);return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("init","generate","validate","execute","report","run"));parser.add_argument("directory",type=Path)
    parser.add_argument("--opt",type=Path,default=OPT);parser.add_argument("--load-only",action="store_true");parser.add_argument("--safety",action="store_true");parser.add_argument("--safety-tool",choices=("memcheck","racecheck","synccheck","initcheck"));args=parser.parse_args()
    try:
        if args.command=="init":result=str(init(args.directory,args.opt))
        elif args.command=="generate":result=generate(args.directory)
        elif args.command=="validate":result=validate(args.directory)
        elif args.command=="execute":result=execute(args.directory,args.load_only,args.safety,args.safety_tool)
        elif args.command=="report":result=report(args.directory)
        else:init(args.directory,args.opt);generate(args.directory);validate(args.directory);result=report(args.directory)
        print(json.dumps(dict(status="success",command=args.command,directory=str(args.directory),result=result if args.command in ("report","init") else None)),flush=True)
    except Exception as error:
        if args.command=="execute" and args.directory.is_dir():
            failure=dict(status="failed",stage=getattr(error,"stage","execution-integrity"),code=getattr(error,"code","execution-failed"),reason=type(error).__name__+": "+str(error))
            write(args.directory/"worker-failure.json",failure)
        print(json.dumps(dict(status="failed",command=args.command,directory=str(args.directory),error=type(error).__name__+": "+str(error))),file=sys.stderr,flush=True)
        raise

if __name__=="__main__":main()
