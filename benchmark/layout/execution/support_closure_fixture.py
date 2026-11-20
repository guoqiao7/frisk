"""Declared current execution domain and independent numerical cases.
The finite acceptance matrix is not the backend capability predicate.
"""
import hashlib
import itertools
import json
import re
import numpy as np
from calibration_fixture import fixture as calibration_program
from calibration_fixture import inputs as previous_inputs, reference as f32_reference

SEED = 20261102
PROFILE_RELATIVE = "guoqiao/optimize_plan/cost_resource_calibration_artifacts/run-calibration-r3-20261031T105856Z/fit/profile.json"
PROFILE_SHA256 = "6b4947af82c2748fc8db75268ddb621fc45852d238b63deee12cf17c6cde7388"
GPU_UUID = "GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618"


def specifications():
    rows = []
    for name, shape in [("diamond",(32,32)),("remat",(128,16)),("mixed",(128,32)),("conflict",(256,8)),
                        ("diamond",(32,16)),("remat",(32,16)),("mixed",(32,16)),("conflict",(64,8))]:
        rows.append(dict(key=name+"-"+str(shape[0])+"x"+str(shape[1]),
            case_id="no-conflict" if name=="mixed" else name, shape=list(shape),
            tail="mixed" if name=="mixed" else "none", abi_stratum="weak-vector",
            dtype="float32",threads=128,graph_name=name))
    for abi in ("weak-vector","strong-cpasync","strong-tma"):
        rows.append(dict(key="attention-"+abi,case_id="attention",shape=[64,16],tail="none",
                         abi_stratum=abi,dtype="float16",threads=128,graph_name="attention"))
    return rows


def source(spec, root):
    if spec["case_id"] == "attention":
        suffix={"weak-vector":"fixed","strong-cpasync":"cp-async","strong-tma":"tma"}[spec["abi_stratum"]]
        return (root/("test/Transforms/attention-layout-"+suffix+"-abi.mlir")).read_text()
    text=calibration_program(spec)
    # Remove unused controlled-domain metadata. No A/B candidate injection occurs.
    text=text.replace(",joint.A=#A,joint.B=#B","")
    return "\n".join(line for line in text.splitlines() if not re.match(r"#(?:A|B)=",line))+"\n"


def numerical_cases(spec):
    if spec["case_id"]=="attention":
        from run_joint_ablation import attention_inputs
        return attention_inputs()
    result=[("seed"+str(i),*previous_inputs(spec,i)) for i in range(3)]
    n=int(np.prod(spec["shape"]))
    coord=np.linspace(-.75,.75,n,dtype=np.float32)
    alternate=np.where(np.arange(n)%2,np.float32(-.5),np.float32(.5)).astype(np.float32)
    zero=np.zeros(n,np.float32); one=np.ones(n,np.float32)
    result += [("zeroQ",zero,one,coord),("constant",one*.5,one*1.5,one*-.25),
               ("signedcoords",coord,np.add(one*1.5,coord*.25,dtype=np.float32),coord[::-1].copy()),
               ("cancellation",alternate,np.add(one*1.5,alternate*.5,dtype=np.float32),alternate*.5)]
    return result


def reference(spec,q,k,v):
    if spec["case_id"]=="attention":
        from run_joint_ablation import attention_reference
        return attention_reference(q,k,v)
    return f32_reference(spec,q,k,v)


def check_numerics(spec,got,expected):
    if got.shape!=expected.shape or got.dtype!=np.float32 or expected.dtype!=np.float32:
        raise ValueError("numerical dtype/shape mismatch")
    if not np.isfinite(got).all() or not np.isfinite(expected).all():
        return dict(passed=False,finite=False,mode="finite-required",max_abs_error=None)
    error=np.abs(got-expected)
    if spec["case_id"]=="attention":
        mode="abs<=.002+.002*abs(reference)"; valid=np.all(error<=.002+.002*np.abs(expected))
    elif spec["tail"]=="mixed":
        mode="abs<=1e-5+2e-6*abs(reference)"; valid=np.all(error<=1e-5+2e-6*np.abs(expected))
    else:
        mode="float32-bit-exact-including-signed-zero"
        valid=np.array_equal(got.view(np.uint32),expected.view(np.uint32))
    return dict(passed=bool(valid),finite=True,mode=mode,max_abs_error=float(error.max(initial=0)),
                coordinate=list(map(int,np.unravel_index(int(error.argmax()),error.shape))))


def offsets(spec,copy_path):
    from run_joint_ablation import attention_offsets
    if spec["case_id"]=="attention":return attention_offsets(copy_path)
    if copy_path!="vector":raise ValueError("f32 acceptance programs require weak synchronous ABI")
    return [(x,x) for x in (1,2,3)]


def safety_input_identity(spec,info,abi):
    # Reuse requires the exact input/spec protocol as well as native bytes/config.
    cases=[dict(name=name,inputs=[hashlib.sha256(x.tobytes()).hexdigest() for x in (q,k,v)])
           for name,q,k,v in numerical_cases(spec)[:1]]
    return dict(spec=spec,cases=cases,offsets=offsets(spec,info["copy_path"]),physical_abi=abi,
                repeats=1,guard_byte=165,threads=128,grid=[1,1,1],default_stream=True)


def protocol():
    return dict(schema_version=1,stage="current-support-domain-gpu-closure",stage_tag=SEED,
        specifications=specifications(),models=["legacy","calibrated"],
        profile=dict(path=PROFILE_RELATIVE,sha256=PROFILE_SHA256,format="calibrated-v2",semantics="calibrated-v3"),
        execution=dict(gpu_uuid=GPU_UUID,device_ordinal=1,grid=[1,1,1],threads=128,
                       entry="frisk_execution",num_warps=4,enable_fp_fusion=False,output_aliasing="O disjoint from Q/K/V; read-only Q/K/V may overlap; positive nonoverflowing uint64 ranges"),
        selection="normal frisk-layout-pipeline; unchanged default budgets; no controlled A/B injection or producer forcing",
        numerics=dict(random_seed_base=20261031,random_case_indices=[0,1,2],attention_seed_indices=list(range(10)),f32_cases="3 previous fixed seeds + zeroQ/constant/signedcoords/cancellation",
            attention_cases="existing 15 independent reference cases",ordinary_repeats=3,
            auxiliary="all f32 cases and Attention seed0: nondefault stream and exact 3-node capture",
            safety="first fixed seed, all registered offsets, once under each sanitizer",
            ordinary="uint32 bit equality including signed zero",mixed="finite abs(error)<=1e-5+2e-6*abs(reference)",
            attention="finite abs(error)<=.002+.002*abs(reference)"),
        safety=dict(tools=["memcheck","racecheck","synccheck","initcheck"],
            environment="named load-only safety workers permit exactly one frozen sanitizer runtime-directory LD_LIBRARY_PATH prefix, followed by the unchanged original path; record raw/effective/canonical; every payload hash is frozen",
            reuse="exact cubin bytes + complete emission config + physical ABI + identical spec and safety input protocol"),
        outcome="all 22 predeclared natural configurations must fully execute; no silent filtering/substitution",
        limits="checked backend subset only; single CTA; no multi-CTA/tail/overlap claim",
        no_kernel_timing=True,no_performance_claim=True)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
