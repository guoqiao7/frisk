"""Frozen complete-program calibration/holdout specifications, separate from legacy fixtures."""
import hashlib
import json
import numpy as np

SEED = 20261031

def specifications():
    def item(key, split, case, shape, tail="none"):
        return dict(key=key, split=split, case_id=case, shape=list(shape),
                    tail=tail, abi_stratum="weak-vector", dtype="float32", threads=128)
    rows = [
        item("train-no-conflict-64x16", "train", "no-conflict", (64,16)),
        item("train-conflict-64x16", "train", "conflict", (64,16)),
        item("train-remat-64x16", "train", "remat", (64,16)),
        item("train-mul-32x16", "train", "no-conflict", (32,16), "mul8"),
        item("train-div-32x16", "train", "no-conflict", (32,16), "div6"),
        item("train-exp-32x16", "train", "no-conflict", (32,16), "exp1"),
    ]
    for abi in ("weak-vector","strong-cpasync","strong-tma"):
        rows.append(dict(key="train-attention-"+abi, split="train", case_id="attention",
                         shape=[64,16], tail="none", abi_stratum=abi, dtype="float16", threads=128))
    rows += [
        item("holdout-diamond-32x32", "holdout", "diamond", (32,32)),
        item("holdout-remat-128x16", "holdout", "remat", (128,16)),
        item("holdout-mixed-64x32", "holdout", "no-conflict", (64,32), "mixed"),
    ]
    return rows

def encoding(shape, swap=False):
    rb, cb = [int(np.log2(x)) for x in shape]
    if any(2**b != x for b,x in zip((rb,cb),shape)) or rb+cb < 7:
        raise ValueError("power-of-two shape and at least 128 elements required")
    register_bits=rb+cb-7
    basis=list(range(rb+cb))
    if swap:
        basis[register_bits],basis[register_bits+5]=basis[register_bits+5],basis[register_bits]
    # Logical row bits precede column bits; physical bits enumerate flat row-major.
    output_bits=list(range(cb,cb+rb))+list(range(cb))
    matrix=["["+",".join("true" if b==ob else "false" for b in basis)+"]" for ob in output_bits]
    return ('#frisk.distributed<map=#frisk.bit_linear<inputs=["register","lane","warp"],'
            'input_bits=['+f'{register_bits},5,2'+'],outputs=["dim0","dim1"],output_bits=['+
            f'{rb},{cb}'+'],matrix=dense<['+','.join(matrix)+']> : tensor<'+
            f'{rb+cb}x{rb+cb}'+'xi1>>,topology=['+f'{2**register_bits},32,4,1,1'+'],replication=1>')

def inputs(spec, batch=0):
    n=int(np.prod(spec["shape"]))
    seed=SEED+batch
    rng=np.random.default_rng(seed)
    return tuple(rng.uniform(lo,hi,n).astype(np.float32) for lo,hi in [(0.25,1.25),(1.5,2.5),(-.75,-.25)])

def reference(spec,q,k,v):
    mul=lambda a,b:np.multiply(a,b,dtype=np.float32)
    case=spec["case_id"]
    p=q if case=="diamond" else mul(q,q if case=="remat" else k)
    x=p if case=="no-conflict" else np.subtract(mul(p,p),mul(mul(p,p),k if case=="conflict" else v),dtype=np.float32)
    tail=spec["tail"]
    if tail=="mul8":
        for _ in range(8):x=mul(x,q)
    elif tail=="div6":
        for _ in range(6):x=np.divide(x,k,dtype=np.float32)
    elif tail=="exp1":x=np.exp(x,dtype=np.float32)
    elif tail=="mixed":x=np.subtract(mul(np.divide(np.exp(x,dtype=np.float32),k,dtype=np.float32),q),v,dtype=np.float32)
    elif tail!="none":raise ValueError("unknown tail")
    return x

def fixture(spec):
    if spec["case_id"]=="attention":raise ValueError("Attention uses checked repository source")
    from joint_fixture import fixture as legacy
    text=legacy(spec["case_id"])
    start=text.index("#A="); end=text.index("#F=",start)
    text=text[:start]+"#A="+encoding(spec["shape"])+"\n#B="+encoding(spec["shape"],True)+"\n"+text[end:]
    rows,cols=spec["shape"];n=rows*cols
    text=text.replace("64x16",f"{rows}x{cols}").replace("input_extents=[64,16]",f"input_extents=[{rows},{cols}]").replace("16*i+j",f"{cols}*i+j").replace("output_extents=[4096,8]",f"output_extents=[{4*n},8]")
    store=text.index("frisk.tile_store")
    line=text[store:text.index("\n",store)]
    source=line.split()[1].split(",")[0]
    tail=spec["tail"];ops=[];ty=f"tensor<{rows}x{cols}xf32>"
    def append(op,operands):
        name="%cal"+str(len(ops));ops.append(f"{name}={op} "+",".join(operands)+f" : {ty}\n");return name
    if tail=="mul8":
        for _ in range(8):source=append("arith.mulf",[source,"%q0"])
    elif tail=="div6":
        for _ in range(6):source=append("arith.divf",[source,"%k0"])
    elif tail=="exp1":source=append("math.exp",[source])
    elif tail=="mixed":
        source=append("math.exp",[source]);source=append("arith.divf",[source,"%k0"]);source=append("arith.mulf",[source,"%q0"]);source=append("arith.subf",[source,"%v0"])
    elif tail!="none":raise ValueError("unknown tail")
    return text[:store]+"".join(ops)+text[store:].replace(line,"frisk.tile_store "+source+",%ov : "+ty+f",memref<{rows}x{cols}xf32,1>",1)

def protocol():
    return dict(schema_version=1,stage="cost-resource-calibration",seed=SEED,
                specifications=specifications(),split_rule="entire-program specification; all variants/candidates of a spec share split",
                fit=dict(method="nonnegative active-set least squares with ridge",ridge=1e-6,max_iterations=20000,tolerance=1e-10,
                         feature_scaling="training column maximum",target="paired-batch median graph-amortized kernel microseconds",
                         ticks_per_microsecond=1000000,intercept="nonnegative; analysis only, excluded from selection profile",
                         inactive_coefficient_ticks=1),
                measurement=dict(batches=10,bootstrap_resamples=10000,warmup=100,pilot_nodes=100,target_graph_ms=50,replays=5,
                                 foreign_pid_rule="invalidate entire paired batch, retain and never replace",
                                 confidence="paired percentile 95%; per-comparison, no multiplicity adjustment"),
                limits="static SM90 one-CTA programs; Attention all ABI strata are training/functional baseline cases, not holdout",
                holdout_rule="never fit, select hyperparameters, or change protocol from holdout results")

def protocol_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
