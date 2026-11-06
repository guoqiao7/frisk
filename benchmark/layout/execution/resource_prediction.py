"""Resource layers with explicit provenance: planning, pre-RA hint, compiler fact.

No empirical register hint is an allocation limit. Native feedback is read from
an actual cubin before any GPU execution; it is never labeled a prediction.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


class ResourceEvidenceError(RuntimeError):
    pass


def _sha(data): return hashlib.sha256(data).hexdigest()

def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def parse_compiler_resource_usage(text, entry="frisk_execution"):
    if type(entry) is not str or not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$.]*", entry):
        raise ValueError("invalid exact kernel entry name")
    found=[];current=None
    for line in text.splitlines():
        function=re.fullmatch(r"\s*Function\s+([^\s:]+)\s*:\s*", line)
        if function:
            current={} if function.group(1)==entry else None
            if current is not None: found.append(current)
            continue
        if current is None: continue
        for field,value in re.findall(r"\b(REG|STACK|SHARED|LOCAL):([0-9]+)(?=\s|$)",line):
            if field in current: raise ResourceEvidenceError("duplicate resource field "+field)
            current[field]=int(value)
    if len(found)!=1 or set(found[0])!={"REG","STACK","SHARED","LOCAL"}:
        raise ResourceEvidenceError("one complete resource record for the exact entry is required")
    value=found[0]
    if any(v>2**63-1 for v in value.values()):raise ResourceEvidenceError("resource integer overflow")
    return dict(registers_per_thread=value["REG"], raw_shared_bytes=value["SHARED"],
                stack_bytes_per_thread=value["STACK"], raw_local_bytes=value["LOCAL"],
                local_bytes_per_thread=0 if value["LOCAL"]==value["STACK"]==0 else None,
                needs_driver_local_query=bool(value["LOCAL"] or value["STACK"]),
                spill_count=None)


def _check_resources(resources):
    required={"registers_per_thread","raw_shared_bytes","stack_bytes_per_thread","raw_local_bytes",
              "local_bytes_per_thread","needs_driver_local_query","spill_count"}
    if type(resources) is not dict or set(resources)!=required:
        raise ResourceEvidenceError("malformed cached resource record")
    for name in ("registers_per_thread","raw_shared_bytes","stack_bytes_per_thread","raw_local_bytes"):
        if type(resources[name]) is not int or not 0<=resources[name]<=2**63-1:
            raise ResourceEvidenceError("invalid cached resource integer")
    nonzero=bool(resources["stack_bytes_per_thread"] or resources["raw_local_bytes"])
    if (type(resources["needs_driver_local_query"]) is not bool or
        resources["needs_driver_local_query"]!=nonzero or resources["spill_count"] is not None or
        (resources["local_bytes_per_thread"] is not None if nonzero else
         type(resources["local_bytes_per_thread"]) is not int or resources["local_bytes_per_thread"]!=0)):
        raise ResourceEvidenceError("invalid cached local-memory interpretation")


def compiler_exact_resources(cubin_path, entry="frisk_execution", *,
                             tool_path="/usr/local/cuda/bin/cuobjdump", cache_dir=None,
                             expected_cubin_sha256=None):
    """CPU-only postcompile feedback, keyed by exact image/tool/entry/schema.

    Stack and LOCAL remain separate. For nonzero local allocations the driver
    must confirm its own LOCAL_SIZE_BYTES interpretation; no sum is guessed.
    A corrupt or stale cache raises rather than returning a plausible zero.
    """
    cubin=Path(cubin_path).resolve();binary=cubin.read_bytes();cubin_sha=_sha(binary)
    if binary[:4]!=b"\x7fELF":raise ResourceEvidenceError("expected a compiled ELF cubin")
    if expected_cubin_sha256 is not None and cubin_sha!=expected_cubin_sha256:
        raise ResourceEvidenceError("compiled image identity mismatch")
    parse_compiler_resource_usage("Resource usage:\n Function "+entry+":\n REG:0 STACK:0 SHARED:0 LOCAL:0\n",entry)
    tool=Path(tool_path).resolve();tool_sha=_sha(tool.read_bytes())
    identity=dict(schema_version=2,cubin_sha256=cubin_sha,entry=entry,tool_sha256=tool_sha)
    def stable_original_inputs():
        if _sha(cubin.read_bytes())!=cubin_sha or Path(tool_path).resolve()!=tool or _sha(tool.read_bytes())!=tool_sha:
            raise ResourceEvidenceError("image/tool changed during native resource query")
    key=_sha(_canonical(identity));cache=Path(cache_dir)/ (key+".json") if cache_dir is not None else None
    if cache is not None and cache.exists():
        try:record=json.loads(cache.read_text())
        except Exception as e:raise ResourceEvidenceError("unreadable resource cache") from e
        if (type(record) is not dict or set(record)!={"identity","payload","payload_sha256"} or
            record["identity"]!=identity or type(record["payload"]) is not dict or
            _sha(_canonical(record["payload"]))!=record["payload_sha256"]):
            raise ResourceEvidenceError("resource cache integrity/identity mismatch")
        payload=record["payload"]
        if (payload.get("kind")!="compiler-exact-postcompile" or payload.get("identity")!=identity or
            type(payload.get("raw_resource_usage")) is not str or
            _sha(payload["raw_resource_usage"].encode())!=payload.get("raw_resource_sha256") or
            parse_compiler_resource_usage(payload["raw_resource_usage"],entry)!=payload.get("resources")):
            raise ResourceEvidenceError("resource cache payload mismatch")
        _check_resources(payload["resources"])
        stable_original_inputs()
        return dict(payload,cache_key=key,cache_hit=True)
    # Query an owned byte snapshot, never a mutable caller path carrying the key.
    stable_original_inputs()
    with tempfile.TemporaryDirectory(prefix="frisk-resource-image-") as temporary:
        snapshot=Path(temporary)/"image.cubin";snapshot.write_bytes(binary)
        command=[str(tool),"--dump-resource-usage",str(snapshot)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        if _sha(snapshot.read_bytes())!=cubin_sha:
            raise ResourceEvidenceError("native tool modified its immutable image snapshot")
        stable_original_inputs()
    if result.returncode:raise ResourceEvidenceError("native resource query failed: "+result.stderr[-1000:])
    resources=parse_compiler_resource_usage(result.stdout,entry);_check_resources(resources)
    payload=dict(kind="compiler-exact-postcompile",estimation_stage="after-native-compilation-before-GPU",
                 identity=identity,resources=resources,command=command,
                 raw_resource_usage=result.stdout,raw_resource_sha256=_sha(result.stdout.encode()),
                 shared_interpretation="raw cuobjdump SHARED; may include device-reserved bytes, not user allocation",
                 source_cubin_path=str(cubin),input_snapshot_sha256=cubin_sha,
                 local_interpretation="nonzero STACK/LOCAL requires independent driver query; not a spill count")
    if cache is not None:
        cache.parent.mkdir(parents=True,exist_ok=True)
        record=dict(identity=identity,payload=payload,payload_sha256=_sha(_canonical(payload)))
        with tempfile.NamedTemporaryFile(mode="w",dir=cache.parent,prefix="resource-",suffix=".tmp",delete=False) as f:
            json.dump(record,f,sort_keys=True);temporary=Path(f.name)
        temporary.replace(cache)
    stable_original_inputs()
    return dict(payload,cache_key=key,cache_hit=False)


def predict_resources(execution, ptx_text=None, model=None):
    """Pre-native layers. This function never reads a cubin or driver result."""
    from shared_allocation import plan_shared_allocation
    shared=plan_shared_allocation(execution)
    register=dict(status="unavailable",reason="PTX not supplied",requires_compilation=True)
    if ptx_text is not None:
        from register_pressure import analyze_ptx_pressure
        register=analyze_ptx_pressure(ptx_text)
        register=dict(register,requires_compilation=True)
    if model is not None:
        # A profile schema and estimator must explicitly declare applicability;
        # accepting arbitrary coefficients would repeat the old SSA fit error.
        raise ResourceEvidenceError("physical register profile interface is not configured")
    return dict(schema_version=2,kind="resource-plan-and-pre-RA-hint",shared=shared,
                registers=register,physical_registers=None,
                authoritative_resource_requires_compilation=True)


def resolve_resources(execution, ptx_text, cubin_path, *, cache_dir=None,
                      tool_path="/usr/local/cuda/bin/cuobjdump", expected_cubin_sha256=None):
    """Combine compiler registers and structural Shared without conflating units.

    cuobjdump raw SHARED can include reserved device memory (1024B on the
    measured H100). Its mapping to user bytes is confirmed with device/driver
    facts, not silently subtracted here or learned from candidate measurements.
    """
    prediction=predict_resources(execution,ptx_text)
    native=compiler_exact_resources(cubin_path,tool_path=tool_path,cache_dir=cache_dir,
                                    expected_cubin_sha256=expected_cubin_sha256)
    actual=native["resources"];planned=prediction["shared"]["physical_shared_bytes"]
    return dict(schema_version=2,kind="resolved-compiler-resources",prediction=prediction,
                compiler_exact=native,registers_per_thread=actual["registers_per_thread"],
                shared_bytes=planned,raw_native_shared_bytes=actual["raw_shared_bytes"],
                local_bytes=actual["local_bytes_per_thread"],
                resource_sources=dict(registers="compiled-cubin",shared="pre-native-structural-allocation",
                                      local="zero-native-allocation-or-driver-required"),
                registers_are_predictions=False,shared_is_structural_prediction=True,
                shared_native_interpretation="requires-target-driver-confirmation",
                driver_confirmation_required=True)


def confirm_driver_resources(resolved, driver, *, loaded_cubin_sha256=None, loaded_entry=None):
    """Bind compiler facts to the exact bytes/entry actually loaded by CUDA."""
    identity=resolved.get("compiler_exact",{}).get("identity",{})
    if (type(loaded_cubin_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}",loaded_cubin_sha256) or
        loaded_cubin_sha256!=identity.get("cubin_sha256") or loaded_entry!=identity.get("entry")):
        raise ResourceEvidenceError("loaded kernel image/entry identity is required and must match compiler feedback")
    for key in ("registers_per_thread","shared_bytes","local_bytes"):
        if type(driver.get(key)) is not int or driver[key]<0:
            raise ResourceEvidenceError("invalid driver resource field "+key)
    if resolved.get("kind")!="resolved-compiler-resources":
        raise ResourceEvidenceError("a resolved compiler resource record is required")
    for key in ("registers_per_thread","shared_bytes"):
        if resolved[key]!=driver[key]:raise ResourceEvidenceError("compiler/plan/driver mismatch: "+key)
    if resolved["local_bytes"] is not None and resolved["local_bytes"]!=driver["local_bytes"]:
        raise ResourceEvidenceError("compiler/driver local memory mismatch")
    reserved=driver.get("device_limits",{}).get("reserved_shared_bytes_per_block")
    if type(reserved) is not int or reserved<0:
        raise ResourceEvidenceError("explicit target reserved Shared property required")
    raw=resolved["raw_native_shared_bytes"];user=driver["shared_bytes"]
    if not ((raw==user==0) or (raw>0 and raw==user+reserved)):
        raise ResourceEvidenceError("native raw Shared / user allocation / target reserve are inconsistent")
    return dict(status="confirmed",registers_per_thread=driver["registers_per_thread"],
                shared_bytes=user,raw_native_shared_bytes=raw,reserved_shared_bytes_per_block=reserved,
                local_bytes=driver["local_bytes"],local_source="CUDA-driver-LOCAL_SIZE_BYTES",spill_count=None,
                loaded_cubin_sha256=loaded_cubin_sha256,entry=loaded_entry,
                physical_counts_are_predictions=False)
