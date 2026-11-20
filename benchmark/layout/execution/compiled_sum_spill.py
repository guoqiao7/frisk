"""Compile-time evidence for non-repeating local accesses in, fixed-I/O SM90 Gluon Sum placements."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
import subprocess
import tempfile


@dataclass(frozen=True)
class LocalTraffic:
    load_words: int
    store_words: int
    loads: int
    stores: int


def local_traffic(listing, entry):
    """Count 32-bit words per lane in the complete, named cuobjdump function."""
    functions = re.split(r"(?m)^\s*Function\s*:\s*(\S+)\s*$", listing)
    bodies = [body for name, body in zip(functions[1::2], functions[2::2]) if name == entry]
    if len(bodies) != 1:
        raise ValueError(f"expected one disassembly for {entry}")
    count = {"LDL": 0, "STL": 0}
    words = count.copy()
    pc_next = 0
    exited = False
    back_edges = []
    local_pcs = []
    barriers = {}
    for line in bodies[0].splitlines():
        pc = re.match(r"^\s*/\*([0-9a-fA-F]+)\*/\s+", line)
        if not pc:
            continue
        if int(pc[1], 16) != pc_next:
            raise ValueError("non-contiguous SASS listing")
        pc_next += 16
        ins = re.match(r"(?:(@!?[A-Za-z0-9]+)\s+)?([A-Z0-9_]+)((?:\.[A-Z0-9_]+)*)\b", line[pc.end():])
        if not ins:
            raise ValueError("undecodable instruction")
        pred, op, suffix = ins.groups()
        address = int(pc[1], 16)
        if op == "EXIT" and not pred:
            exited = True
        if op == "BRA":
            dest = re.search(r"\b0x([0-9a-fA-F]+)\s*;", line[pc.end()+ins.end():])
            if not dest:
                raise ValueError("unresolved branch target")
            target = int(dest[1], 16)
            if target <= address:
                back_edges.append((target, address))
        if op == "BSSY":
            setup = re.search(r"\b(B[0-9]+),\s*0x([0-9a-fA-F]+)\s*;", line[pc.end()+ins.end():])
            if not setup:
                raise ValueError("unresolved convergence target")
            barriers[setup[1]] = int(setup[2], 16)
        if op == "BREAK":
            use = re.search(r"\b(B[0-9]+)\s*;", line[pc.end()+ins.end():])
            if not use or barriers.get(use[1], -1) <= address:
                raise ValueError("non-forward convergence break")
        if op in ("BRX", "JMP", "JMX", "CALL", "RET", "BRXU", "JMXU", "BRK", "CONT"):
            raise ValueError("unbounded or indirect control flow in spill evidence")
        if op not in count:
            continue
        local_pcs.append(address)
        modifiers = set(suffix.split(".")) - {""}
        if modifiers - {"LU", "64", "128", "32"}:
            raise ValueError(f"unsupported local-memory instruction {op}{suffix}")
        width = 4 if "128" in modifiers else 2 if "64" in modifiers else 1
        count[op] += 1
        words[op] += width
    if not exited:
        raise ValueError("incomplete SASS listing")
    # Forward control flow gives a static full-mask upper estimate. A back
    # edge spanning local traffic could multiply the charge by an unknown trip count.
    if any(lo <= addr <= hi for lo, hi in back_edges for addr in local_pcs):
        raise ValueError("repeated local traffic needs a loop trip count")
    return LocalTraffic(words["LDL"], words["STL"], count["LDL"], count["STL"])


@dataclass
class CompiledSumArm:
    image: object
    digest: str
    traffic: LocalTraffic
    listing: str


def compile_sum_arms(layouts, compile_layout, *, cuobjdump="cuobjdump"):
    """Compile without timing, then inspect the very cubins available for dispatch."""
    arms = {}
    for layout in layouts:
        if layout in arms:
            raise ValueError("duplicate layout")
        image = compile_layout(layout)
        meta = image.metadata
        if (meta.target.backend, meta.target.arch, meta.num_warps, meta.num_stages, meta.num_ctas) != ("cuda", 90, 4, 1, 1):
            raise ValueError("expected sm90, one CTA, four warps and one pipeline stage")
        cubin = image.asm["cubin"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arm.cubin"
            path.write_bytes(cubin)
            listing = subprocess.check_output([cuobjdump, "-sass", str(path)], text=True)
        arms[layout] = CompiledSumArm(image, sha256(cubin).hexdigest(),
                                      local_traffic(listing, image.name), listing)
    if not arms:
        raise ValueError("empty compiled candidate set")
    return arms


def evidence_attr(arms, *, read, write, axis, read_storage, write_storage):
    """Return an MLIR dictionary for the fixed load -> sum -> store kernel."""
    for alias in [*arms, read, write]:
        if not re.fullmatch(r"#[A-Za-z_][A-Za-z0-9_]*", alias):
            raise ValueError("expected an MLIR encoding alias")
    if axis not in (0, 1):
        raise ValueError("expected a rank-two reduction axis")
    records = [f'{{layout={k}, cubin="{a.digest}", load_words={a.traffic.load_words}:i64, '
               f'store_words={a.traffic.store_words}:i64}}' for k,a in arms.items()]
    return (f'{{backend="gluon34", threads=128:i64, stages=1:i64, axis={axis}:i64, '
            f'read_storage={read_storage}, write_storage={write_storage}, read={read}, write={write}, arms=[{", ".join(records)}]}}')


def selected_image(arms, ir):
    """Select the compiled binary named by the Frisk solver's audited receipt."""
    digests = re.findall(r'compiled_spill_cubin = "([0-9a-f]{64})"', ir)
    if len(digests) != 1:
        raise ValueError("missing or ambiguous compiled spill receipt")
    for arm in arms.values():
        if arm.digest == digests[0] and sha256(arm.image.asm["cubin"]).hexdigest() == arm.digest:
            return arm.image
    raise ValueError("selected cubin is missing or stale")
