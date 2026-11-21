"""Score verified whole-graph normalization layout alternatives on SM90 Gluon.

prepare(materialized_path) compiles that graph using the caller's fixed launch
contract and returns its dispatchable compiled image. It must implement the
graph exactly or raise. No timing or counters are consulted during selection.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import re
import subprocess
import tempfile

from compiled_sum_spill import compile_sum_arms
from compiled_global_access import UnsupportedGlobalAccess, global_access, physical_choice


def striped_encoding(rows, columns, width, threads=128):
    """A warp reads adjacent vectors of width FP32 values in a row-major tile."""
    if any(v < 1 or v & (v - 1) for v in (rows, columns, threads, width)):
        raise ValueError("expected power-of-two dimensions and vector width")
    if threads != 128 or width not in (1, 2, 4) or width > columns:
        raise ValueError("unsupported normalization execution geometry")
    slots = rows * columns // threads
    if slots < width:
        raise ValueError("vector width exceeds elements per thread")
    bits = lambda x: x.bit_length() - 1
    size = bits(rows * columns)
    low = bits(width)
    order = [*range(low), *range(low + bits(threads), size),
             *range(low, low + bits(threads))]
    dims = [*range(bits(columns), size), *range(bits(columns))]
    matrix = [[int(i == j) for j in order] for i in dims]
    reg_bits = bits(slots)
    names = '"register","lane","warp"' if reg_bits else '"lane","warp"'
    widths = [reg_bits, 5, 2] if reg_bits else [5, 2]
    return (f'#frisk.distributed<map=#frisk.bit_linear<inputs=[{names}],'
            f'input_bits={widths},outputs=["dim0","dim1"],'
            f'output_bits=[{bits(rows)},{bits(columns)}],matrix=dense<{matrix}>'
            f' : tensor<{size}x{size}xi1>>,topology=[{slots},32,4,1,1],replication=1>')


def layout_trial(source, opt, folder, request):
    folder.mkdir()
    # Restrict the adapter to a single textual module; refuse to replace
    # a pre-existing request or compiled cost record.
    text = Path(source).read_text()
    if "frisk.normalization_trial" in text or "frisk.compiled_sum_spill" in text:
        raise ValueError("source already contains a compiled-layout request")
    match = re.search(r'(?m)^\s*module\s+(attributes\s*\{|\{)', text)
    if not match:
        raise ValueError("expected an MLIR module")
    tag = "frisk.normalization_trial = {" + request + "}"
    prefix = ("attributes {" + tag + "} {" if match[1] == "{" else "attributes {" + tag + ", ")
    text = text[:match.start(1)] + prefix + text[match.end(1):]
    inp, ir = folder / "input.mlir", folder / "materialized.mlir"
    inp.write_text(text)
    out = subprocess.run([str(opt), str(inp), "--frisk-protected-store-layout=anchor-solver=integer",
                          "-verify-each", "-o", str(ir)], capture_output=True, text=True)
    (folder / "diagnostic.txt").write_text(out.stderr)
    if out.returncode:
        raise ValueError(out.stderr.splitlines()[0] if out.stderr else "layout trial failed")
    receipt = next(line for line in ir.read_text().splitlines() if "frisk.protected_graph =" in line)
    def field(key):
        m = re.search(r'\b' + key + r' = (\d+) : i64', receipt)
        if not m:
            raise ValueError("missing whole-graph receipt field " + key)
        return int(m[1])
    service = sum(field(key) for key in ("access_units", "conversion_units", "reduction_units", "main_loop_units"))
    return ir, dict(service=service, conversions=field("distinct_conversions"),
                    reduction_choices=field("normalization_reduction_choices"),
                    original_domain="normalization_original_domain = true" in receipt)


def bind_graph_price(opt, ir, cubin_digest, load_words, store_words, *, access=None):
    """Seal the materialized IR and ask frisk-opt to attach the compiled cost."""
    def run(flag, dest):
        subprocess.run([str(opt), str(ir), flag, "-verify-each", "-o", str(dest)],
                       check=True, capture_output=True, text=True)
    sealed, priced = ir.parent / "sealed.mlir", ir.parent / "priced.mlir"
    run("--frisk-price-normalization-spill", sealed)
    match = re.search(r'frisk.normalization_graph_digest = "([0-9a-f]{64})"', sealed.read_text())
    if not match:
        raise ValueError("missing normalization graph seal")
    options = ""
    if access is not None:
        histogram = lambda counts: ",".join(map(str, counts))
        options = f" ldg={histogram(access.loads)} stg={histogram(access.stores)}"
        if access.exposed_reloads is not None:
            options += f" exposed-reloads={access.exposed_reloads}"
        priced = ir.parent / "vector-priced.mlir"
    try:
        run("--frisk-price-normalization-spill=" + f"graph={match[1]} cubin={cubin_digest} "
        f"load-words={load_words} store-words={store_words}" + options, priced)
    except subprocess.CalledProcessError as exc:
        if access is not None and ("global instruction footprint is not certified" in exc.stderr or
                                   "global access evidence requires one load and one store" in exc.stderr):
            raise UnsupportedGlobalAccess(exc.stderr.splitlines()[0]) from exc
        raise
    receipt = priced.read_text()
    total = re.search(r'compiled_service_units = (\d+) : i64', receipt)
    if not total or f'compiled_spill_cubin = "{cubin_digest}"' not in receipt:
        raise ValueError("invalid whole-graph spill receipt")
    return priced, int(total[1])


def attach_global_price(opt, ir, record, listing, entry):
    access = global_access(listing, entry)
    path, physical = bind_graph_price(opt, ir, record["cubin"], record["load_words"],
                                      record["store_words"], access=access)
    text = path.read_text()
    def amount(key):
        return int(re.search(r"\b" + key + r" = (\d+) : i64", text)[1])
    rd, wr = (amount("compiled_global_" + direction + "_sectors") for direction in ("read", "write"))
    issues = 4 * (sum(access.loads) + sum(access.stores))
    if (amount("compiled_global_read_issues") + amount("compiled_global_write_issues") != issues or
            physical != record["total"] - amount("access_units") + rd + wr + amount("compiled_global_issue_weight") * issues):
        raise ValueError("compiled global cost receipt mismatch")
    record.update(physical_total=physical, read_sectors=rd, write_sectors=wr,
                  ldg=access.loads, stg=access.stores, interior_reloads=access.interior_reloads)
    if access.exposed_reloads is not None:
        units = amount("compiled_exposed_reload_units")
        total = amount("compiled_dependency_service_units")
        if (units != 4 * amount("compiled_global_issue_weight") * access.exposed_reloads or
                total != physical + units):
            raise ValueError("compiled local dependency receipt mismatch")
        record.update(exposed_reloads=access.exposed_reloads, dependency_total=total)
    return path


def previous_choice(records):
    if not records or "incumbent" not in records:
        raise ValueError("missing normalization incumbent")
    # The static incumbent is an optimum over the original domain. Without
    # observed local instructions this spill repair has nothing to correct.
    if not records["incumbent"]["local_words"]:
        return "incumbent"
    return min(records, key=lambda key: (records[key]["total"], records[key]["conversions"],
                                         key != "incumbent"))


def choose(records):
    """Select using the complete compiled I/O evidence when available."""
    return physical_choice(records, previous_choice(records))


@dataclass
class NormalizationSelection:
    image: object
    materialized: Path
    receipt: dict


def compile_normalization_layout(source, opt, output_dir, rows, columns, prepare, *, audit=False):
    """Compile legal candidates, bind whole-image spill to each witness, dispatch the winner.

    audit=True additionally compiles alternatives for a spill-free incumbent.
    It does not change the gate or the selected layout.
    """
    location = Path(output_dir)
    location.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="norm-", dir=location))
    from triton.runtime.errors import OutOfResources
    records, arms, errors = {}, {}, {}
    def trial(name, request):
        ir, record = layout_trial(source, opt, work / name, request)
        text = ir.read_text()
        shapes = re.findall(r'source_shape = array<i64: (\d+), (\d+)>', text)
        if not shapes or any((int(m), int(n)) != (rows, columns) for m, n in shapes):
            raise ValueError("candidate does not match the supplied normalization tile")
        arm = compile_sum_arms([name], lambda _: prepare(ir))[name]
        words = arm.traffic.load_words + arm.traffic.store_words
        # One 32-bit word per lane uses four 32-byte sectors per warp. Evidence
        # belongs to the complete cubin, irrespective of SSA uses or reductions.
        sectors = words * 16
        priced, total = bind_graph_price(opt, ir, arm.digest,
                                         arm.traffic.load_words, arm.traffic.store_words)
        if total != record["service"] + sectors:
            raise ValueError("independent layout and spill cost mismatch")
        record.update(load_words=arm.traffic.load_words, store_words=arm.traffic.store_words,
                      local_words=words, spill_sectors=sectors, total=total,
                      cubin=arm.digest, ir_sha256=sha256(ir.read_bytes()).hexdigest())
        (ir.parent / "kernel.cubin").write_bytes(arm.image.asm["cubin"])
        (ir.parent / "kernel.sass").write_text(arm.listing)
        try:
            physical = attach_global_price(opt, ir, record, arm.listing, arm.image.name)
        except UnsupportedGlobalAccess as exc:
            record["global_unavailable"] = str(exc)
            physical = None
        records[name], arms[name] = record, (arm.image, priced, physical)
    trial("incumbent", "")
    if audit or records["incumbent"]["local_words"]:
        trials = [(f"original-{i}", f"reduction_index={i}:i64")
                  for i in range(records["incumbent"]["reduction_choices"])]
        for width in (1, 2, 4):
            if width <= min(columns, rows * columns // 128):
                trials.append((f"striped-{width}", "uniform=" + striped_encoding(rows, columns, width)))
        for name, request in trials:
            try:
                trial(name, request)
            except (ValueError, OutOfResources) as exc:
                errors[name] = str(exc)
    previous = previous_choice(records)
    winner = physical_choice(records, previous)
    image, old_ir, physical_ir = arms[winner]
    complete = all("physical_total" in r for r in records.values())
    ir = physical_ir if complete else old_ir
    if sha256(image.asm["cubin"]).hexdigest() != records[winner]["cubin"]:
        raise ValueError("stale normalization cubin")
    receipt = dict(source_sha256=sha256(Path(source).read_bytes()).hexdigest(), selected=winner,
                   previous=previous, global_certified=complete,
                   dependencies_certified=all("dependency_total" in r for r in records.values()),
                   directory=str(work), records=records, rejected=errors)
    (work / "selection.json").write_text(json.dumps(receipt, indent=2))
    return NormalizationSelection(image, ir, receipt)
