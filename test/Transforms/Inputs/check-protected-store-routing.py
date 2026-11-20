"""Exercise protected store decisions through the CLI, including real MLIR CSE."""
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

mode, opt, fixture = sys.argv[1:]
source = Path(fixture).read_text()
protected = "--frisk-protected-store-layout"
fixed = protected + "=fixed-anchors=true"
replay = " striped-shared-replay=true"
normal = "--frisk-layout-pipeline"


def run(text, *flags):
    return subprocess.run([opt, *flags, "-verify-each"], input=text, text=True,
                          check=True, stdout=subprocess.PIPE).stdout


def stores(text):
    return [line for line in text.splitlines() if "frisk.tile_store " in line]


def charge(line):
    return int(re.search(r"store_route_charge_q10 = (\d+)", line)[1])


def cost(text):
    return int(re.search(r"fixed_store_cost = array<i64: (\d+)", text)[1])


def receipt(text):
    # SSA names can change after CSE. Compare the exact anchor, per-use
    # decisions/schedules and source-vs-routed operand relationships.
    edges = []
    owners = {line.strip().split()[0] for line in text.splitlines()
              if " = frisk.convert_layout " in line and "frisk.protected_store_route" in line}
    for line in stores(text):
        value = re.search(r"frisk.tile_store (%\w+)", line)[1]
        edges.append((value in owners, line.split(" {")[1].split("} : tensor")[0]))
    anchors = [line.split(" {", 1)[1] for line in text.splitlines() if " = frisk.mma " in line]
    return cost(text), len(owners), edges, anchors


def schedule(text, index=0, rows=1):
    lines = text.splitlines()
    ids = [i for i, line in enumerate(lines) if "frisk.tile_store " in line]
    i = ids[index]
    ctx = ("frisk.h100_store_schedule = {copy_overlap = true, instances = 1 : i64, "
           f"mma_issues = 8 : i64, mma_waits = 2 : i64, rows = {rows} : i64, sm_count = 114 : i64}}")
    base, suffix = lines[i].split(" : tensor<", 1)
    base = base[:-1] + ", " + ctx + "}" if base.endswith("}") else base + " {" + ctx + "}"
    lines[i] = base + " : tensor<" + suffix
    return "\n".join(lines) + "\n"


def check_history(inp, open_flag, fixed_flag, rows):
    bare = run(inp, normal)
    opened = run(inp, open_flag)
    assert "frisk.protected_store_use" in stores(opened)[0]
    results = [run(schedule(x, rows=rows), fixed_flag)
               for x in (bare, opened, run(opened, "--cse"))]
    assert receipt(results[0]) == receipt(results[1]) == receipt(results[2])
    assert receipt(results[0])[1] == 0
    assert run(results[1], fixed_flag) == results[1]
    run(opened, normal)
    run(results[1], normal)

    # Before provenance was introduced, an intervening rescore could replace
    # the last positive receipt with an identity score on the already-routed
    # value. Recover the old positive receipt, and reject the ambiguous one.
    legacy = re.sub(r" \{frisk.protected_store_route\}", "", opened)
    legacy = legacy.replace("frisk.protected_store_use, ", "")
    legacy = re.sub(r'frisk.store_route_\w+ = (?:array<i64: [^>]+>|"[^"]+"|\d+ : i64|true|false), ', "", legacy)
    assert "frisk.store_route_kind" not in legacy and "frisk.protected_store_route" not in legacy
    assert receipt(run(schedule(legacy, rows=rows), fixed_flag)) == receipt(results[0])
    old = legacy.replace("module attributes {", "module attributes {frisk.fixed_store_cost = array<i64: 0, 0, 0>, ", 1)
    old = re.sub(r"frisk.store_score = array<i64: [^>]+>", "frisk.store_score = array<i64: 1, 1, 0, 0>", old)
    rejected = subprocess.run([opt, fixed_flag], input=old, text=True, capture_output=True)
    assert rejected.returncode and "legacy store origin is ambiguous" in rejected.stderr


def check_consumers(inp):
    # Two stores to distinct user buffers; only their data SSA is shared.
    line, = stores(inp)
    extra = ("      %out2 = frisk.layout_view %k {layout=#a} : memref<4x128xf32,1> -> memref<4x128xf32,1>\n"
             + line.replace("%og", "%out2"))
    both = inp.replace(line, line + "\n" + extra)
    routed = run(both, fixed)
    x, y = stores(routed)
    assert all("frisk.protected_store_use" in s for s in (x, y))
    assert charge(x) > 0 and charge(y) == 0
    assert cost(routed) == cost(run(inp, fixed)) + 64
    assert receipt(routed)[1] == 1

    # Recreate the *old* duplicate converters, then really let CSE fold them.
    cv, = [l for l in routed.splitlines() if " = frisk.convert_layout " in l]
    name = cv.strip().split()[0]
    duplicate = routed.replace(y, cv.replace(name+" =", "%duplicate =") + "\n"
                               + y.replace("tile_store "+name+",", "tile_store %duplicate,"))
    assert duplicate.count(" = frisk.convert_layout ") == 2
    shared = run(duplicate, "--cse")
    assert shared.count(" = frisk.convert_layout ") == 1
    assert receipt(run(shared, fixed)) == receipt(routed)
    for index in (0, 1):
        direct = run(schedule(both, index), fixed)
        rescored = run(schedule(shared, index), fixed)
        assert receipt(direct) == receipt(rescored)
        edges = receipt(direct)[2]
        assert [e[0] for e in edges] == [i != index for i in range(2)]
        assert charge(stores(direct)[index]) == 0
    # A user-written conversion has no selection receipt or ownership tag.
    explicit = run(inp, fixed)
    explicit = re.sub(r" \{frisk.protected_store_route\}", "", explicit)
    explicit = re.sub(r"(frisk.tile_store [^{}\n]+) \{[^\n]+?\} : tensor", r"\1 : tensor", explicit)
    assert explicit.count(" = frisk.convert_layout ") == 1
    fresh = run(schedule(explicit), fixed)
    assert fresh.count(" = frisk.convert_layout ") == 1
    assert 'store_route_kind = "Identity"' in stores(fresh)[0]


def check_routes(inp):
    for kind, bit in (("Register", None), ("Shuffle", 2), ("Shared", 7)):
        matrix = ast.literal_eval(re.search(r"matrix=dense<(.+?)> :", inp)[1])
        if bit is not None:
            matrix[bit][3] = 1
        text = re.sub(r"matrix=dense<.+?> :", "matrix=dense<"+str(matrix)+"> :", inp)
        scored = run(text, fixed)
        store, = stores(scored)
        assert f'store_route_kind = "{kind}"' in store and "frisk.protected_store_use" in store
        parts = list(map(int, re.search(r"store_route_service = array<i64: ([^>]+)>",store)[1].split(",")))
        assert charge(store) == 1024 * sum(parts[:4])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"execution.json"
            run(scored, "--frisk-export-execution=output="+str(output))
            plans = [o["redistribution"] for o in json.loads(output.read_text())["operations"]
                     if "redistribution" in o]
        plan, = plans
        assert (plan["mode"], plan["scratch_bytes"]) == (kind, parts[4])
        assert (parts[1] > 0) == (kind == "Shuffle")
        assert (parts[2] > 0 and parts[3] > 0) == (kind == "Shared")


if mode == "history":
    check_history(source, protected, fixed, 1)
    check_history(source, protected+"=striped-shared-replay=true", fixed+replay, 1)
elif mode == "consumers":
    check_routes(source)
    check_consumers(source)
else:
    raise ValueError(mode)
