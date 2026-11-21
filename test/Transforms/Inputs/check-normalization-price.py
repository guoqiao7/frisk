"""Verify whole-image charging, source identity, and overflow diagnostics."""
import re
import subprocess
import sys
src = sys.argv[1]
flag = "--frisk-price-normalization-spill"
def run(*args):
    return subprocess.run([sys.argv[2], src, *args], capture_output=True, text=True)
seal = run(flag)
seal.check_returncode()
digest = re.search(r'frisk.normalization_graph_digest = "([a-z0-9]{64})"', seal.stdout)[1]
def price(graph, ld, st):
    return run(f"{flag}=graph={graph} cubin={'5'*64} load-words={ld} store-words={st}")
p = price(digest, 3, 2)
p.check_returncode()
def amount(name):
    return int(re.search(name + r" = (\d+) : i64", p.stdout)[1])
assert amount("compiled_spill_units") == 80
assert amount("compiled_service_units") == 80 + sum(amount(key) for key in
    ("access_units", "conversion_units", "reduction_units", "main_loop_units"))
assert price("0"*64, 3, 2).returncode
assert price(digest, 1 << 62, 0).returncode

# The sealed IR itself is a valid input. A stale IR must not inherit its price.
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    src = str(Path(tmp) / "sealed.mlir")
    Path(src).write_text(seal.stdout)
    price(digest, 3, 2).check_returncode()
    Path(src).write_text(seal.stdout.replace(digest, "0" * 64))
    assert price(digest, 3, 2).returncode

# A single scalar read and write per register slot. Eight slots per lane in
# this 8x128 FP32 fixture. Under-coverage and partial evidence are rejected.
src = sys.argv[1]
vector_flag = f"{flag}=graph={digest} cubin={'5'*64} load-words=3 store-words=2"
priced = run(vector_flag + " ldg=8,0,0 stg=8,0,0")
priced.check_returncode()
def vector_amount(key):
    return int(re.search(key + r" = (\d+) : i64", priced.stdout)[1])
assert vector_amount("compiled_global_read_issues") == 32
assert vector_amount("compiled_global_write_issues") == 32
assert vector_amount("compiled_service_units") == (80 + vector_amount("conversion_units")
    + vector_amount("reduction_units") + vector_amount("main_loop_units")
    + vector_amount("compiled_global_read_sectors") + vector_amount("compiled_global_write_sectors")
    + 64 * vector_amount("compiled_global_issue_weight"))
assert run(vector_flag + " ldg=1,0,0 stg=8,0,0").returncode
assert run(vector_flag + " ldg=8,0,0").returncode

dependency_flag = vector_flag + " ldg=8,0,0 stg=8,0,0"
dep = run(dependency_flag + " exposed-reloads=2")
dep.check_returncode()
assert "compiled_exposed_reload_units = 64 : i64" in dep.stdout
assert f"compiled_dependency_service_units = {vector_amount('compiled_service_units') + 64} : i64" in dep.stdout
assert run(dependency_flag + " exposed-reloads=4").returncode
assert run(vector_flag + " exposed-reloads=1").returncode
