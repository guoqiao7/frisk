"""Smoke-test the actual canonical tool output, including optimization survival."""
import argparse
import json
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--build-dir", default="build")
args = parser.parse_args()
tool = f"{args.build_dir}/exp/layout/frisk_layout_tool"
for form in ("shared", "local"):
    command = [tool, "--target=sm_90a", "--block-m=64", "--block-n=8",
               "--block-k=16", "--threads=128", "--dtype=bf16", f"--a-space={form}"]
    output = json.loads(subprocess.check_output(command, text=True))
    assert output["schema"] == "frisk-canonical-layout-v1"
    assert "#frisk.mma_contract" in output["mma_contract"]
    assert "#frisk.distributed" in output["accumulator_encoding"]
    assert len(output["storage_layouts"]) == (2 if form == "shared" else 1)
    assert "frisk.mma" in output["ir"]
    if form == "local":
        assert "#frisk.distributed" in output["a_encoding"]
bad = subprocess.run([tool, "--target=sm_90", "--block-m=64", "--block-n=8", "--block-k=16"],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
assert bad.returncode != 0, "sm_90 must not silently enable WGMMA"
bad_grid = subprocess.run([tool, "--grid=2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
assert bad_grid.returncode != 0, "unsupported multi-CTA grid must not be ignored"
print("SM90 layout tool: SS, RS, observable canonical output and target rejection passed")
