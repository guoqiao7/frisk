"""Regression for the paired late score and its fixed-graph scope."""
from pathlib import Path
import re
import subprocess
import sys

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo/"benchmark/layout/execution"))
from h100_paired_mainloop import ReplayLoopPair, paired_replay_mainloop

source, opt = Path(sys.argv[1]).read_text(), sys.argv[2]
store = next(x for x in source.splitlines(True) if "frisk.tile_store " in x)
encoding = re.search(r'tensor<\d+x\d+xf32, (#frisk\.distributed<.*>)>, memref<', store)[1]
flags = "--frisk-protected-store-layout=fixed-anchors=true striped-shared-replay=true"

def context(delta):
    data = ReplayLoopPair(encoding, delta, 0).mlir_field()
    return source.replace("copy_overlap = true", data+", copy_overlap = true")

def run(src, flag=flags, error=None):
    proc = subprocess.run([opt,flag,"-verify-each"],input=src,text=True,capture_output=True)
    if error:
        assert proc.returncode and error in proc.stderr, proc.stderr
    else:
        assert proc.returncode == 0, proc.stderr
    return proc.stdout

positive, negative = run(context(100000)), run(context(-100000))
for text, route, delta in [(positive,0,4300000),(negative,1,-4300000)]:
    receipt = next(x for x in text.splitlines() if 'frisk.tile_store ' in x)
    assert f'frisk.mainloop_delta_q10 = {delta} : i64' in receipt
    assert int(re.search(r'frisk.store_score = array<i64: [^>]+, (\d)>', receipt)[1]) == route
    assert run(text) == text
for work, madd, expected in [(0, 40, 0), (2, 40, 2594), (-2, -40, -2594), (2, -40, 86)]:
    text = run(context(work).replace("warp_imads_delta = 0", f"warp_imads_delta = {madd}"))
    assert f"frisk.mainloop_delta_q10 = {expected} : i64" in text
assert 'frisk.mainloop_delta_q10' not in run(source)
run(context(0), "--frisk-protected-store-layout=fixed-anchors=true",
    "paired main-loop scoring requires the matching fixed source")
run(context(0).replace("warp_imads_delta = 0", "warp_imads_delta = 1073741825"),
    error="invalid frisk.h100_store_schedule")
run(source.replace(store,store+store).replace("copy_overlap = true",
    ReplayLoopPair(encoding,0,0).mlir_field()+", copy_overlap = true"),
    error="paired main-loop scoring requires the matching fixed source")

# One QK backedge and one KV backedge. The nested add alias counts only as
# issued work. The multiply additionally incurs the address arithmetic term.
def listing(inner):
    ops = ["MOV R1, RZ", "MOV R2, RZ", inner, "HGMMA.64x32x16.F32 R0, gdesc[UR2], R0",
           "@!P0 BRA 0x20", "HGMMA.64x32x16.F32 R0, R4, gdesc[UR2], R0",
           "WARPGROUP.DEPBAR.LE gsb0, 0x0", "@!P0 BRA 0x10", "EXIT"]
    return "code for sm_90a\n Function : replay\n"+"\n".join(
        f" /*{16*i:04x}*/ {op} ;" for i,op in enumerate(ops))
a, b = listing("IMAD.IADD R2, R3, 1, R4"), listing("IMAD R2, R3, R5, R4")
def pair(x,y):
    return paired_replay_mainloop(x,y,source=encoding,kv_tiles=3,qk_trips=4,warps=8)
assert pair(a,b)==ReplayLoopPair(encoding,0,96)
assert pair(b,a)==ReplayLoopPair(encoding,0,-96)
for invalid in [b.replace("WARPGROUP.DEPBAR", "WARPGROUP.ARRIVE"), b.replace("BRA 0x20","BRA 0x50")]:
    try:
        pair(a,invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("unexpectedly accepted unsupported control flow or pipeline")

# A native pair with certified scalar guards. With all scheduling differentials
# zero, only the previously unpriced sector premium can change the score.
from native_online_fixture import online_fixture
native = run(online_fixture(256, 64, 277, value_dim=32), "--frisk-layout-pipeline")
native_store = next(line for line in native.splitlines() if "frisk.tile_store " in line)
native_encoding = re.search(r'tensor<\d+x\d+xf32, (#frisk\.distributed<.*>)>, memref<', native_store)[1]

def native_wave(rows, instances):
    fields = ('warp_issues_delta', 'warp_imads_delta', 'warp_load_wait_delta',
              'warp_local_word_delta', 'cache_miss_sectors_delta', 'tail_address_issues_delta',
              'tail_cache_miss_sectors_delta', 'drain_address_issues_delta', 'drain_interstore_issues_delta')
    costs = ", ".join(f"{key} = 0 : i64" for key in fields)
    pair = f'paired_mainloop = {{backend = "native", source = {native_encoding}, {costs}, guarded_wave = true}}'
    schedule = (f"frisk.h100_store_schedule = {{rows = {rows} : i64, instances = {instances} : i64, "
                f"sm_count = 114 : i64, mma_issues = 16 : i64, mma_waits = 16 : i64, copy_overlap = false, {pair}}}")
    marked = native_store.replace(" : tensor<", " {" + schedule + "} : tensor<", 1)
    return native.replace(native_store, marked, 1)

native_flags = "--frisk-protected-store-layout=fixed-anchors=true"
for rows, heads in [(33, 8), (277, 8), (277, 128)]:
    ir = native_wave(rows, heads)
    result = run(ir, native_flags)
    def ints(field):
        match = re.search(r"frisk\." + field + r" = array<i64: ([^>]+)>", result)
        return tuple(int(value) for value in match[1].split(", "))
    full_a, full_b, tail_a, tail_b, grid, sms = ints("wave_sectors")
    avg_a, avg_b, rate, _, _, _ = ints("scheduled_store_score")
    assert grid == (rows + 255) // 256 * heads and sms == 114
    peak = max(tail_a - tail_b, full_a - full_b if rows > 256 else 0)
    billed = (rate * avg_a + 512) // 1024 - (rate * avg_b + 512) // 1024
    expected = max(0, (3794 * peak * min(grid, sms) + grid // 2) // grid - billed)
    assert f"frisk.wave_delta_q10 = {-expected} : i64" in result
    assert run(result, native_flags) == result
run(native_wave(256, 8), native_flags, "invalid frisk.h100_store_schedule")
run(native_wave(129, 8).replace("guarded_wave = true", "guarded_wave = false"),
    native_flags, "invalid frisk.h100_store_schedule")
