"""Emit the per-consumer service context for a frozen SM90a scalar F32 row store."""

from dataclasses import dataclass, replace
import re
import subprocess

if __package__:
    from .h100_paired_mainloop import ReplayLoopPair, paired_replay_mainloop
else:
    from h100_paired_mainloop import ReplayLoopPair, paired_replay_mainloop


@dataclass(frozen=True)
class H100StoreSchedule:
    rows: int
    instances: int
    sm_count: int
    mma_issues: int
    mma_waits: int
    copy_overlap: bool
    paired_mainloop: ReplayLoopPair | None = None

    def __post_init__(self):
        if self.paired_mainloop is not None and not isinstance(self.paired_mainloop, ReplayLoopPair):
            raise ValueError("invalid replay loop pair")
        counts = (self.rows, self.instances, self.sm_count, self.mma_issues, self.mma_waits)
        if any(type(v) is not int or not 0 <= v <= 2**30 for v in counts):
            raise ValueError("store schedule counts must be bounded nonnegative integers")
        if min(counts[:3]) == 0 or self.mma_waits > self.mma_issues or type(self.copy_overlap) is not bool:
            raise ValueError("invalid launch dimensions, wait density, or copy overlap")

    def with_compiled_replay(self, direct, striped, *, source, kv_tiles, qk_trips, warps,
                             cuobjdump="cuobjdump"):
        """Attach static feedback from two images emitted for this store and frozen plan.

        Provide the original tensor encoding and the plan's executed trip counts.
        Both candidates must come from the same lowering with only the terminal
        store changed. The extractor checks the compiled tensor/barrier schedule.
        """
        listings = [subprocess.check_output([cuobjdump, "-sass", str(image)], text=True)
                    for image in (direct, striped)]
        pair = paired_replay_mainloop(*listings, source=source, kv_tiles=kv_tiles,
                                      qk_trips=qk_trips, warps=warps)
        return replace(self, paired_mainloop=pair)

    def mlir_attribute(self):
        fields = [f"{key} = {value} : i64" for key, value in vars(self).items()
                  if key not in ("copy_overlap", "paired_mainloop")]
        fields.append("copy_overlap = " + ("true" if self.copy_overlap else "false"))
        if self.paired_mainloop is not None:
            fields.append(self.paired_mainloop.mlir_field())
        return "frisk.h100_store_schedule = {" + ", ".join(fields) + "}"


def online_store_schedule(plan, mainloop_ptx, sm_count):
    """Consume a fixed online-attention plan and its pre-ptxas instruction stream.

    The atom/wait ratio is computed before register allocation. The output
    store may be a placeholder: neither store instructions nor timings enter.
    """
    if plan.get("schema") != "frisk.online-attention-pipeline.v1":
        raise ValueError("expected a frozen online-attention plan")
    order = plan["order"]
    if not {"qk-issue", "pv-issue"}.issubset(order):
        raise ValueError("no supported QK/PV event pair in the frozen plan")
    # The supported plan brackets an asynchronous copy with an issue and a wait.
    # A copy overlapping either tensor stage competes for service in the main loop.
    overlap = any(
        start in order and end in order and order.index(start) < order.index(mma) < order.index(end)
        for start, end in (("v-issue", "v-wait0"), ("k-next-issue", "k-next-wait0"))
        for mma in ("qk-issue", "pv-issue")
    )
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", mainloop_ptx, flags=re.S)
    issues = len(re.findall(r"\bwgmma\.mma_async\.", text))
    waits = len(re.findall(r"\bwgmma\.wait_group\.", text))
    if not issues or not waits:
        raise ValueError("missing tensor issue or wait instructions")
    spec = plan["spec"]
    return H100StoreSchedule(spec["N"], spec["B"] * spec["H"], sm_count, issues, waits, overlap)
