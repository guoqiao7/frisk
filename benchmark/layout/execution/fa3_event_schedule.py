"""Steady-state events for the native FA3 consumer. The producer protocol is fixed."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FA3EventSchedule:
    frame: str
    steady: tuple

    def __post_init__(self):
        events = self.steady
        if self.frame not in ("same_tile", "lookahead") or not isinstance(events, tuple):
            raise ValueError("invalid FA3 event frame or sequence")
        if not all(isinstance(e,str) for e in events) or sorted(events) != ["pv", "qk", "release_k", "update"]:
            raise ValueError("each FA3 steady event must occur exactly once")
        if self.frame == "same_tile":
            # The producer issues next K before current V. Keep the original
            # same-tile lifetime, including for a one-version K ring.
            if events != ("qk", "release_k", "update", "pv"):
                raise ValueError("invalid same-tile FA3 dependencies")
        elif events.index("release_k") < events.index("qk") or events[-1] != "update":
            # PV(t) consumes the old probabilities and accumulator before the
            # next softmax update. Release follows QK(t+1)'s wait_group 0.
            raise ValueError("invalid lookahead FA3 dependencies")

    @property
    def contract_kind(self):
        return "fa3_same_tile" if self.frame == "same_tile" else "fa3_qk_next"

    def statements(self):
        current = "t" if self.frame == "same_tile" else "t+1"
        joined = self.steady.index("release_k") == self.steady.index("qk") + 1
        result = []
        for event in self.steady:
            if event == "qk":
                # Keep the original fused primitive when the return is adjacent.
                op = "scores" if joined else "scores_defer_k"
                result.append(f"{op}({current});")
            elif event == "release_k":
                if not joined:
                    result.append(f"bars[K_FREE+kr.slot({current})].arrive();")
            elif event == "pv":
                result.append("accumulate(t);")
            else:
                result.append("update(s,p,o,m,l);")
        return result

    def cuda_definitions(self):
        # Both constexpr branches must parse. Use canonical statements for the
        # inactive frame so legacy schedules retain their exact native lowering.
        same = self if self.frame == "same_tile" else SCHEDULES["same_tile"]
        ahead = self if self.frame == "lookahead" else SCHEDULES["qk_next"]
        lines = []
        if ahead.steady.index("release_k") != ahead.steady.index("qk") + 1:
            lines.append("#define FRISK_FA3_DEFER_K 1")
        for name, schedule in (("SAME_TILE", same), ("LOOKAHEAD", ahead)):
            body = " ".join(schedule.statements())
            lines.append(f"#define FRISK_FA3_{name}_EVENTS {body}")
        return "\n".join(lines)


SCHEDULES = {
    "same_tile": FA3EventSchedule("same_tile", ("qk", "release_k", "update", "pv")),
    "qk_next": FA3EventSchedule("lookahead", ("qk", "release_k", "pv", "update")),
    "pv_first": FA3EventSchedule("lookahead", ("pv", "qk", "release_k", "update")),
    "late_k": FA3EventSchedule("lookahead", ("qk", "pv", "release_k", "update")),
}


def resolve_events(contract_kind, requested=None):
    if contract_kind not in ("fa3_same_tile", "fa3_qk_next"):
        raise ValueError("FA3 events require a native FA3 execution contract")
    if requested is None:
        requested = "same_tile" if contract_kind == "fa3_same_tile" else "qk_next"
    if isinstance(requested, FA3EventSchedule):
        return requested
    if not isinstance(requested, str) or requested not in SCHEDULES:
        raise ValueError(f"unknown FA3 event schedule {requested!r}")
    return SCHEDULES[requested]
