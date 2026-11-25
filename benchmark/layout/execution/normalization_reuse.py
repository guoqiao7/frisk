"""Cross-launch locality for dense single-input, single-output SM90 normalization."""
from dataclasses import dataclass
from itertools import product
from typing import Literal

Distance = int | Literal["cold"] | None


def _endpoints(span):
    start, size = span
    if type(start) is not int or start < 0 or type(size) is not int or size <= 0:
        raise ValueError("expected a nonnegative byte address and positive byte length")
    return start, start + size


def _union_bytes(intervals):
    total = 0
    end = 0
    for lo, hi in sorted(intervals):
        total += max(0, hi - max(lo, end))
        end = max(end, hi)
    return total


@dataclass(frozen=True)
class NormalizationReuse:
    """Unique bytes touched over an operand's reuse interval, including the operand.

    A positive distance estimates recurring access, 'cold' denotes a first touch
    or streaming allocation, and None denotes missing information. A direct
    distance may include work from intervening graph nodes on the same device.
    """
    input_distance_bytes: Distance = None
    output_distance_bytes: Distance = None

    def __post_init__(self):
        for distance in (self.input_distance_bytes, self.output_distance_bytes):
            if distance is not None and distance != "cold" and (type(distance) is not int or distance <= 0):
                raise ValueError("reuse distance must be positive bytes, 'cold', or None")

    @classmethod
    def from_launches(cls, launches, index, *, cyclic=False, cold_start=False):
        """Infer both distances from actual launches in one ordered device stream.

        Each launch supplies ((input address, bytes), (output address, bytes)).
        cyclic=True describes steady-state replay. cold_start=True certifies the
        start of a nonrepeating trace. An unobserved prefix or partial coverage
        yields unknown. Whole-kernel footprints approximate intra-launch order.
        Only byte distances survive in the result.
        """
        if cyclic and cold_start:
            raise ValueError("a cyclic trace cannot start cold")
        spans = [(_endpoints(x), _endpoints(y)) for x, y in launches]
        if type(index) is not int or not 0 <= index < len(spans):
            raise ValueError("launch index outside reuse trace")
        x, y = spans[index]
        if max(x[0], y[0]) < min(x[1], y[1]):
            return cls()  # In-place read/write ordering needs finer-grain evidence.

        def distance(target):
            intervals = []
            steps = len(spans) if cyclic else index
            for step in range(1, steps + 1):
                pair = spans[(index - step) % len(spans)]
                intervals.extend(pair)
                if any(lo <= target[0] and target[1] <= hi for lo, hi in pair):
                    return _union_bytes(intervals)
                if any(max(lo, target[0]) < min(hi, target[1]) for lo, hi in pair):
                    return None
            return "cold" if cold_start else None

        return cls(distance(x), distance(y))


def validate_reuse(reuse, l2_bytes):
    if isinstance(reuse, NormalizationReuse) and l2_bytes is None:
        if all(v is None or v == "cold" for v in (reuse.input_distance_bytes, reuse.output_distance_bytes)):
            return
    if not isinstance(reuse, NormalizationReuse) or type(l2_bytes) is not int or l2_bytes <= 0:
        raise ValueError("reuse ranking requires NormalizationReuse and positive device L2 bytes")


def miss_fractions(distance, capacity):
    if distance is None:
        return (0.0, 1.0)
    if distance == "cold":
        return (1.0,)
    # Piecewise approximation of the H100 PCIe streaming-copy capacity curve.
    # The same residency curve is evaluated independently for each operand.
    points = ((0.50, 0.), (0.64, .04), (.72, .14), (.80, .41), (.88, .89), (.96, .98), (1., 1.))
    ratio = distance / capacity
    if ratio <= points[0][0]:
        return (0.,)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if ratio <= x1:
            return (y0 + (y1 - y0) * (ratio - x0) / (x1 - x0),)
    return (1.,)


def reuse_scenarios(reuse, capacity):
    validate_reuse(reuse, capacity)
    return product(miss_fractions(reuse.input_distance_bytes, capacity),
                   miss_fractions(reuse.output_distance_bytes, capacity))


def reuse_service(record, grid, sms, read_miss, write_miss, min_read, min_write):
    """Relative work: scheduled execution plus layout-dependent off-chip service.

    Discount overlap among independent global-load waits within a resident wave.
    Retain per-warp issue, local/shared dependencies and the sector service floor.
    Requested LSU work uses a nominal two scheduling cycles per warp instruction
    under streaming pressure. Excess transactions use four cycles per 32B sector.
    Common payload bytes cancel when comparing layouts of this same launch.
    """
    profile = record["schedule"]
    cap = profile["resident_ctas"]
    full, tail = divmod((grid + sms - 1) // sms, cap)
    sectors = record["read_sectors"] + record["write_sectors"] + record["spill_sectors"]

    def wave(n):
        w = profile["waves"][n - 1]
        issue = w["issue"] + w["local_wait"] + w["shared_wait"] + w["global_wait"] / n
        return max(issue, n * sectors / 4)

    scheduled = full * wave(cap) + (wave(tail) if tail else 0)
    load = 2 * sum(record["ldg"]) + 4 * (record["read_sectors"] - min_read)
    store = 2 * sum(record["stg"]) + 4 * (record["write_sectors"] - min_write)
    return scheduled + max(1., grid / sms) * (read_miss * load + write_miss * store)
