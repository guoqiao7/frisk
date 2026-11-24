"""Compile-time global I/O evidence for the four-warp SM90 normalization scorer."""
from dataclasses import dataclass
import re


class UnsupportedGlobalAccess(ValueError):
    """The disassembly or its address footprint cannot certify global service."""


@dataclass(frozen=True)
class GlobalAccess:
    loads: tuple
    stores: tuple
    interior_reloads: int
    exposed_reloads: int | None = None
    fp32_issues: int | None = None


# On SM90, the high instruction word holds the producer's write-dependency
# barrier in bits 46..48 and the consumer's wait mask in bits 52..57.
_ENCODED = re.compile(
    r"(?m)^\s*/\*([0-9a-fA-F]+)\*/\s+(?:(@!?\w+)\s+)?"
    r"([A-Z][A-Z0-9_]*)([.A-Z0-9_]*)([^;\n]*);\s*"
    r"/\* 0x[0-9a-fA-F]{16} \*/\s*/\* 0x([0-9a-fA-F]{16}) \*/")
_FP_CONSUMERS = {"FADD", "FMUL", "FFMA", "FSEL", "MUFU", "F2F", "F2I", "I2F"}


# Ordinary real FP32 arithmetic emitted by the supported normalization lowering.
# Keep address arithmetic, lane communication and memory issues on their own bills.
_FP32 = {"FADD", "FMUL", "FFMA", "FMNMX", "FSEL"}
_FP32_MUFU = {".RSQ", ".RCP", ".EX2", ".LG2", ".SIN", ".COS", ".TANH"}


def _compiled_work(body):
    """Return exposed local dependencies and uniform FP32 issues per warp.

    Count dependencies only in closed I/O-free BAR phases. Count arithmetic
    across the executed function up to its unconditional exit. Missing words
    or a control edge invalidate both counts. Predicated arithmetic invalidates
    its issue count independently of the local dependency evidence.
    """
    fp32 = 0
    uniform_fp32 = True
    previous = None
    closed = has_global = False
    phase = total = 0
    for index, match in enumerate(_ENCODED.finditer(body)):
        address, pred, op, mods, operands, high = match.groups()
        if int(address, 16) != 16 * index:
            return None, None
        if op == "EXIT":
            if pred:
                return None, None
            return total, fp32 if uniform_fp32 else None
        if op in {"BRA", "BRX", "BSSY", "BREAK", "BSYNC", "CALL", "RET", "JMP", "JMX"}:
            return None, None
        if op in _FP32 or op == "MUFU":
            # 64-bit approximation opcodes also use MUFU.
            if pred or (op == "MUFU" and mods not in _FP32_MUFU):
                uniform_fp32 = False
            fp32 += 1
        control = int(high, 16)
        if op == "BAR" and "SYNC" in mods.split("."):
            if pred:
                return None, None
            if closed and not has_global:
                total += phase
            closed, has_global, phase = True, False, 0
        elif op in ("LDG", "STG"):
            has_global = True
        elif previous and not pred and op in _FP_CONSUMERS:
            slot, first, width = previous
            if (control >> 52) & (1 << slot):
                sources = re.findall(r"\bR(\d+)\b", operands.partition(",")[2])
                if any(first <= int(reg) < first + width for reg in sources):
                    phase += 1
        previous = None
        if op == "LDL" and not pred:
            dest = re.fullmatch(r"\s*R(\d+)\s*", operands.partition(",")[0])
            slot = (control >> 46) & 7
            if dest and slot < 6:
                width = 4 if ".128" in mods else 2 if ".64" in mods else 1
                previous = slot, int(dest[1]), width
    return None, None


def exposed_reload_uses(body):
    return _compiled_work(body)[0]


def global_access(listing, entry):
    """Count emitted widths and reloads in I/O-free, barrier-bounded phases.

    The caller already validates the whole-image SASS with local_traffic().
    A control edge that skips or repeats global instructions invalidates the
    static I/O count. The interior reload count is a scheduling proxy.
    """
    functions = re.split(r"(?m)^\s*Function\s*:\s*(\S+)\s*$", listing)
    bodies = [b for fn, b in zip(functions[1::2], functions[2::2]) if fn == entry]
    if len(bodies) != 1:
        raise UnsupportedGlobalAccess("ambiguous global I/O function")
    hist = {"LDG": [0, 0, 0], "STG": [0, 0, 0]}
    pcs, edges, exits = [], [], []
    closed = False
    has_global = False
    reloads = interior = 0
    for match in re.finditer(r"(?m)^\s*/\*([0-9a-fA-F]+)\*/\s+(?:(@!?\w+)\s+)?"
                             r"([A-Z][A-Z0-9_]*)([.A-Z0-9_]*)([^\n;]*);", bodies[0]):
        address, pred, opcode, mods, args = match.groups()
        pc = int(address, 16)
        if opcode in hist:
            fields = set(mods.strip(".").split(".")) - {""}
            if pred or fields - {"E", "32", "64", "128"} or len(fields & {"32", "64", "128"}) > 1:
                raise UnsupportedGlobalAccess("unsupported predication or global instruction width")
            hist[opcode][2 if "128" in fields else 1 if "64" in fields else 0] += 1
            pcs.append(pc)
            has_global = True
        elif opcode == "LDL":
            reloads += 1
        elif opcode == "BAR" and "SYNC" in mods.split("."):
            if closed and not has_global:
                interior += reloads
            closed, has_global, reloads = True, False, 0
        elif opcode in ("BRA", "BSSY"):
            dest = re.search(r"\b0x([a-fA-F0-9]+)\s*$", args)
            if not dest:
                raise UnsupportedGlobalAccess("unknown global control-flow span")
            # A convergence marker is conservative when paired with BREAK.
            edges.append((pc, int(dest[1], 16)))
        elif opcode == "EXIT":
            exits.append(pc)
    if not pcs or not exits or any(pc > min(exits) for pc in pcs):
        raise UnsupportedGlobalAccess("missing global I/O or path-dependent exit")
    for start, end in edges:
        if any(min(start,end) <= pc < max(start,end) for pc in pcs):
            raise UnsupportedGlobalAccess("path-dependent global instruction count")
    if not all(sum(v) for v in hist.values()):
        raise UnsupportedGlobalAccess("missing global load or store")
    return GlobalAccess(tuple(hist["LDG"]), tuple(hist["STG"]), interior, *_compiled_work(bodies[0]))


def physical_choice(records, previous):
    """Compare complete physical bills while protecting the serial read phase."""
    if not records["incumbent"]["local_words"] or any("physical_total" not in r for r in records.values()):
        return previous
    reference = records[previous]
    def eligible(r):
        extra_reloads = r["interior_reloads"] - reference["interior_reloads"]
        saved_loads = sum(reference["ldg"]) - sum(r["ldg"])
        return extra_reloads <= max(saved_loads, 0)
    price = "dependency_total" if all("dependency_total" in r for r in records.values()) else "physical_total"
    if all("compute_total" in r for r in records.values()):
        price = "compute_total"
    return min((k for k,r in records.items() if eligible(r)),
               key=lambda k:(records[k][price], records[k]["conversions"], k != previous))
