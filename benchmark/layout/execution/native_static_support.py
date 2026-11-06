"""Capability bound of the current explicit static-PTX Shared emitter.

This is an emitter/toolchain limit, not the device's total Shared capacity or
its opt-in dynamic Shared limit. Keep the full emitted configuration unchanged.
"""

STATIC_SHARED_CONTRACT = "frisk-static-ptx-shared-v1"
STATIC_SHARED_LIMIT_BYTES = 48 * 1024


class StaticSharedUnsupported(ValueError):
    """A valid allocation exceeds the current static-PTX support domain."""

    def __init__(self, actual_bytes):
        self.static_shared_bytes = actual_bytes
        self.static_shared_limit_bytes = STATIC_SHARED_LIMIT_BYTES
        super().__init__(
            "current static PTX backend unsupported: static_shared_bytes="
            + str(actual_bytes) + " exceeds static Shared limit "
            + str(STATIC_SHARED_LIMIT_BYTES)
            + " bytes; device total/opt-in dynamic Shared capacity is a separate limit")


def check_static_shared(info):
    """Validate an emission's static allocation and return its bounded proof.

    Missing or malformed allocation data is invalid, never implicit zero.
    StaticSharedUnsupported distinguishes an otherwise valid out-of-domain
    allocation from malformed configuration. No caller-owned data is changed.
    """
    if type(info) is not dict:
        raise ValueError("emitted configuration must be an object")
    if "static_shared_bytes" not in info:
        raise ValueError("emitted configuration missing static_shared_bytes")
    actual = info["static_shared_bytes"]
    if type(actual) is not int or actual < 0:
        raise ValueError("static_shared_bytes must be a nonnegative integer")
    if actual > STATIC_SHARED_LIMIT_BYTES:
        raise StaticSharedUnsupported(actual)
    return dict(schema_version=1, contract=STATIC_SHARED_CONTRACT,
                allocation_kind="static-ptx", supported=True,
                static_shared_bytes=actual,
                static_shared_limit_bytes=STATIC_SHARED_LIMIT_BYTES)


# The SM90 register allocator rounds each warp to 256-register units.
_SM90_REGISTERS_PER_SM = 65536
_SM90_WARP_REGISTER_UNIT = 256
_SM90_WARPS_PER_CTA = 4
_SM90_WARP_SIZE = 32
_SM90_SHARED_PER_SM = 228 * 1024
_SM90_RESERVED_SHARED_PER_CTA = 1024
_SM90_TTGIR_HEADER = ('module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, '
                      '"ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:90"')


def target_sm90_residency(ttgir, info, blocks_per_sm):
    """Apply a compile-only register budget to checked SM90 single-CTA TTGIR.

    This schedules no additional candidate. The caller must verify native
    resources and loaded driver residency before admitting a capped binary.
    """
    check_static_shared(info)
    shared = info.get('physical_shared_bytes')
    if type(shared) is not int or not info['static_shared_bytes'] <= shared <= STATIC_SHARED_LIMIT_BYTES:
        raise ValueError('checked physical Shared allocation required for SM90 residency')
    if type(blocks_per_sm) is not int or not 2 <= blocks_per_sm <= 4:
        raise ValueError('SM90 residency target must be two to four CTAs per SM')
    if (shared + _SM90_RESERVED_SHARED_PER_CTA) * blocks_per_sm > _SM90_SHARED_PER_SM:
        raise ValueError('SM90 Shared capacity precludes the requested residency')
    if (type(ttgir) is not str or not ttgir.startswith(_SM90_TTGIR_HEADER + '} {\n')
            or '"ttg.maxnreg"' in ttgir):
        raise ValueError('expected uncapped four-warp SM90 TTGIR')
    units = _SM90_REGISTERS_PER_SM // (
        blocks_per_sm * _SM90_WARPS_PER_CTA * _SM90_WARP_REGISTER_UNIT)
    cap = units * _SM90_WARP_REGISTER_UNIT // _SM90_WARP_SIZE
    attribute = f', "ttg.maxnreg" = {cap} : i32'
    constrained = _SM90_TTGIR_HEADER + attribute + ttgir[len(_SM90_TTGIR_HEADER):]
    return constrained, dict(contract='frisk-sm90-residency-target-v1',
                             blocks_per_sm=blocks_per_sm, max_registers_per_thread=cap,
                             threads_per_block=_SM90_WARPS_PER_CTA * _SM90_WARP_SIZE,
                             static_shared_bytes=info['static_shared_bytes'], physical_shared_bytes=shared)


def check_sm90_residency_binary(target, ptx, resources, mnemonics):
    """Refuse a budgeted binary that spills or exceeds the planned register cap."""
    import re

    if target.get('contract') != 'frisk-sm90-residency-target-v1':
        raise ValueError('SM90 residency target proof required')
    cap = target['max_registers_per_thread']
    if (re.findall(r'^\s*\.maxnreg\s+(\d+)\s*$', ptx, re.MULTILINE) != [str(cap)]
            or re.findall(r'^\s*\.reqntid\s+(\d+)\s*$', ptx, re.MULTILINE) !=
            [str(target['threads_per_block'])]):
        raise ValueError('native PTX did not preserve the residency cap and thread contract')
    if not 0 < resources['registers_per_thread'] <= cap:
        raise ValueError('native allocation exceeded the SM90 register target')
    if resources['stack_bytes_per_thread'] or resources['local_bytes'] or not mnemonics or any(
            count and name.split('.', 1)[0] in ('LDL', 'STL') for name, count in mnemonics.items()):
        raise ValueError('SM90 residency target requires zero local memory and spill instructions')
    return dict(target, native_registers_per_thread=resources['registers_per_thread'],
                native_stack_bytes_per_thread=0, native_local_bytes=0, native_spill_instructions=0)
