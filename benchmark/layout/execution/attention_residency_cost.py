"""Timing-free SM90 residency costs for the checked Attention K4 planner."""

from math import ceil


H100_PCIE_SMS = 114
WARP_SIZE = 32
WARP_REGISTER_UNIT = 256
REGISTERS_PER_SM = 65536
SHARED_PER_SM = 228 * 1024
DRIVER_SHARED_PER_CTA = 1024
WARPS_PER_SM = 64


def residency(*, registers, shared_bytes, threads=128, grid, sms=H100_PCIE_SMS):
    """Use ptxas register allocation granularity and conservative Shared overhead."""
    if (type(registers) is not int or not 1 <= registers <= 255
            or type(shared_bytes) is not int or shared_bytes <= 0
            or type(threads) is not int or threads <= 0 or threads % WARP_SIZE
            or type(grid) is not int or grid < 1 or type(sms) is not int or sms < 1):
        raise ValueError('invalid SM90 residency input')
    warps = threads // WARP_SIZE
    allocated = ceil(registers * WARP_SIZE / WARP_REGISTER_UNIT) * WARP_REGISTER_UNIT * warps
    limits = dict(registers=REGISTERS_PER_SM // allocated, shared=SHARED_PER_SM // shared_bytes,
                  warps=WARPS_PER_SM // warps, cta=32)
    resident = min(limits.values())
    if resident == 0:
        raise ValueError('native resource limits preclude one resident CTA')
    capacity = sms * resident
    full_waves, residual = divmod(grid, capacity)
    # Any SM that receives tail work after a full wave needs at least one
    # extra CTA slot. Divide its slot demand by the resident capacity.
    # This is a scheduling pressure proxy, rather than an achieved occupancy
    # measurement. It avoids charging a whole extra wave to a one-CTA tail.
    slots_on_busiest_sm = ceil(grid / sms)
    pressure = max(1.0, slots_on_busiest_sm / resident)
    return dict(registers_per_thread=registers, registers_allocated_per_cta=allocated,
                shared_bytes_per_cta=shared_bytes, per_sm_limits=limits,
                ctas_per_sm=resident, capacity=capacity, full_waves=full_waves,
                tail_ctas=residual, slots_on_busiest_sm=slots_on_busiest_sm,
                pressure=pressure)


def score(work, *, registers, shared_bytes, grid, threads=128, sms=H100_PCIE_SMS):
    evidence = residency(registers=registers, shared_bytes=shared_bytes, grid=grid,
                         threads=threads, sms=sms)
    return dict(evidence, work=work, cost=work * evidence['pressure'])


def compiled_score(features, proof, resources):
    """Re-rank a paid native slot using its actual ptxas resources."""
    raw_shared = resources['raw_shared_bytes']
    static_shared = proof['static_shared_bytes']
    if raw_shared not in (static_shared, static_shared + DRIVER_SHARED_PER_CTA):
        raise ValueError('native Shared bytes do not agree with the checked plan')
    result = score(features['work'], registers=resources['registers_per_thread'],
                   shared_bytes=max(raw_shared, static_shared + DRIVER_SHARED_PER_CTA),
                   grid=proof['grid'][0], threads=proof['block'][0])
    result['local_or_stack'] = any(resources[key] for key in
                                  ('raw_local_bytes', 'stack_bytes_per_thread'))
    return result
