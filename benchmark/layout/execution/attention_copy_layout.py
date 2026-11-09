"""Proof-bound Copy16 producer layouts for checked compact D64 Attention."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json

from access_plan import linear
from attention_copy_schedule import prove_copy_schedule
from attention_tile_mma import ROOT_BYTES, THREADS, TILE, shared_offsets_64x64
from valid_domain import dense_copy_execution, plan_guarded_copy_owners


_PAYLOAD = 8
_REGISTERS = TILE * TILE // THREADS


def native_copy_plan(execution, rows, *, owner_policy, shared_swizzle=128,
                     destination_offsets=None):
    if destination_offsets is None:
        destination_offsets = shared_offsets_64x64(shared_swizzle)
    return plan_guarded_copy_owners(execution, valid_shape=(rows, TILE),
        source_row_stride_bytes=TILE * 2, source_span_bytes=rows * TILE * 2,
        destination_offsets=destination_offsets, destination_span_bytes=ROOT_BYTES,
        owner_policy=owner_policy)


def _execution(logical, shared_swizzle):
    first = [0] * len(logical)
    for holder, slot in enumerate(logical):
        first[slot] = holder
    execution = dict(shape=[TILE, TILE], topology=[_REGISTERS, 32, 4, 1, 1],
                     logical=logical, first=first)
    native_copy_plan(execution, TILE, owner_policy='execution', shared_swizzle=shared_swizzle)
    return execution


def _packet_with_copy_execution(packet, execution):
    derived = deepcopy(packet)
    for operation in derived['operations']:
        if operation['name'] == 'frisk.copy':
            operation['execution'] = deepcopy(execution)
    return derived


def derive_attention_copy_layouts(tile_packet, *, shared_swizzle=128):
    """Derive two tied Q/K/V owner assignments from one checked C++ export.

    The coalesced packet reifies the legacy physical GF(2) copy plan. The
    other exchanges one register-group bit with the high warp bit, leaving
    individual 16-byte payloads and their Shared endpoints unchanged.
    Lower both packets with copy_owner_policy='execution'.
    """
    from tiled_attention import _checked_tile_layouts

    distribution, _ = _checked_tile_layouts(tile_packet)
    dense = dense_copy_execution((TILE, TILE))
    if any(distribution.get(key) != dense[key] for key in ('shape', 'topology', 'logical', 'first')):
        raise ValueError('Copy16 layout derivation requires the original dense C++ execution witness')
    optimized = native_copy_plan(distribution, TILE, owner_policy='gf2', shared_swizzle=shared_swizzle)
    coalesced = [group.source_offsets[tid] // 2 + reg for tid in range(THREADS)
                 for group in optimized.groups for reg in range(_PAYLOAD)]
    register_group_bit = _PAYLOAD
    high_warp_bit = _REGISTERS * (THREADS // 2)
    interleaved = []
    for holder in range(len(coalesced)):
        swapped = holder
        if bool(holder & register_group_bit) != bool(holder & high_warp_bit):
            swapped ^= register_group_bit | high_warp_bit
        interleaved.append(coalesced[swapped])
    return {name: _packet_with_copy_execution(tile_packet, _execution(logical, shared_swizzle))
            for name, logical in (('coalesced', coalesced), ('warp-interleaved', interleaved))}


def attention_copy_layout_evidence(tile_packet, *, valid_rows=TILE, storage_rows=TILE,
                                   shared_swizzle=128, destination_offsets=None):
    """Return exact holder and static transaction facts for a bound owner map."""
    from tiled_attention import _checked_tile_layouts

    if (type(valid_rows) is not int or type(storage_rows) is not int
            or storage_rows not in (8, 16, 32, TILE) or not 1 <= valid_rows <= storage_rows):
        raise ValueError('invalid Copy16 active and published row extents')
    distribution, _ = _checked_tile_layouts(tile_packet)
    native_offsets = (shared_offsets_64x64(shared_swizzle) if destination_offsets is None
                      else destination_offsets)
    plan = native_copy_plan(distribution, valid_rows, owner_policy='execution',
                            destination_offsets=native_offsets)
    origin, columns = linear(distribution['logical'])
    groups = []
    for group in plan.groups:
        active = tuple(tid for tid in range(THREADS) if (group.active_mask >> tid) & 1)
        published = tuple(tid for tid in range(THREADS) if group.rows[tid] < storage_rows)
        warps = tuple(tuple(tid for tid in active if tid // 32 == warp) for warp in range(4))
        sectors = [sum(len({(group.source_offsets[tid] + phase) // 32 for tid in warp})
                       for warp in warps) for phase in (0, 16)]
        excess = 0
        for begin in range(0, THREADS, 8):
            bank_groups = Counter((group.destination_offsets[tid] >> 4) & 7
                                  for tid in published if begin <= tid < begin + 8)
            if bank_groups:
                excess += max(bank_groups.values()) - 1
        groups.append(dict(active_vectors=len(active), initialized_vectors=len(published),
                           zero_vectors=len(published) - len(active),
                           load_warps=sum(bool(warp) for warp in warps), global_sectors32=sectors,
                           shared_excess_wavefronts=excess))
    if (sum(group['active_vectors'] for group in groups) != valid_rows * TILE // _PAYLOAD
            or sum(group['initialized_vectors'] for group in groups) != storage_rows * TILE // _PAYLOAD):
        raise ValueError('Copy16 evidence differs from the complete active and published domains')
    return dict(origin=origin, gf2_columns=columns, topology=distribution['topology'],
                packet_content_sha256=sha256(json.dumps(tile_packet, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest(),
                valid_rows=valid_rows, storage_rows=storage_rows, groups=groups, shared_swizzle=shared_swizzle,
                global_sectors32=[sum(group['global_sectors32'][phase] for group in groups)
                                  for phase in range(2)],
                shared_excess_wavefronts=sum(group['shared_excess_wavefronts'] for group in groups),
                carry_free_translates=prove_copy_schedule(plan, native_offsets=native_offsets) is not None)

