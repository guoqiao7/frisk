"""Exact same-warp lowering for the exported canonical Frisk reduction tree."""

from dataclasses import dataclass
import math

from execution_support import LogicalTable, normalize_distribution


_WARP = 32
_MAX_POINTS = 65536


def _power(value):
    return type(value) is int and value > 0 and not value & (value - 1)


def _holder(index, topology):
    coordinates = []
    for extent in topology:
        index, digit = divmod(index, extent)
        coordinates.append(digit)
    return tuple(reversed(coordinates))


@dataclass(frozen=True)
class _Layout:
    shape: tuple
    topology: tuple
    logical: tuple
    first: tuple


def _layout(distribution):
    if type(distribution) is not dict:
        raise ValueError('invalid reduction distribution')
    topology, shape = distribution.get('topology'), distribution.get('shape')
    if (type(topology) is not list or len(topology) != 5 or not all(map(_power, topology))
            or topology[1] != _WARP or topology[4] != 1
            or type(shape) is not list or not shape or not all(map(_power, shape))):
        raise ValueError('invalid reduction layout geometry')
    holders, volume = math.prod(topology), math.prod(shape)
    if max(holders, volume) > _MAX_POINTS:
        raise ValueError('reduction layout exceeds the execution proof bound')
    try:
        distribution = normalize_distribution(distribution)
    except (KeyError, IndexError, TypeError, OverflowError) as error:
        raise ValueError('malformed symbolic reduction layout') from error
    logical, first = distribution.get('logical'), distribution.get('first')
    if (type(logical) not in (list, LogicalTable) or len(logical) != holders
            or type(first) is not list or len(first) != volume):
        raise ValueError('incomplete reduction holder table')
    logical = tuple(logical)
    expected = [-1] * volume
    counts = [0] * volume
    for hardware, slot in enumerate(logical):
        if type(slot) is not int or not 0 <= slot < volume:
            raise ValueError('reduction layout has an invalid logical coordinate')
        if counts[slot] == 0:
            expected[slot] = hardware
        counts[slot] += 1
    replication = distribution.get('replication')
    if (type(replication) is not int or replication <= 0 or any(c != replication for c in counts)
            or first != expected or any(type(x) is not int for x in first)):
        raise ValueError('reduction layout has invalid first-owner or replication')
    return _Layout(tuple(shape), tuple(topology), logical, tuple(first))


def _expect_holder(actual, expected):
    return (type(actual) is list and len(actual) == 5
            and all(type(x) is int for x in actual) and tuple(actual) == expected)


def _expect_node(node, logical, left, right, owner):
    if (type(node) is not dict or set(node) != {'logicalInput', 'left', 'right', 'holder'}
            or any(type(node[k]) is not int for k in ('logicalInput', 'left', 'right'))
            or (node['logicalInput'], node['left'], node['right']) != (logical, left, right)
            or not _expect_holder(node['holder'], owner)):
        raise ValueError('noncanonical reduction node or first-owner holder')


def _tree(leaves):
    layer = tuple(leaves)
    while len(layer) > 1:
        layer = tuple(('combine', layer[i], layer[i + 1]) for i in range(0, len(layer), 2))
    return layer[0]


@dataclass(frozen=True)
class _Panel:
    lanes: tuple
    leaf_lanes: tuple
    leaf_registers: tuple


def _affine_offsets(table):
    base = table[0]
    offsets = [0]
    for bit in range(5):
        column = table[1 << bit] ^ base
        offsets += [offset ^ column for offset in offsets]
    if any(value != base ^ offset for value, offset in zip(table, offsets)):
        return None
    return tuple(offsets)


@dataclass(frozen=True)
class _VerifiedReduction:
    scope: str
    source_registers: int
    source_projection: tuple
    single_warpgroup: bool
    panels: tuple

    def eager_registers(self):
        """Return the original ordered AST with uniform absolute source-lane tables."""
        if not self.single_warpgroup or self.scope == 'cta_shared_tree':
            return None
        roots = []
        for panels in self.panels:
            lanes = [None] * _WARP
            for panel in panels:
                for lane in panel.lanes:
                    prior = lanes[lane]
                    if prior is not None and (prior.leaf_lanes != panel.leaf_lanes
                                              or prior.leaf_registers != panel.leaf_registers):
                        return None
                    lanes[lane] = panel
            registers = lanes[0].leaf_registers
            if any(panel.leaf_registers != registers for panel in lanes):
                return None
            leaves = []
            for index, register in enumerate(registers):
                table = tuple(panel.leaf_lanes[index] for panel in lanes)
                if _affine_offsets(table) is None:
                    return None
                leaves.append(('leaf', register, table))
            roots.append(_tree(leaves))
        return tuple(roots)


def _verify_projected_result(source, result, projection):
    # Reconstruct the C++ ordered basis of the projected register columns.
    # The result keeps lane/warp coordinates and drops dependent register bits.
    basis = {}
    register_offsets = [0]
    for bit in range(source.topology[0].bit_length() - 1):
        position = 1 << bit
        column = projection[position] ^ projection[0]
        while column:
            pivot = column & -column
            if pivot not in basis:
                basis[pivot] = column
                register_offsets += [x | position for x in register_offsets]
                break
            column ^= basis[pivot]
    if len(register_offsets) != result.topology[0]:
        raise ValueError('reduction result does not have the projected register basis')
    expected = (projection[base + offset]
                for base in range(0, len(projection), source.topology[0]) for offset in register_offsets)
    if any(a != b for a, b in zip(result.logical, expected)):
        raise ValueError('reduction result differs from the natural projection')


def verified_warp_reduction_trees(operation, source_distribution, result_distribution):
    """Reconstruct every leaf, edge, and broadcast of canonical_fiber_tree_v1.

    Malformed input raises ValueError. Valid certificates outside the compact
    warp strategies remain valid, with no optimized or eager warp plan.
    """
    source = _layout(source_distribution)
    result = _layout(result_distribution)
    if type(operation) is not dict or operation.get('name') != 'frisk.reduce_tensor':
        raise ValueError('expected a Frisk reduction operation')
    attrs = operation.get('attrs')
    if type(attrs) is not dict:
        raise ValueError('missing reduction attributes')
    axis, kind = attrs.get('dim'), attrs.get('kind')
    if type(kind) is str:
        kind = kind.strip('"')
    if (type(axis) is not int or not 0 <= axis < len(source.shape) or len(source.shape) < 2
            or kind not in ('max', 'sum') or source.topology[1:] != result.topology[1:]
            or source.shape[:axis] + source.shape[axis + 1:] != result.shape):
        raise ValueError('unsupported reduction axis, kind, or result geometry')
    contract = attrs.get('frisk.reduction_contract')
    expected = dict(algorithm='canonical_fiber_tree_v1', axis=axis, dtype='f32',
                    input_policy='first_owner', kind=kind, output_policy='broadcast_complete',
                    result_shape=list(result.shape), source_shape=list(source.shape), target='sm_90a',
                    threads=math.prod(source.topology[1:4]), version=1)
    if (type(contract) is not dict or set(contract) != set(expected) | {'scope'}
            or any(type(contract[k]) is not type(v) or contract[k] != v for k, v in expected.items())
            or any(type(x) is not int for key in ('source_shape', 'result_shape') for x in contract[key])):
        raise ValueError('invalid canonical Frisk reduction contract')

    extent = source.shape[axis]
    suffix = math.prod(source.shape[axis + 1:])
    projected = tuple((x // (extent * suffix)) * suffix + x % suffix for x in source.logical)
    _verify_projected_result(source, result, projected)
    leaves = [[] for _ in result.first]
    for hardware, x in enumerate(source.logical):
        if source.first[x] == hardware:
            leaves[projected[hardware]].append((x, _holder(hardware, source.topology)))
    broadcasts = [[] for _ in result.first]
    for hardware, y in enumerate(result.logical):
        broadcasts[y].append(_holder(hardware, result.topology))

    fibers = operation.get('fibers')
    if type(fibers) is not list or len(fibers) != len(leaves):
        raise ValueError('incomplete canonical reduction fibers')
    panels = [[] for _ in range(result.topology[0])]
    scope = 0

    def edge_scope(left, right):
        if left[:3] != right[:3]:
            return 2
        return int(left[3] != right[3])

    for y, (fiber, owners, destinations) in enumerate(zip(fibers, leaves, broadcasts)):
        if (type(fiber) is not dict or set(fiber) != {'logicalOutput', 'nodes', 'root', 'broadcasts'}
                or type(fiber['logicalOutput']) is not int or fiber['logicalOutput'] != y
                or type(fiber['root']) is not int or fiber['root'] != 2 * extent - 2
                or type(fiber['nodes']) is not list or len(fiber['nodes']) != 2 * extent - 1
                or type(fiber['broadcasts']) is not list
                or len(fiber['broadcasts']) != len(destinations)):
            raise ValueError('incomplete or malformed canonical reduction fiber')
        for actual, destination in zip(fiber['broadcasts'], destinations):
            if not _expect_holder(actual, destination):
                raise ValueError('noncanonical reduction broadcast holder')
        nodes = fiber['nodes']
        holders = []
        for i, (logical, owner) in enumerate(owners):
            _expect_node(nodes[i], logical, -1, -1, owner)
            holders.append(owner)
        layer = tuple(range(extent))
        while len(layer) > 1:
            merged = []
            for left, right in zip(layer[::2], layer[1::2]):
                owner = holders[left]
                index = len(holders)
                _expect_node(nodes[index], -1, left, right, owner)
                scope = max(scope, edge_scope(owner, holders[right]))
                holders.append(owner)
                merged.append(index)
            layer = tuple(merged)
        owner_lanes = tuple(owner[3] for _, owner in owners)
        owner_registers = tuple(owner[4] for _, owner in owners)
        by_register = {}
        for destination in destinations:
            scope = max(scope, edge_scope(holders[-1], destination))
            by_register.setdefault((*destination[:3], destination[4]), []).append(destination[3])
        for (*_, register), lanes in by_register.items():
            panels[register].append(_Panel(tuple(lanes), owner_lanes, owner_registers))

    scope_name = ('register', 'warp', 'cta_shared_tree')[scope]
    if contract['scope'] != scope_name:
        raise ValueError('reduction scope differs from verified communication')
    return _VerifiedReduction(scope_name, source.topology[0], projected, source.topology[3] == 1,
                              tuple(tuple(group) for group in panels))


@dataclass(frozen=True)
class ButterflyStage:
    xor_mask: int
    parity_mask: int
    left_parity: int


@dataclass(frozen=True)
class WarpReductionRegister:
    local_tree: tuple
    stages: tuple


@dataclass(frozen=True)
class WarpReductionPlan:
    registers: tuple


def _lane_kernel(proof):
    # A fiber is the coset of the kernel of lane -> projected output. Prove
    # the same linear part for every source register and warp before using it.
    registers = proof.source_registers
    projected = proof.source_projection
    offsets = _affine_offsets(projected[:_WARP * registers:registers])
    if offsets is None:
        return None
    for warp_start in range(0, len(projected), _WARP * registers):
        for register in range(registers):
            start = warp_start + register
            base = projected[start]
            if any(projected[start + lane * registers] != base ^ offset
                   for lane, offset in enumerate(offsets)):
                return None
    return tuple(lane for lane, offset in enumerate(offsets) if offset == 0)


def plan_ordered_warp_reduction(proof):
    """Factor a verified ordered tree into lane-local trees and XOR stages.

    Every lane panel must contain identical local register sequences, and
    the butterfly stages must reproduce the canonical ascending owner order.
    No commutativity or reassociation of the floating-point operation is used.
    """
    if not proof.single_warpgroup or proof.scope == 'cta_shared_tree':
        return None
    kernel = _lane_kernel(proof)
    if not kernel:
        return None
    size = len(kernel)
    stages = tuple(ButterflyStage(mask, 1 << (mask.bit_length() - 1), 0)
                   for mask in (kernel[1 << bit] for bit in range(size.bit_length() - 1)))
    plans = []
    for panels in proof.panels:
        local_registers = None
        for panel in panels:
            lanes = panel.lanes
            if len(lanes) != size or tuple(sorted(lanes[0] ^ mask for mask in kernel)) != lanes:
                return None
            count, remainder = divmod(len(panel.leaf_lanes), size)
            if remainder or not _power(count):
                return None
            registers = panel.leaf_registers[:count]
            if (local_registers is not None and registers != local_registers
                    or panel.leaf_registers != registers * size
                    or panel.leaf_lanes != tuple(lane for lane in lanes for _ in range(count))):
                return None
            for bit, stage in enumerate(stages):
                if any(lanes[index ^ (1 << bit)] != lane ^ stage.xor_mask
                       or int(bool(lane & stage.parity_mask)) != ((index >> bit) & 1)
                       for index, lane in enumerate(lanes)):
                    return None
            local_registers = registers
        plans.append(WarpReductionRegister(_tree(('leaf', register) for register in local_registers), stages))
    return WarpReductionPlan(tuple(plans))
