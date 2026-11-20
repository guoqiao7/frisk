"""Verified warp-local GF(2) pullback networks; no CUDA or Triton dependency.

Independent implementation of elementary binary-linear transformations. References:
Linear Layouts, section 5.4, https://arxiv.org/abs/2505.23819 ; and the decomposition
ideas in Triton https://github.com/triton-lang/triton/tree/1887644bdea84fde07825813158782fa526871fd .
No Triton source code is copied. Complete canonical holder maps, rather than only
source support sets, are checked. Replication/non-bijective maps retain the exact
support lowering. Counts describe the documented PTX grammar, not GPU latency.
"""

CONTRACT = 'gf2-shuffle-network-v1'
IDENTITY_LANE = [1 << i for i in range(5)]


def _apply(columns, value):
    result = 0
    while value:
        bit = value & -value
        result ^= columns[bit.bit_length() - 1]
        value ^= bit
    return result


def _rank(columns):
    pivots = {}
    for value in columns:
        while value:
            bit = value.bit_length() - 1
            if bit not in pivots:
                pivots[bit] = value
                break
            value ^= pivots[bit]
    return len(pivots)


def _rows(columns, size):
    return [sum(((column >> i) & 1) << j for j, column in enumerate(columns))
            for i in range(size)]


def _fit(plan):
    if not isinstance(plan, dict) or plan.get('mode') != 'Shuffle':
        return None
    registers = plan.get('source_registers')
    if (type(registers) is not int or not 1 <= registers <= 512 or
            registers & (registers - 1) or plan.get('destination_registers') != registers or
            type(plan.get('destination_registers')) is not int or
            plan.get('threads') != 128 or plan.get('warps') != 4 or
            plan.get('payload') != dict(bits=32, words=[dict(bit_offset=0, live_bits=32)])):
        return None
    entries = plan.get('destinations')
    if not isinstance(entries, list) or len(entries) != 128 * registers:
        return None
    holders = []
    for index, entry in enumerate(entries):
        owner = entry['source']
        if (any(type(owner[k]) is not int for k in ('reg', 'lane', 'warp', 'warp_group')) or
                not 0 <= owner['reg'] < registers or not 0 <= owner['lane'] < 32 or
                owner['warp'] != index // (32 * registers) or owner['warp_group'] != 0):
            return None
        holders.append((owner['warp'] * 32 + owner['lane']) * registers + owner['reg'])
    register_bits = registers.bit_length() - 1
    dimension = register_bits + 5
    low_mask = (32 * registers) - 1
    low = [index & low_mask for index in holders]
    base = low[0]
    columns = [low[1 << bit] ^ base for bit in range(register_bits + 7)]
    expanded = [base]
    for column in columns:
        expanded += [value ^ column for value in expanded]
    if expanded != low or _rank(columns[:dimension]) != dimension:
        return None
    offsets = [low[warp << dimension] for warp in range(4)]
    return registers, register_bits, dimension, base, columns, offsets, holders


def _append(steps, step, registers):
    op = step['op']
    if op == 'reg_select' and not step['thread_mask']:
        if not step['constant']:
            return
        step = dict(op='reg_permute', permutation=[r ^ (1 << step['register_bit'])
                                                 for r in range(registers)])
        op = 'reg_permute'
    if op == 'reg_permute' and step['permutation'] == list(range(registers)):
        return
    if (op == 'lane' and step['columns'] == IDENTITY_LANE and
            not any(step['offsets']) and not any(step['register_masks'])):
        return
    steps.append(step)


def _lane(columns, registers, offsets=None, masks=None):
    return dict(op='lane', columns=list(columns), offsets=list(offsets or [0] * 4),
                register_masks=list(masks or [0] * registers))


def _offset_steps(registers, register_bits, offsets):
    steps = []
    for bit in range(register_bits):
        constant = (offsets[0] >> bit) & 1
        mask = (((offsets[1] ^ offsets[0]) >> bit) & 1) * 32
        mask |= (((offsets[2] ^ offsets[0]) >> bit) & 1) * 64
        _append(steps, dict(op='reg_select', register_bit=bit,
                           thread_mask=mask, constant=constant), registers)
    _append(steps, _lane(IDENTITY_LANE, registers,
                        offsets=[value >> register_bits for value in offsets]), registers)
    return steps


def instruction_counts(steps, registers):
    """Exact counts for the backend grammar, including shared predicate setup.

    Single-bit predicates use and+setp; general parity uses and+popc+and+setp.
    A nonconstant lane base uses Emitter.map once, plus one xor for each nonzero
    register mask. Pure xor lanes use immediate bfly with zero masks aliased.
    """
    shfl = selects = maps = 0
    predicates = set()
    for step in steps:
        op = step['op']
        if op == 'reg_permute':
            continue
        if op in ('reg_select', 'mixed_swap'):
            mask = step['thread_mask'] if op == 'reg_select' else 1 << step['lane_bit']
            constant = step['constant'] if op == 'reg_select' else 0
            key = (mask, constant)
            if key not in predicates:
                maps += 2 if mask.bit_count() == 1 else 4
                predicates.add(key)
            if op == 'reg_select':
                selects += registers
            else:
                shfl += registers // 2
                selects += 3 * registers // 2
        elif op == 'lane':
            columns, offsets, masks = step['columns'], step['offsets'], step['register_masks']
            if columns == IDENTITY_LANE and len(set(offsets)) == 1:
                shfl += sum(bool(offsets[0] ^ mask) for mask in masks)
            else:
                shfl += registers
                maps += 1 + 4 * (sum(bool(c) for c in columns) +
                                 bool(offsets[1] ^ offsets[0]) + bool(offsets[2] ^ offsets[0]))
                maps += sum(bool(mask) for mask in masks)
        else:
            raise ValueError('unknown shuffle network step: ' + str(op))
    return dict(shfl=shfl, selects=selects, map_instructions=maps,
                total_work=shfl + selects + maps,
                shuffle_warp_issues=4 * shfl,
                address_alu_warp_issues=4 * (selects + maps))


def simulate_network(network, values=None):
    """Execute the specified *steps*, not the fitted map, on every holder.

    Unique holder IDs are the default payload, proving arbitrary-word transport.
    The emitter runs steps forward. Composed pullbacks are S1*S2*...*Sn; the
    equivalent coordinate evaluation would visit the elementary steps in reverse.
    """
    registers = network['registers']
    count = 128 * registers
    data = list(range(count)) if values is None else list(values)
    if len(data) != count:
        raise ValueError('shuffle network payload extent')
    for step in network['steps']:
        op = step['op']
        out = []
        if op == 'reg_permute':
            permutation = step['permutation']
            for thread in range(128):
                start = thread * registers
                out.extend(data[start + r] for r in permutation)
        elif op == 'reg_select':
            bit, mask, constant = step['register_bit'], step['thread_mask'], step['constant']
            for thread in range(128):
                start = thread * registers
                flip = (((thread & mask).bit_count() & 1) ^ constant) << bit
                out.extend(data[start + (r ^ flip)] for r in range(registers))
        elif op == 'mixed_swap':
            register_bit, lane_bit = step['register_bit'], step['lane_bit']
            for thread in range(128):
                lane = thread % 32
                lane_value = (lane >> lane_bit) & 1
                for r in range(registers):
                    register_value = (r >> register_bit) & 1
                    delta = lane_value ^ register_value
                    source_thread = thread ^ (delta << lane_bit)
                    source_register = r ^ (delta << register_bit)
                    out.append(data[source_thread * registers + source_register])
        elif op == 'lane':
            columns, offsets, masks = step['columns'], step['offsets'], step['register_masks']
            for thread in range(128):
                warp = thread // 32
                base = _apply(columns, thread % 32) ^ offsets[warp]
                out.extend(data[(warp * 32 + (base ^ masks[r])) * registers + r]
                           for r in range(registers))
        else:
            raise ValueError('unknown shuffle network step: ' + str(op))
        data = out
    return data


def _permutation_steps(registers, register_bits, columns, offset_steps):
    dimension = register_bits + 5
    rows = _rows(columns[:dimension], dimension)
    positions = [row.bit_length() - 1 for row in rows]
    crossing_registers = [i for i in range(register_bits) if positions[i] >= register_bits]
    crossing_lanes = [j for j in range(5) if positions[register_bits + j] < register_bits]
    assert len(crossing_registers) == len(crossing_lanes)
    pairs = list(zip(crossing_registers, crossing_lanes))
    block_rows = rows.copy()
    for i, j in pairs:
        block_rows[i], block_rows[register_bits + j] = block_rows[register_bits + j], block_rows[i]
    register_permutation = [sum(((r >> (block_rows[i].bit_length() - 1)) & 1) << i
                                for i in range(register_bits)) for r in range(registers)]
    lane_rows = [row >> register_bits for row in block_rows[register_bits:]]
    lane_columns = _rows(lane_rows, 5)
    ship = [dict(step) for step in offset_steps]
    for i, j in pairs:
        _append(ship, dict(op='mixed_swap', register_bit=i, lane_bit=j), registers)
    _append(ship, dict(op='reg_permute', permutation=register_permutation), registers)
    _append(ship, _lane(lane_columns, registers), registers)
    fused = [dict(step) for step in offset_steps]
    for i, j in pairs:
        _append(fused, dict(op='reg_select', register_bit=i,
                            thread_mask=1 << j, constant=0), registers)
    masks = [sum(((r >> i) & 1) << j for i, j in pairs) for r in range(registers)]
    _append(fused, _lane(lane_columns, registers, masks=masks), registers)
    for i, j in pairs:
        lane_control = lane_rows[j].bit_length() - 1
        _append(fused, dict(op='reg_select', register_bit=i,
                            thread_mask=1 << lane_control, constant=0), registers)
    _append(fused, dict(op='reg_permute', permutation=register_permutation), registers)
    return [('bit-permutation-ship', ship), ('bit-permutation-fused', fused)]


def _linear_steps(registers, register_bits, columns, offset_steps):
    dimension = register_bits + 5
    rows = _rows(columns[:dimension], dimension)
    steps = [dict(step) for step in offset_steps]
    def swap(i, j):
        if i < register_bits and j < register_bits:
            permutation = [r ^ ((((r >> i) ^ (r >> j)) & 1) * ((1 << i) | (1 << j)))
                           for r in range(registers)]
            _append(steps, dict(op='reg_permute', permutation=permutation), registers)
        elif i >= register_bits and j >= register_bits:
            lane_columns = IDENTITY_LANE.copy()
            lane_columns[i - register_bits], lane_columns[j - register_bits] = lane_columns[j - register_bits], lane_columns[i - register_bits]
            _append(steps, _lane(lane_columns, registers), registers)
        else:
            if i >= register_bits:
                i, j = j, i
            _append(steps, dict(op='mixed_swap', register_bit=i, lane_bit=j - register_bits), registers)
    def add(target, control):
        if target < register_bits and control < register_bits:
            _append(steps, dict(op='reg_permute', permutation=[r ^ (((r >> control) & 1) << target)
                                                              for r in range(registers)]), registers)
        elif target >= register_bits and control >= register_bits:
            lane_columns = IDENTITY_LANE.copy()
            lane_columns[control - register_bits] ^= 1 << (target - register_bits)
            _append(steps, _lane(lane_columns, registers), registers)
        elif target < register_bits:
            _append(steps, dict(op='reg_select', register_bit=target,
                                thread_mask=1 << (control - register_bits), constant=0), registers)
        else:
            masks = [((r >> control) & 1) << (target - register_bits) for r in range(registers)]
            _append(steps, _lane(IDENTITY_LANE, registers, masks=masks), registers)
    # If E_n ... E_1 A=I, then A=E_1 ... E_n: emitter/pullback order is
    # elimination order, not reversed elimination order.
    for column in range(dimension):
        pivot = next(row for row in range(column, dimension) if rows[row] >> column & 1)
        if pivot != column:
            swap(column, pivot)
            rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(dimension):
            if row != column and rows[row] >> column & 1:
                add(row, column)
                rows[row] ^= rows[column]
    assert rows == [1 << i for i in range(dimension)]
    return [('linear-shear', steps)]


def shuffle_network_candidates(plan):
    """Return independently full-holder-verified network alternatives.

    Invalid/ineligible input is a normal unsupported result, not a numeric cost.
    The caller still validates the redistribution's logical ownership proof.
    """
    try:
        fitted = _fit(plan)
    except (KeyError, IndexError, TypeError, ValueError):
        return []
    if fitted is None:
        return []
    registers, register_bits, dimension, base, columns, offsets, holders = fitted
    prefix = _offset_steps(registers, register_bits, offsets)
    monomial = all(value and value & (value - 1) == 0 for value in columns[:dimension])
    alternatives = (_permutation_steps if monomial else _linear_steps)(registers, register_bits, columns, prefix)
    thread_columns = columns[register_bits:]
    source_support = 1 << _rank([column & (registers - 1) for column in thread_columns])
    register_map = 1 + 4 * sum(bool(column & (registers - 1)) for column in thread_columns)
    lane_map = 1 + 4 * sum(bool(column >> register_bits) for column in thread_columns)
    legacy_work = registers * (register_map + lane_map + 3 * source_support - 1)
    results = []
    for kind, steps in alternatives:
        candidate = dict(contract=CONTRACT, registers=registers, kind=kind, steps=steps,
                         counts=instruction_counts(steps, registers), legacy_work=legacy_work,
                         legacy_shfl=registers * source_support, verified_holders=128 * registers,
                         affine_base=base, affine_columns=columns)
        if simulate_network(candidate) != holders:
            raise AssertionError('shuffle network decomposition failed full-holder proof: ' + kind)
        results.append(candidate)
    return results


def plan_shuffle_network(plan, *, objective='min-shfl'):
    """Minimize communication subject to both exact-support work bounds.

    This deterministic policy is a work bound and communication objective within
    these alternatives, not an optimality or latency claim over all algorithms.
    """
    if type(objective) is not str or objective not in ('min-shfl', 'min-work'):
        raise ValueError('unknown shuffle network objective')
    candidates = [candidate for candidate in shuffle_network_candidates(plan)
                  if candidate['counts']['total_work'] <= candidate['legacy_work']
                  and candidate['counts']['shfl'] <= candidate['legacy_shfl']]
    order = ('shfl', 'total_work') if objective == 'min-shfl' else ('total_work', 'shfl')
    return min(candidates, key=lambda candidate: (candidate['counts'][order[0]],
                                                candidate['counts'][order[1]],
                                                candidate['kind'])) if candidates else None


def network_resource_proxy(network):
    """Count emitted definitions, not physical registers or resource fit."""
    if network is None:
        return None
    registers = network['registers']
    predicates = set()
    temporaries = 0
    stages = 0
    for step in network['steps']:
        op = step['op']
        if op == 'reg_permute':
            continue
        if op in ('reg_select', 'mixed_swap'):
            key = ((step['thread_mask'], step['constant']) if op == 'reg_select'
                   else (1 << step['lane_bit'], 0))
            if key not in predicates:
                predicates.add(key)
                temporaries += 1
            temporaries += registers if op == 'reg_select' else 2 * registers
            if op == 'mixed_swap':
                stages += 1
        elif op == 'lane':
            columns, offsets, masks = step['columns'], step['offsets'], step['register_masks']
            immediate = columns == IDENTITY_LANE and len(set(offsets)) == 1
            if immediate:
                active = sum(bool(offsets[0] ^ mask) for mask in masks)
                temporaries += active
                stages += bool(active)
            else:
                thread_columns = columns + [offsets[1] ^ offsets[0], offsets[2] ^ offsets[0]]
                temporaries += 1 + sum(bool(value) for value in thread_columns)
                temporaries += sum(bool(mask) for mask in masks) + registers
                stages += bool(registers)
        else:
            raise ValueError('unknown shuffle network resource step')
    return dict(contract='emitted-network-definition-proxy-v1', logical_carrier_words=registers,
                b32_temporary_definitions=temporaries, predicate_definitions=len(predicates),
                communication_stages=stages, physical_register_prediction=False)
