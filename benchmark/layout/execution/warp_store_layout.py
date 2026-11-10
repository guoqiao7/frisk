"""A bounded GF(2) proposal for the carrier of an aligned terminal f32 store."""
import math


def plan_warp_store(source, offsets, alignment):
    """Move one logical address direction from a lane bit to a register bit.

    The returned synthetic conversion has exactly the exported source holder
    map. The caller still verifies redistribution, the resulting warp network,
    and all integer addresses of the 16-byte stores before using the plan.
    """
    encoding = source['distributed']
    shape, topology = encoding['shape'], encoding['topology']
    if len(topology) != 5:
        raise ValueError('GF(2) writeback requires a bounded rank-2 FP32 carrier')
    registers = topology[0]
    if ('xf32' not in source['type'].split(',')[0] or len(shape) != 2
            or any(type(size) is not int or size <= 0 or size & (size - 1) for size in shape)
            or shape[1] < 4 or topology[1:] != [32, 4, 1, 1]
            or type(registers) is not int or not 4 <= registers <= 64
            or registers & (registers - 1)):
        raise ValueError('GF(2) writeback requires a bounded rank-2 FP32 carrier')
    volume = math.prod(shape)
    if volume != registers * 128 or encoding.get('replication', 1) != 1:
        raise ValueError('GF(2) writeback requires bijective holder ownership')
    logical, first = list(encoding['logical']), encoding['first']
    if (len(logical) != volume or len(first) != volume
            or any(type(slot) is not int or not 0 <= slot < volume
                   or type(first[slot]) is not int or first[slot] != holder
                   for holder, slot in enumerate(logical))):
        raise ValueError('GF(2) writeback requires exact canonical source owners')
    if (type(alignment) is not int or alignment < 16 or alignment & (alignment - 1)
            or len(offsets) != volume or 0 not in offsets
            or offsets[0] % 16
            or any(offsets.get(slot) != offsets[0] + 4 * slot for slot in range(volume))):
        raise ValueError('GF(2) writeback requires an aligned dense FP32 output view')

    base = logical[0]
    register_bits = registers.bit_length() - 1
    expanded = [base]
    for bit in range(register_bits + 7):
        column = logical[1 << bit] ^ base
        expanded += [value ^ column for value in expanded]
    if expanded != logical:
        raise ValueError('GF(2) writeback requires affine source ownership')

    # The unique preimages of the two low logical bits must both become
    # register-local. A single swap can remove exactly one lane direction.
    support = first[base ^ 1] | first[base ^ 2]
    outside = support & ~(registers - 1)
    available = (registers - 1) & ~support
    if (not 0 < outside < 32 * registers or outside & (outside - 1)
            or not available):
        raise ValueError('GF(2) writeback requires one movable lane direction')
    register_bit = available.bit_length() - 1
    lane_bit = outside.bit_length() - 1 - register_bits
    register_mask = 1 << register_bit
    swap_mask = register_mask | outside
    source_holders = [index ^ swap_mask if (bool(index & register_mask) != bool(index & outside))
                      else index for index in range(volume)]
    target_logical = [logical[index] for index in source_holders]
    target_first = [0] * volume
    for index, slot in enumerate(target_logical):
        target_first[slot] = index

    def owner(index):
        thread, register = divmod(index, registers)
        warp, lane = divmod(thread, 32)
        return dict(reg=register, lane=lane, warp=warp, warp_group=0)

    target = dict(type=source['type'], distributed=dict(shape=shape, topology=topology, replication=1,
                                                       logical=target_logical, first=target_first))
    plan = dict(mode='Shuffle', threads=128, warps=4, source_registers=registers,
                destination_registers=registers, volume=volume, scratch_bytes=0,
                payload=dict(bits=32, words=[dict(bit_offset=0, live_bits=32)]),
                source_slots=logical, global_writers=[owner(index) for index in first],
                destinations=[dict(slot=slot, source=owner(index))
                              for slot, index in zip(target_logical, source_holders)])
    return target, dict(operands=[0], results=[1], redistribution=plan), (register_bit, lane_bit)
