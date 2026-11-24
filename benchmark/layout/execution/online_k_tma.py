"""Hopper tensor-map transfers for the single-execution-group, two-version FA loop."""


def configure(e, allocation):
    if not (e.ring_stride and (e.qtile == 64 or (e.qtile == 128 and e.dim == 64 and e.threads == 256
                    and e.group > 1 and not e.causal and e.output_bytes == 4))
            and e.dim == e.value_dim and e.dim in (64,128) and e.kv_tile in (64,128)):
        raise ValueError("K TMA requires a certified two-slot K ring")
    source, target = e.kcopy["operands"]
    root = e.storage(target)["root"]
    slots = allocation["native_kv_versions"][root]
    src, dst = e.offsets(source), e.offsets(target)
    elements = e.kv_tile * e.dim
    tile_bytes = elements * 2
    # Each SW128 box covers 64 FP16 columns. The solved K layout places
    # the next column panel after all rows of the preceding TMA box.
    def shared_offset(index):
        row, column = divmod(index,e.dim)
        panel, inner = divmod(column,64)
        byte = panel*e.kv_tile*128 + row*128 + inner*2
        return byte ^ ((row & 7) << 4)
    if (len(src) != elements or len(dst) != elements or any(off % 1024 for off in slots)
            or any(src[i] != 2*i or dst[i] != shared_offset(i) for i in range(elements))):
        raise ValueError("K tensor map disagrees with the solved global or SW128 storage")
    offset = (allocation["physical_shared_bytes"] + 7) & ~7
    size = offset + 16
    if size > min(allocation["shared_limit_bytes"], 1 << 18):
        raise ValueError("insufficient shared memory for two K transaction barriers")
    e.k_tma = dict(slots=slots, barriers=[offset, offset+8], transaction_bytes=tile_bytes,
                   box=[64, e.kv_tile, 1], dimensions=[e.dim, e.kv_rows, e.heads//e.group],
                   strides=[2*e.dim, e.pitches[1]*2*e.dim], swizzle_bytes=128,
                   abi=e.k_map_abi, issuer_thread=0, phase="(kv_iteration >> 1) & 1", descriptor_arg=len(e.roots))
    return dict(allocation, physical_shared_bytes=size, static_shared_bytes=size,
                tma_barrier_bytes=16, tma_barrier_offset=offset)


def initialize(e):
    e.k_issuer = e.tmp("pred")
    e.emit(f"setp.eq.u32 {e.k_issuer}, tid, 0;")
    # A host-uploaded global map crosses proxies. A grid-constant parameter
    # is published by the launch itself. Both use the same K completion barrier.
    if e.k_tma['abi'] == 'global':
        descriptor = e.k_tma['descriptor_arg'] + 1
        e.emit(f"@{e.k_issuer} fence.proxy.tensormap::generic.acquire.sys [${descriptor}], 128;")
    e.k_barriers = []
    for offset in e.k_tma["barriers"]:
        address = e.tmp()
        e.emit(f"add.u32 {address}, smem, {offset};")
        e.emit(f"@{e.k_issuer} mbarrier.init.shared::cta.b64 [{address}], 1;")
        e.k_barriers.append(address)
    e.emit("fence.proxy.async.shared::cta;")
    e.emit("bar.sync 0;")


def issue(e, version):
    barrier = e.k_barriers[version]
    address = e.tmp()
    e.emit(f"add.u32 {address}, smem, {e.k_tma['slots'][version]};")
    size = e.k_tma["transaction_bytes"]
    if getattr(e, 'v_tma', None):
        size += e.v_tma['transaction_bytes']
    issuer = e.k_issuer
    descriptor = e.k_tma['descriptor_arg'] + 1
    e.emit(f"@{issuer} mbarrier.arrive.expect_tx.shared::cta.b64 _, [{barrier}], {size};")
    for column in range(0,e.dim,64):
        panel = address
        if column:
            panel = e.tmp()
            e.emit(f"add.u32 {panel}, {address}, {column*e.kv_tile*2};")
        e.emit(f"@{issuer} cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
               f"[{panel}], [${descriptor}, {{{column}, {e.krow}, {e.kv_head}}}], [{barrier}];")


def wait(e, index, *, label=None):
    version = e.loop_version
    phase, ready = e.tmp(), e.tmp("pred")
    barrier = e.k_barriers[version]
    e.emit(f"shr.u32 {phase}, {index}, 1;")
    e.emit(f"and.b32 {phase}, {phase}, 1;")
    label = label or f"frisk_k_ready_{version}"
    e.emit(label + ":")
    e.emit(f"mbarrier.try_wait.parity.shared::cta.b64 {ready}, [{barrier}], {phase};")
    e.emit(f"@!{ready} bra {label};")
    repair_v_tail(e, index, label)


def configure_v(e, allocation):
    """Certify a five-dimensional TMA box against the solved MN-major V map."""
    if (not e.k_tma or e.k_tma['abi'] != 'value' or e.kv_rows < 8
            or e.pitches[2] != e.kv_rows):
        raise ValueError('V TMA requires at least one complete eight-row group and a packed V head')
    source, target = e.vcopy['operands']
    slots = allocation['native_kv_versions'][e.storage(target)['root']]
    src, dst = e.offsets(source), e.offsets(target)
    columns=e.value_dim
    elements = columns * e.kv_tile
    if len(src) != elements or len(dst) != elements or any(offset % 256 for offset in slots):
        raise ValueError('V tile extent or SW32 slot alignment disagrees with the tensor map')
    desc = e.pv_op['attrs']['frisk.mma_contract']['b_descriptor']
    if desc['swizzle'] == 128 and desc['major'] == 'mn' and columns == 64:
        if any(slot % 1024 for slot in slots) or any(
                src[i] != 2*i or dst[i] != ((2*i) ^ (((i // columns) & 7) << 4))
                for i in range(elements)):
            raise ValueError('V tensor map disagrees with the solved row-major SW128 layout')
        return dict(slots=slots, dimensions=[columns, e.kv_rows, e.heads // e.group],
                    strides=[2*columns, 2*columns*e.pitches[2]], box=[columns, e.kv_tile, 1],
                    swizzle_bytes=128, abi='value', transaction_bytes=2*elements,
                    descriptor_arg=len(e.roots)+1)
    for i in range(elements):
        row, col = divmod(i, columns)
        # Box order is col%16, row%8, col//16, row//8, head.
        byte = 2 * ((row // 8 * (columns//16) + col // 16) * 128 + row % 8 * 16 + col % 16)
        if src[i] != 2*i or dst[i] != (byte ^ ((byte >> 7 & 1) << 4)):
            raise ValueError('V tensor map disagrees with the solved copy or SW32 storage')
    plan = dict(slots=slots, dimensions=[16, 8, columns//16, e.kv_rows // 8, e.heads // e.group],
                strides=[2*columns, 32, 16*columns, 2*columns * e.pitches[2]],
                box=[16, 8, columns//16, e.kv_tile // 8, 1], swizzle_bytes=32, abi='value',
                transaction_bytes=2*elements, descriptor_arg=len(e.roots)+1)

    if e.kv_rows % 8:
        plan['tail_rows'] = e.kv_rows % 8
    return plan


def repair_v_tail(e, index, label):
    """The truncated TMA box zero-fills the last group. Overwrite valid vectors after completion."""
    plan = getattr(e, 'v_tma', None)
    if not plan or not plan.get('tail_rows'):
        return
    last_index = (e.kv_rows - 1) // e.kv_tile
    if e.loop_version != last_index % 2:
        return
    tail_rows = plan['tail_rows']
    first_row = e.kv_rows - tail_rows
    skip, active = e.tmp('pred'), e.tmp('pred')
    done = label + '_v_tail_done'
    e.emit(f'setp.ne.u32 {skip}, {index}, {last_index};')
    e.emit(f'@{skip} bra.uni {done};')
    # Each participating thread owns 16 contiguous bytes. Never form a
    # memory transaction for rows in the following head, including the last head.
    e.emit(f'setp.lt.u32 {active}, tid, {e.value_dim//8*tail_rows};')
    row, address, vector_offset = e.tmp(), e.tmp('b64'), e.tmp()
    e.emit(f'add.u32 {row}, {e.vbase}, {first_row};')
    e.emit(f'mad.wide.u32 {address}, {row}, {2*e.value_dim}, ${e.roots[e.vroot]+1};')
    e.emit(f'shl.b32 {vector_offset}, tid, 4;')
    e.emit(f'mad.wide.u32 {address}, {vector_offset}, 1, {address};')
    offsets = e.offsets(e.vcopy['operands'][1])
    start = first_row % e.kv_tile * e.value_dim
    within = e.map([offsets[start + (thread % e.value_dim)*8] for thread in range(e.threads)])
    dest = e.tmp()
    e.emit(f'add.u32 {dest}, smem, {plan["slots"][e.loop_version]};')
    e.emit(f'add.u32 {dest}, {dest}, {within};')
    vector = ', '.join(e.tmp() for _ in range(4))
    e.emit(f'@{active} ld.global.v4.b32 {{{vector}}}, [{address}];')
    e.emit(f'@{active} st.shared.v4.b32 [{dest}], {{{vector}}};')
    e.emit(done + ':')


def issue_v(e, version):
    plan = e.v_tma
    if plan['swizzle_bytes'] == 128:
        dst = e.tmp()
        desc = plan['descriptor_arg'] + 1
        e.emit(f"add.u32 {dst}, smem, {plan['slots'][version]};")
        e.emit(f"@{e.k_issuer} cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
               f"[{dst}], [${desc}, {{0, {e.krow}, {e.kv_head}}}], [{e.k_barriers[version]}];")
        return
    group, dst = e.tmp(), e.tmp()
    e.emit(f"shr.u32 {group}, {e.krow}, 3;")
    e.emit(f"add.u32 {dst}, smem, {plan['slots'][version]};")
    desc = plan['descriptor_arg'] + 1
    e.emit(f"@{e.k_issuer} cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes "
           f"[{dst}], [${desc}, {{0, 0, 0, {group}, {e.kv_head}}}], [{e.k_barriers[version]}];")
