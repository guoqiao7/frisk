"""Two versions of the verified K/V tiles for a single consumer warpgroup."""
from native_online_memory import _packed_groups, issue_packed_copy


def prepare_ring(e, allocation):
    """Place both new versions beyond the original roots, scratch and padding."""
    groups = [_packed_groups(e, copy) for copy in (e.kcopy, e.vcopy)]
    if not all(groups) or allocation['tma_barrier_bytes']:
        return allocation
    roots = [e.storage(copy['operands'][1])['root'] for copy in (e.kcopy, e.vcopy)]
    slots = {item['root_id']: item for item in allocation['allocations']}
    if len(set(roots)) != 2 or any(root not in slots for root in roots):
        return allocation
    first = min(slots[root]['offset'] for root in roots)
    # Keep the WGMMA swizzle phase and any stronger root alignment.
    alignment = max(1024, *(slots[root]['alignment'] for root in roots))
    end = allocation['physical_shared_bytes']
    stride = ((end - first + alignment - 1) // alignment) * alignment
    versions = {root: [slots[root]['offset'], slots[root]['offset'] + stride] for root in roots}
    size = max(versions[root][1] + slots[root]['bytes'] for root in roots)
    if size > allocation['shared_limit_bytes'] or size >= (1 << 18):
        return allocation
    e.ring_groups = groups
    e.ring_roots = set(roots)
    e.ring_stride = stride
    return dict(allocation, native_kv_versions=versions, native_ring_extra_bytes=size-end,
                static_shared_bytes=size, physical_shared_bytes=size, tma_barrier_offset=size)


def issue_tile(e, index, version, *, copy_k=True, copy_v=True):
    """Issue one KV transaction group. Tails use the existing zero-fill proof."""
    original_row = e.krow
    e.krow = e.tmp()
    e.emit(f'mul.lo.u32 {e.krow}, {index}, {e.kv_tile};')
    e.copy_version = version
    for root, base, groups, dim in ((e.k, e.kbase, e.ring_groups[0], e.dim),
                                    (e.vroot, e.vbase, e.ring_groups[1], e.value_dim)):
        if (root == e.k and not copy_k) or (root == e.vroot and not copy_v):
            continue
        if root == e.vroot and getattr(e, 'v_tma', None):
            from online_k_tma import issue_v
            issue_v(e, version)
            continue
        if root == e.k and e.k_tma:
            from online_k_tma import issue
            issue(e, version)
            continue
        row = e.tmp()
        e.emit(f'add.u32 {row}, {base}, {e.krow};')
        e.set_pointer(root, row, 2*dim)
        for group in groups:
            issue_packed_copy(e, group)
    e.copy_version = 0
    e.krow = original_row
    if copy_v and not getattr(e, 'v_tma', None):
        e.emit('cp.async.commit_group;')


def enter_iteration(e, index):
    # All consumers completed PV on the other slot before this CTA rendezvous.
    if getattr(e, 'cross_iteration_pv', False) or getattr(e, 'prefetch_after_qk', False):
        from online_k_tma import wait
        wait(e, index, label='frisk_k_prime' if e.loop_prime else None)
        return
    e.emit('cp.async.wait_group 0;')
    if e.k_tma:
        from online_k_tma import wait
        wait(e, index)
    e.emit('bar.sync 0;')
    e.emit('fence.proxy.async.shared::cta;')
    if getattr(e, 'kv_prefetch_edges', None) is None:
        prefetch_next(e, index)
    else:
        issue_at_qk_edge(e, index, 'before')


def next_tile(e, index):
    ahead, last = e.tmp(), e.tmp('pred')
    e.emit(f'add.u32 {ahead}, {index}, 1;')
    e.emit(f'setp.ge.u32 {last}, {ahead}, {e.kv_bound};')
    return ahead, last


def prefetch_next(e, index):
    ahead, last = next_tile(e, index)
    issue_guarded(e, ahead, last)


def issue_guarded(e, ahead, last, *, copy_k=True, copy_v=True):
    done = 'frisk_ring_prime' if getattr(e,'loop_prime',False) else f'frisk_ring_ready_{e.loop_version}'
    if not copy_v:
        done += '_k'
    e.emit(f'@{last} bra.uni {done};')
    issue_tile(e, ahead, 1-e.loop_version, copy_k=copy_k, copy_v=copy_v)
    e.emit(done+':')


def issue_at_qk_edge(e, index, edge):
    # Both current operands are ready and all consumers have released the other
    # slot. Only the issue point of the next tile changes at these two edges.
    k_edge, v_edge = e.kv_prefetch_edges
    if k_edge == v_edge:
        if edge == k_edge:
            prefetch_next(e, index)
        return

    if edge == 'before':
        e.next_kv_tile = next_tile(e, index)
    ahead, last = e.next_kv_tile
    # The loop bound is CTA-uniform. Branch on it before the K issuer's
    # per-thread predicate so ptxas can keep the TMA address calculation uniform.
    issue_guarded(e, ahead, last, copy_k=edge == k_edge, copy_v=edge == v_edge)
