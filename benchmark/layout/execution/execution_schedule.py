"""Restricted scheduling for reusable, synchronously initialized Shared roots."""

from collections import defaultdict


def shared_schedule(packet):
    """Sink each complete copy to its first read and prove root lifetimes.

    This policy is for the one-CTA straight-line packet. Memory operations
    other than known synchronous copies, MMA reads, and the terminal Global
    store may not observe a rescheduled root.
    """
    operations = packet['operations']
    values = {v['id']: v for v in packet['values']}
    shared = {v['storage']['root'] for v in values.values()
              if 'storage' in v and 'memory_space Shared' in v['storage']['layout']}
    if not shared:
        return operations, {}
    copies, uses = {}, defaultdict(list)
    for index, op in enumerate(operations):
        name = op['name']
        operands = op['operands']
        roots = [values[v]['storage']['root'] for v in operands if 'storage' in values[v]]
        touched = shared.intersection(roots)
        if name == 'frisk.copy':
            contract = op['attrs']['frisk.copy_contract']
            if (len(operands) != 2 or len(roots) != 2 or roots[0] in shared or roots[1] not in shared
                    or 'memory_space Global' not in values[roots[0]]['storage']['layout']
                    or contract['path'] != 'vector' or contract['completion'] != 'synchronous'
                    or contract['vector_bytes'] != 16 or roots[1] in copies):
                raise ValueError('reuse requires a unique synchronous full Copy16 per Shared root')
            dst = values[operands[1]]['storage']['addresses']
            root = values[roots[1]]['storage']['addresses']
            if sorted((x['begin_bit'], x['end_bit']) for x in dst) != sorted(
                    (x['begin_bit'], x['end_bit']) for x in root):
                raise ValueError('reuse requires complete Shared initialization')
            copies[roots[1]] = index
        elif name == 'frisk.mma':
            for root in touched:
                uses[root].append(index)
        elif touched and name not in ('frisk.storage_contract', 'frisk.layout_view'):
            raise ValueError('unknown Shared effect in reusable schedule')
    stores = [i for i, op in enumerate(operations) if op['name'] == 'frisk.tile_store']
    if (set(copies) != shared or set(uses) != shared or len(stores) != 1
            or stores[0] <= max(max(x) for x in uses.values())):
        raise ValueError('reuse requires complete Shared lifetimes and one terminal store')
    if any(copies[root] >= uses[root][0] for root in shared):
        raise ValueError('Shared read must follow initialization')
    output = values[operations[stores[0]]['operands'][1]]['storage']['root']
    if output in shared or any(values[operations[i]['operands'][0]]['storage']['root'] == output for i in copies.values()):
        raise ValueError('reusable schedule requires disjoint Global input and output roots')

    insert = defaultdict(list)
    for root, index in copies.items():
        insert[uses[root][0]].append((index, root))
    start, last = {}, {}
    copy_indices = set(copies.values())
    result = []
    for index, op in enumerate(operations):
        for previous, root in sorted(insert[index]):
            start[root] = len(result)
            result.append(operations[previous])
        if index not in copy_indices:
            if op['name'] == 'frisk.mma':
                for operand in op['operands']:
                    storage = values[operand].get('storage')
                    if storage and storage['root'] in shared:
                        last[storage['root']] = len(result)
            result.append(op)
    return result, {root: (start[root], last[root]) for root in copies}


def place_reusable_roots(extents, alignments, intervals, limit, *, reuse_after=None):
    """Assign disjoint ranges to intersecting lifetimes and order reuse barriers."""
    from shared_allocation import reserve_shared

    reuse_after = {} if reuse_after is None else reuse_after
    if not set(reuse_after) <= intervals.keys():
        raise ValueError('Shared reuse destination missing from lifetime plan')
    placed, barriers = [], []
    for root, (start, end) in sorted(intervals.items(), key=lambda item: (item[1], item[0])):
        size, align = extents[root][0], alignments[root]
        active = sorted((p['offset'], p['offset'] + p['bytes']) for p in placed
                        if p['lifetime'][1] >= start)
        cursor = 0
        for low, high in active:
            if (cursor + align - 1) // align * align + size <= low:
                break
            cursor = max(cursor, high)
        if root in reuse_after:
            prior = next((p for p in placed if p['root_id'] == reuse_after[root]), None)
            if (prior is None or prior['lifetime'][1] >= start or prior['bytes'] != size
                    or prior['offset'] % align):
                raise ValueError('exact Shared reuse requires a dead, equal-sized aligned source interval')
            cursor = prior['offset']
            if any(cursor < hi and lo < cursor + size for lo, hi in active):
                raise ValueError('exact Shared reuse overlaps a live interval')
        upper = reserve_shared(cursor, size, align, limit)
        offset = upper - size
        if any(p['lifetime'][1] < start and offset < p['offset'] + p['bytes']
               and p['offset'] < upper for p in placed):
            barriers.append(start)
        placed.append(dict(root_id=root, offset=offset, bytes=size, alignment=align,
                           padding=offset-cursor, lifetime=[start, end]))
    return placed, barriers


def attention_late_schedule(packet):
    """Identify Q, K and V in the checked full-score two-MMA Attention schedule."""
    operations, intervals = shared_schedule(packet)
    values = {v['id']: v for v in packet['values']}
    mmas = [(i, op) for i, op in enumerate(operations) if op['name'] == 'frisk.mma']
    reductions = [i for i, op in enumerate(operations) if op['name'] == 'frisk.reduce_tensor']
    stores = [op for op in operations if op['name'] == 'frisk.tile_store']
    if len(mmas) != 2 or len(reductions) != 2 or len(intervals) != 3 or len(stores) != 1:
        raise ValueError('late V scheduling requires full checked Attention')
    (qk_index, qk), (pv_index, pv) = mmas
    a, b = qk['attrs'], pv['attrs']
    if (a['frisk.mma_contract']['form'] != 'ss' or b['frisk.mma_contract']['form'] != 'rs'
            or a['m'] != b['m'] or a['n'] != b['k'] or a['k'] != b['n']
            or a.get('trans_b') != -1 or b.get('trans_b') != 0
            or not all(qk_index < index < pv_index for index in reductions)
            or stores[0]['operands'][0] not in pv['results']):
        raise ValueError('late V scheduling requires a checked QK-softmax-PV chain')

    producers = {result: i for i, op in enumerate(operations) for result in op['results']}
    work = [pv['operands'][0]]
    dependencies = set()
    while work:
        producer = producers.get(work.pop())
        if producer is not None and producer not in dependencies:
            dependencies.add(producer)
            work.extend(operations[producer]['operands'])
    if not {qk_index, *reductions} <= dependencies:
        raise ValueError('PV input must depend on QK and both softmax reductions')

    roles = {}
    global_inputs = set()
    for role, operand, reader in zip(('q', 'k', 'v'),
                                    (*qk['operands'][:2], pv['operands'][1]),
                                    (qk_index, qk_index, pv_index)):
        root = values[operand].get('storage', {}).get('root')
        if root not in intervals:
            raise ValueError('Attention operand lacks a checked Shared root')
        start, end = intervals[root]
        source = operations[start]['operands'][0]
        global_inputs.add(values[source]['storage']['root'])
        if end != reader:
            raise ValueError('late V scheduling found an unexpected Shared consumer')
        roles[role] = dict(root=root, copy_operation=start, reader_operation=reader)
    q, k, v = (roles[role]['copy_operation'] for role in ('q', 'k', 'v'))
    if (len({row['root'] for row in roles.values()}) != 3 or len(global_inputs) != 3
            or (q, k, v) != (qk_index-2, qk_index-1, pv_index-1)):
        raise ValueError('late V requires adjacent Q/K copies and V immediately before PV')
    return operations, intervals, roles


def attention_publication(roles, allocation, batches, *, reuse_k):
    """Verify actual Q/K/V allocation and the intended publication boundaries."""
    allocated = {row['root_id']: row for row in allocation['allocations']}
    q, k, v = (roles[role]['copy_operation'] for role in ('q', 'k', 'v'))
    if [batch['operations'] for batch in batches] != [[q, k]]:
        raise ValueError('late V requires precisely one Q/K publication batch')
    barriers = allocation.get('reuse_barriers', [])
    if barriers != ([v] if reuse_k else []):
        raise ValueError('unexpected CTA barrier placement for late V')
    proof = {}
    for role, events in roles.items():
        entry = allocated[events['root']]
        proof[role] = dict(events, offset=entry['offset'], bytes=entry['bytes'])
    q, k, v = (proof[role] for role in ('q', 'k', 'v'))

    def overlaps(a, b):
        return a['offset'] < b['offset'] + b['bytes'] and b['offset'] < a['offset'] + a['bytes']

    if overlaps(q,k) or overlaps(q,v) or (reuse_k and (
            (k['offset'],k['bytes']) != (v['offset'],v['bytes'])
            or k['reader_operation'] >= v['copy_operation'])) or (not reuse_k and overlaps(k,v)):
        raise ValueError('unexpected physical Q/K/V interval overlap')
    return dict(version='attention-late-copy-v1', strategy='reuse-k' if reuse_k else 'late', roots=proof,
                reuse_barrier_operation=v['copy_operation'] if reuse_k else None,
                publish_operations=[k['copy_operation'],v['copy_operation']])


def verify_attention_publication(proof, code_by_operation):
    """Check that lowering drains WGMMA and orders both generic/async publications."""
    wait = 'wgmma.wait_group.sync.aligned 0;'
    commit = 'wgmma.commit_group.sync.aligned;'
    barrier = 'bar.sync 0;'
    fence = 'fence.proxy.async.shared::cta;'
    q, k, v = (proof['roots'][role] for role in ('q', 'k', 'v'))
    qcode, kcode, vcode = (code_by_operation[row['copy_operation']] for row in (q, k, v))
    v_barriers = 2 if proof['reuse_barrier_operation'] is not None else 1
    if (barrier in qcode or fence in qcode or kcode[-2:] != [barrier, fence]
            or kcode.count(barrier) != 1 or kcode.count(fence) != 1
            or vcode[-2:] != [barrier, fence] or vcode.count(barrier) != v_barriers or vcode.count(fence) != 1
            or (v_barriers == 2 and vcode[0] != barrier)):
        raise ValueError('emitted Attention copy publication differs from the checked schedule')
    if any(not any(line.startswith('st.shared.') for line in code) for code in (qcode, kcode, vcode)):
        raise ValueError('checked Attention copy is missing Shared writes')
    drains = {}
    for role, row in (('qk', k), ('pv', v)):
        mma = code_by_operation[row['reader_operation']]
        issued, pending, drained = False, False, 0
        for line in mma:
            if line.startswith('wgmma.mma_async.'):
                issued = True
            elif line == commit:
                if not issued:
                    raise ValueError('WGMMA commit without preceding MMA work')
                issued, pending = False, True
            elif line == wait:
                if issued or not pending:
                    raise ValueError('WGMMA wait without completed issue and commit')
                pending = False
                drained += 1
        if issued or pending or not drained or mma[-1] != wait:
            raise ValueError('late V requires fully drained WGMMA groups')
        drains[role] = drained
    return dict(proof, wgmma_wait_group_0=drains)


def synchronous_copy_batches(operations, values, allocation):
    """Prove a shared publication point for consecutive uniform Copy16 operations.

    The emitter still checks each copy's exact full-thread first-owner and
    vector access proof. Here, the checked allocation and endpoint bounds
    establish that no operation observes or overwrites a pending destination.
    """
    global_roots = set(allocation['global_roots'])
    shared_roots = {item['root_id']: item for item in allocation['allocations']}
    reuse_barriers = set(allocation.get('reuse_barriers', []))
    batches, pending = [], []

    def finish():
        if len(pending) > 1:
            batches.append({
                'operations': [ordinal for ordinal, _, _ in pending],
                'shared_bit_ranges': [[begin, end] for _, begin, end in pending],
            })
        pending.clear()

    for ordinal, op in enumerate(operations):
        if ordinal in reuse_barriers:
            finish()
        if op['name'] != 'frisk.copy':
            finish()
            continue
        attrs = op['attrs']
        contract = attrs['frisk.copy_contract']
        if (contract.get('path') != 'vector' or contract.get('completion') != 'synchronous'
                or contract.get('vector_bytes') != 16 or contract.get('guarded') != 0
                or contract.get('issuer') != 'execution-map' or attrs.get('frisk.writer_policy') != 'first_owner'):
            finish()
            continue

        source, target = (values[value]['storage'] for value in op['operands'])
        root = shared_roots.get(target['root'])
        if source['root'] not in global_roots or root is None:
            finish()
            continue
        addresses = target['addresses']
        lower = min(a['begin_bit'] for a in addresses)
        upper = max(a['end_bit'] for a in addresses)
        if not 0 <= lower < upper <= 8 * root['bytes']:
            raise ValueError('copy batching destination exceeds its Shared allocation')
        begin = 8 * root['offset'] + lower
        end = 8 * root['offset'] + upper
        if any(begin < hi and lo < end for _, lo, hi in pending):
            finish()
        pending.append((ordinal, begin, end))
    finish()
    return batches
