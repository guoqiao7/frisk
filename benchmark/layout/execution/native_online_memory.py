"""Verified packed input copies and uniform causal KV bounds for native attention."""


def identity_suffix(ops, loop, qk, pv, logits):
    """Recognize the online recurrence before omitting fully masked suffix tiles."""
    if len(loop['body_arguments']) != 4:
        return False
    _, carry_m, carry_l, carry_o = loop['body_arguments']
    symbols = {carry_m:'m', carry_l:'l', carry_o:'o'}
    def value(key):
        return symbols.get(key,key)
    score = 'masked_logits'
    mx = ('max',score)
    new_m = ('arith.maximumf','m',mx)
    phase = ops[ops.index(loop)+1:loop['body_terminator']]
    exps = {op['name'] for op in phase if op['name'] in ('math.exp','math.exp2')}
    if len(exps)!=1:
        return False
    exp, = exps
    alpha = (exp,('arith.subf','m',new_m))
    prob = (exp,('arith.subf',score,('row',new_m)))
    source_p = ('arith.truncf',prob)
    scaled_o = ('arith.mulf','o',('row',alpha))
    new_l = ('arith.addf',('arith.mulf','l',alpha),('sum',prob))
    start = ops.index(loop)
    for op in ops[start+1:loop['body_terminator']]:
        name, args = op['name'], [value(x) for x in op['operands']]
        if op is qk or name in ('frisk.copy','linalg.yield'):
            continue
        if op is logits:
            expr = score
        elif op is pv:
            if (args[0],args[2]) != (source_p,scaled_o):
                return False
            expr = 'new_o'
        elif name == 'frisk.convert_layout':
            expr = args[0]
        elif name == 'linalg.broadcast' and op['attrs']['dimensions']==[1]:
            expr = ('row',args[0])
        elif name == 'frisk.reduce_tensor' and op['attrs']['dim']==1:
            expr = (op['attrs']['kind'],args[0])
        elif name in ('arith.maximumf','arith.mulf','arith.addf','arith.subf','math.exp','math.exp2','arith.truncf'):
            expr = (name,*args)
        else:
            return False
        result, = op['results']
        symbols[result] = expr
    return [value(x) for x in ops[loop['body_terminator']]['operands']] == [new_m,new_l,'new_o']


def _packed_groups(emitter, op):
    """Group eight existing, first-owned f16 elements into one SM80+ copy.

    The copy is still synchronous at the SSA boundary. Runtime alignment dispatch
    preserves the original byte-aligned global ABI. Partial last rows zero-fill.
    """
    e = emitter
    a,b = op['operands']
    layout, attrs = op['execution'], op['attrs']
    cert = attrs['frisk.copy_contract']
    if not (cert['path']=='vector' and cert['completion']=='synchronous'
            and cert['vector_bytes'] in (1,2) and not cert.get('guarded')
            and attrs['frisk.writer_policy']=='first_owner'
            and all('xf16' in e.v[x]['type'] for x in (a,b))
            and layout['topology'][1:]==[32,4,e.threads//128,1]
            and all(attrs[k]=='affine_map<() -> ()>' for k in ('srcMap','dstMap'))
            and attrs['srcExtents']==attrs['dstExtents']==layout['shape']):
        return False
    nr = layout['topology'][0]
    slots = layout['logical']
    if nr<8 or nr%8 or len(slots)!=nr*e.threads or slots[1]!=slots[0]+1:
        return False
    src,dst = e.offsets(a),e.offsets(b)
    shape = layout['shape']
    shared_root = e.storage(b)['root']
    if e.smem[shared_root]%16 or e.storage(shared_root)['alignment']<16:
        return False
    if len(slots)!=len(src) or len(dst)!=len(slots) or any(layout['first'][x]!=i for i,x in enumerate(slots)):
        return False
    groups=[]
    for i in range(0,nr,8):
        sa,da=[],[]
        for t in range(e.threads):
            indices=slots[t*nr+i:t*nr+i+8];first=indices[0]
            if first//shape[1] != indices[-1]//shape[1]:
                return False
            if any(table[first]%16 or any(table[x]!=table[first]+2*j for j,x in enumerate(indices))
                   for table in (src,dst)):
                return False
            sa.append(src[first]);da.append(dst[first])
        groups.append((sa,da))
    root=e.storage(a)['root']
    return [(root,b,sa,da) for sa,da in groups]


def issue_packed_copy(e, group):
    root,b,sa,da=group
    offset=e.map(sa);src_addr=e.address(root,offset);dst_addr=e.shared(b,e.map(da))
    guard=e.access_guard(root,offset)
    size='16'
    if guard:
        size=e.tmp()
        e.emit(f'selp.u32 {size}, 16, 0, {guard.strip()[1:]};')
    e.emit(f'cp.async.cg.shared.global [{dst_addr}], [{src_addr}], 16, {size};')


def emit_packed_copies(e, ops):
    batches = [_packed_groups(e,op) for op in ops]
    if not all(batches):
        return False
    roots = [batch[0][0] for batch in batches]
    targets = [e.storage(batch[0][1])['root'] for batch in batches]
    if len(set(targets))!=len(targets):
        return False
    groups = [group for batch in batches for group in batch]
    label='frisk_packed_'+str(len(e.packed_copies))
    e.packed_copies.extend(roots)
    # Each packed head and tile stride is a multiple of 16 bytes.
    # Access proofs above also checked all per-thread displacements.
    proven_aligned = getattr(e, "aligned_online_inputs", False) or all(e.storage(op['operands'][0])['alignment'] % 16 == 0
                         and e.storage(op['operands'][0])['alignment'] >= 16 for op in ops)
    if not proven_aligned:
        _branch_on_alignment(e, roots, label)
    e.emit('bar.sync 0;')
    split = getattr(e, 'defer_v_wait', False) and len(ops) == 2
    for index,(root,b,sa,da) in enumerate(groups):
        if split and index == len(batches[0]):
            e.emit('cp.async.commit_group;')
        issue_packed_copy(e,(root,b,sa,da))
    e.emit('cp.async.commit_group;')
    e.emit('cp.async.wait_group 1;' if split else 'cp.async.wait_group 0;')
    e.emit('bar.sync 0;')
    e.emit('fence.proxy.async.shared::cta;')
    if proven_aligned:
        return True
    e.emit(f'bra.uni {label}_done;')
    e.emit(label+'_scalar:')
    e.emit('bar.sync 0;')
    for i,(root,b,sa,da) in enumerate(groups):
        offset=e.map(sa);source=e.address(root,offset);destination=e.shared(b,e.map(da))
        guard=e.access_guard(root,offset)
        byte=e.tmp();count=e.tmp();more=e.tmp('pred')
        e.emit(f'mov.u32 {count}, 0;')
        e.emit(f'{label}_byte_{i}:')
        if guard:
            e.emit(f'mov.u32 {byte}, 0;')
        e.emit(f'{guard}ld.global.u8 {byte}, [{source}];')
        e.emit(f'st.shared.u8 [{destination}], {byte};')
        e.emit(f'add.u64 {source}, {source}, 1;')
        e.emit(f'add.u32 {destination}, {destination}, 1;')
        e.emit(f'add.u32 {count}, {count}, 1;')
        e.emit(f'setp.lt.u32 {more}, {count}, 16;')
        e.emit(f'@{more} bra.uni {label}_byte_{i};')
    e.emit('bar.sync 0;')
    e.emit('fence.proxy.async.shared::cta;')
    e.emit(label+'_done:')
    return True


def _branch_on_alignment(e, roots, label):
    low=e.tmp('b64');aligned=e.tmp('pred')
    e.emit(f'mov.u64 {low}, {e.ptr[roots[0]]};')
    for root in roots[1:]:
        e.emit(f'or.b64 {low}, {low}, {e.ptr[root]};')
    e.emit(f'and.b64 {low}, {low}, 15;')
    e.emit(f'setp.eq.u64 {aligned}, {low}, 0;')
    e.emit(f'@!{aligned} bra.uni {label}_scalar;')
