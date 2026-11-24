"""Native multi-CTA launch of Frisk's two-warpgroup online attention."""
from wide_execution import WideEmitter


class NativeOnlineEmitter(WideEmitter):
    def run(self):
        ops = self.data['operations']
        module, = [op for op in ops if op['name'] == 'builtin.module']
        cfg = module['attrs']['frisk.online_launch']
        if set(cfg)-{'kv_rows'} != {'rows', 'heads', 'kv_heads', 'causal'}:
            raise ValueError('invalid online launch fields')
        parallel, = [op for op in ops if op['name']=='frisk.parallel']
        self.threads = parallel['attrs']['threads']
        if self.threads not in (128,256):
            raise ValueError('invalid native attention domain')
        self.rows, self.heads, kv_heads = (cfg[k] for k in ('rows', 'heads', 'kv_heads'))
        self.kv_rows = cfg.get('kv_rows',self.rows)
        if (any(type(n) is not int or n <= 0 for n in (self.rows,self.kv_rows,self.heads,kv_heads))
                or self.heads % kv_heads or max(self.rows*self.heads,self.kv_rows*kv_heads) >= 2**30
                or type(cfg['causal']) not in (int,bool) or cfg['causal'] not in (0,1,-1)):
            raise ValueError('invalid packed online attention extent or head grouping')
        self.causal = bool(cfg['causal'])
        self.group = self.heads // kv_heads
        ss,rs = [op for op in ops if op['name']=='frisk.mma']
        self.kv_tile = ss['attrs']['n']
        if self.kv_tile not in (64,128):
            raise ValueError('unsupported online KV tile')
        self.ktiles = (self.kv_rows+self.kv_tile-1)//self.kv_tile
        kernel, = [op for op in ops if op['name']=='frisk.kernel']
        self.q, self.k, self.vroot, self.out = kernel['arguments']
        loop, = [op for op in ops if op['name']=='scf.for']
        if any(op['name'].startswith('scf.') and op['name'] not in ('scf.for','scf.yield') for op in ops):
            raise ValueError('online specialization requires a single uniform KV loop')
        constants = {op['results'][0]:op['attrs']['value'] for op in ops if op['name']=='arith.constant'}
        if [constants[x] for x in loop['operands'][:3]] != [0,self.ktiles,1]:
            raise ValueError('KV loop bounds disagree with the packed launch')
        self.iv = loop['body_arguments'][0]
        copies = [op for op in ops if op['name']=='frisk.copy']
        roots = [self.storage(op['operands'][0])['root'] for op in copies]
        if roots != [self.q,self.k,self.vroot]:
            raise ValueError('expected one Q copy followed by K and V copies in the KV loop')
        lo,hi=ops.index(loop),loop['body_terminator']
        if not ops.index(copies[0]) < lo < ops.index(copies[1]) < ops.index(copies[2]) < hi:
            raise ValueError('unexpected online copy scheduling')
        owners = copies[0]['execution']
        self.packed_owners = (owners['topology'][0]>=8 and
                              owners['logical'][:8]==list(range(8)))
        self.kcopy = copies[1]
        self.vcopy = copies[2]
        self.adjacent_kv = ops.index(copies[2]) == ops.index(copies[1])+1
        self.batched_kv = False
        self.qtile, self.dim = ss['attrs']['m'], ss['attrs']['k']
        self.value_dim = rs['attrs']['n']
        if (self.value_dim not in (32,64,128) or self.qtile not in (64,128,256,512) or self.dim not in (64,128,256) or
                [rs['attrs'][k] for k in ('m','n','k')] != [self.qtile,self.value_dim,self.kv_tile]):
            raise ValueError('incompatible online QK/PV tile geometry')
        start = getattr(self,'online_query_start',0)
        if not (type(start) is int and 0<=start<self.rows and start%self.qtile==0):
            raise ValueError('query slice must start on a query-tile boundary')
        self.qtiles = (self.rows-start+self.qtile-1)//self.qtile
        lengths=(self.rows,self.kv_rows,self.kv_rows,self.rows)
        pitches=getattr(self,'online_head_pitches',None)
        self.pitches=lengths if pitches is None else pitches
        if len(self.pitches)!=4 or any(type(p) is not int or p<n or
                p*self.heads*max(self.dim,self.value_dim)>=2**31 for p,n in zip(self.pitches,lengths)):
            raise ValueError('invalid online head pitches')
        if not (lo < ops.index(ss) < ops.index(rs) < hi and
                [o['attrs']['frisk.mma_contract']['form'] for o in (ss,rs)]==['ss','rs']):
            raise ValueError('expected a certified QK/PV pair')
        self.logits, = [op for op in ops if op['name']=='arith.mulf' and ss['results'][0] in op['operands']]
        store, = [op for op in ops if op['name']=='frisk.tile_store']
        if self.storage(store['operands'][1])['root'] != self.out or ops.index(store) <= hi:
            raise ValueError('expected one post-loop output store')
        self.output_bytes = 2 if 'xf16' in self.v[self.out]['type'].split(',')[0] else 4
        views=[op['operands'][0] for op in copies]+[store['operands'][1]]
        for view,rows,cols,width in zip(views,(self.qtile,self.kv_tile,self.kv_tile,self.qtile),
                                       (self.dim,self.dim,self.value_dim,self.value_dim),(2,2,2,self.output_bytes),strict=True):
            offsets=self.offsets(view)
            if (len(offsets)!=rows*cols or
                    any(offsets[i]!=i*width for i in range(rows*cols))):
                raise ValueError('packed online attention requires row-major Q/K/V/O views')
        if any(op['name']=='frisk.tile_load' for op in ops):
            raise ValueError('unexpected global input in the online graph')
        from native_online_memory import identity_suffix
        self.pruned_loop = loop if self.causal and identity_suffix(ops,loop,ss,rs,self.logits) else None
        self.qk_op, self.pv_op = ss, rs
        if getattr(self, 'defer_v_wait', False):
            target = self.storage(self.vcopy['operands'][1])['root']
            consumers = [op for op in ops[ops.index(self.vcopy)+1:hi] if any(
                'memref<' in self.v[value]['type'] and self.storage(value)['root'] == target
                for value in op['operands'])]
            if not self.adjacent_kv or consumers != [rs] or self.storage(rs['operands'][1])['root'] != target:
                raise ValueError('the V tile must have only its certified PV consumer before reuse')
        self.k_tma = None
        self.v_tma = None
        self.ring_stride = 0
        self.copy_version = 0
        self.carry_alias = {}
        self.positive_sum_ops = set()
        self.invariants = []
        self.descriptors = {}
        self.kv_loop = loop
        schedule = getattr(self, 'online_loop_schedule', 'auto')
        grouped_qk = (schedule == 'grouped-qk-ring' and self.dim == self.value_dim == 128
                      and self.qtile == 64 and self.group > 1)
        # Both WGMMA groups consume the same completed K/V versions. A 2x1
        # grid gives each group a distinct 64-row slice of Q and the output.
        paired_rows = (self.output_bytes == 4 and self.qtile == 128 and self.dim == self.value_dim == 64 and
                       self.group > 1 and not self.causal and self.threads == 256 and
                       getattr(self, 'kv_prefetch_edges', None) is None and
                       getattr(self, 'k_transport', None) == 'tma' and self.k_map_abi == 'value' and
                       ss['attrs']['frisk.mma_contract']['grid'] == [2,1] and
                       rs['attrs']['frisk.mma_contract']['grid'] == [2,1])
        group_k = 16 * self.mma_group_size
        self.ring_candidate = (
            (schedule == 'auto' or grouped_qk)
            and getattr(self, 'defer_v_wait', False) and self.aligned_online_inputs
            and (self.threads == 128 or paired_rows) and self.ktiles >= 4
            and (self.output_bytes == 4 or (self.dim == self.value_dim == 64 and self.k_transport == "tma"))
            and (ss['attrs']['k'] == group_k or (grouped_qk and ss['attrs']['k'] % group_k == 0))
            and rs['attrs']['k'] % group_k == 0)
        if self.ring_candidate:
            # Changing the physical version is legal only when the two original
            # shared tiles are private to their respective matrix instructions.
            for copy, consumer in ((self.kcopy, ss), (self.vcopy, rs)):
                root = self.storage(copy['operands'][1])['root']
                users = [op for op in ops[ops.index(copy)+1:] if any(
                    'memref<' in self.v[v]['type'] and self.storage(v)['root'] == root
                    for v in op['operands'])]
                self.ring_candidate &= users == [consumer]
        if self.ring_candidate:
            self.bind_loop_accumulator(ops, loop, rs)
            exponentials = {v for op in ops[lo:hi] if op['name'] == 'math.exp2' for v in op['results']}
            self.positive_sum_ops = {id(op) for op in ops[lo:hi] if op['name'] == 'frisk.reduce_tensor'
                                    and op['attrs']['kind'].strip('"') == 'sum' and op['operands'][0] in exponentials}
        self.cross_iteration_pv = getattr(self, 'cross_iteration_pv', False)
        self.prefetch_after_qk = getattr(self, 'prefetch_after_qk', False)
        self.kv_prefetch_edges = getattr(self, 'kv_prefetch_edges', None)
        if (self.prefetch_after_qk or self.kv_prefetch_edges is not None) and not (
                self.ring_candidate and (self.qtile == self.dim == self.value_dim == 64 or paired_rows)
                and self.k_transport == 'tma' and self.k_map_abi == 'value'
                and ops[lo+1:ops.index(ss)] == [self.kcopy, self.vcopy]
                and ss['attrs']['frisk.mma_contract']['repeats'] == [1,1,4]):
            raise ValueError('QK-relative prefetch requires the certified D64 by-value K-TMA ring')
        if self.cross_iteration_pv:
            if not (self.ring_candidate and not self.causal and self.kv_tile == 128
                    and self.qtile == self.dim == self.value_dim == 64
                    and self.k_transport == 'tma' and self.k_map_abi == 'value'
                    and ops[lo+1:ops.index(ss)] == [self.kcopy,self.vcopy]
                    and self.carry_alias and ss['operands'][2] in constants and ops[hi-1] is rs
                    and ss['results'][0] not in ops[hi]['operands']
                    and ss['attrs']['frisk.mma_contract']['repeats'] == [1,1,4]
                    and rs['attrs']['frisk.mma_contract']['repeats'] == [1,1,8]):
                raise ValueError('cross-iteration PV requires the ordinary D64 KV128 single-warpgroup ring')
            self.packed_pv_carry = {}
            self.pending_pv_by_slot = {}
        self.normalizer_ops = ()
        self.pending_normalizer = []
        if (self.ring_candidate and (self.qtile==self.dim==self.value_dim==64 or paired_rows)
                and getattr(self,'k_transport',None)=='tma' and self.k_map_abi=='value'
                and self.overlap_pv_normalizer):
            self.normalizer_ops = self.independent_normalizer(ops, lo, hi, rs)
        self.packed_copies = []
        text, info = super().run()
        if getattr(self, 'defer_v_wait', False) and not self.batched_kv:
            raise ValueError('the deferred V wait needs two certified packed copies')
        info['defer_v_wait'] = bool(getattr(self, 'defer_v_wait', False) and not self.ring_stride)
        info['online_kv_schedule'] = 'two-slot-lookahead' if self.ring_stride else 'original'
        info['loop_optimizations'] = dict(ring=bool(self.ring_stride), carry=bool(self.carry_alias),
                                          interior_mask=bool(self.ring_stride and (self.causal or self.kv_rows % self.kv_tile)),
                                          positive_sum=bool(self.positive_sum_ops))
        info.update(batched_kv=self.batched_kv, causal_kv_pruned=self.pruned_loop is not None, packed_copy_roots=self.packed_copies)
        info.update(launch_grid=[self.qtiles*self.heads,1,1], query_start=start, head_pitches=list(self.pitches), kv_tiles=self.ktiles,
                    native_online_launch=cfg, q_tile=self.qtile, head_dim=self.dim)
        if self.k_tma:
            info['kv_tile'] = self.kv_tile
            info['overlap_pv_normalizer'] = bool(self.normalizer_ops and self.ring_stride)
            info["k_tensor_map"] = self.k_tma
            info["operand_transport"] = {"Q":"cp.async", "K":"tma", "V":"cp.async"}
        if self.v_tma:
            info['v_tensor_map'] = self.v_tma
            info['operand_transport']['V'] = 'tma'
            info['kv_completion'] = dict(issuer_thread=0, slots=2, barriers=self.k_tma['barriers'],
                                         expected_bytes=self.k_tma['transaction_bytes'] + self.v_tma['transaction_bytes'],
                                         phase=self.k_tma['phase'], wait='joint-before-qk')
        if self.kv_prefetch_edges is not None:
            k_edge, v_edge = self.kv_prefetch_edges
            info['online_kv_schedule'] = 'independent-kv-prefetch'
            info['next_tile_prefetch'] = dict(K=k_edge, V=v_edge, waits='before-qk', pv='same-iteration')
        if self.prefetch_after_qk:
            info['online_kv_schedule'] = 'post-qk-prefetch'
        if self.cross_iteration_pv:
            info['online_kv_schedule'] = 'previous-pv-next-qk'
            info['pv_completion'] = dict(in_flight=['PV(i), last group','QK(i+1)'],
                                         release='wait_group 1, cp.async.wait_group 0 and CTA barrier before slot reuse',
                                         tail='last PV group, wait_group 0', carried_packed_registers=16)
        if self.value_dim != self.dim:
            info["value_dim"] = self.value_dim
        return text,info

    def independent_normalizer(self, ops, lo, hi, pv):
        # These two register-only operations update the loop-carried row sum.
        # Neither the PV operands nor a second observer may consume their result.
        body = ops[lo+1:hi]
        sums = [op for op in body if id(op) in self.positive_sum_ops]
        if len(sums) != 1:
            return ()
        reduction, = sums
        total, = reduction['results']
        users = [op for op in ops if total in op['operands']]
        if len(users)!=1 or users[0]['name']!='arith.addf':
            return ()
        update, = users
        result, = update['results']
        if (not (ops.index(reduction)<ops.index(update)<ops.index(pv)<hi)
                or [op for op in ops if result in op['operands']] != [ops[hi]]):
            return ()
        return reduction, update

    def packed_mma_register(self, op, k, j, depth):
        if self.cross_iteration_pv and op is self.pv_op and k >= depth-self.mma_group_size:
            key = (k,j)
            if key not in self.packed_pv_carry:
                self.packed_pv_carry[key] = self.tmp()
            return self.packed_pv_carry[key]
        return super().packed_mma_register(op,k,j,depth)

    def emit_mma_group(self, op, instructions, *, last):
        if not self.cross_iteration_pv:
            return super().emit_mma_group(op, instructions, last=last)
        if op is self.pv_op and last:
            # Only the RS fragment and the partial O are carried. No asynchronous
            # group crosses the loop branch. The V slot is kept until the next QK.
            self.pending_pv_by_slot[self.loop_version] = instructions
            return
        if op is not self.qk_op:
            return super().emit_mma_group(op, instructions, last=last)

        self.emit('setp.eq.u32 mma_acc, 0, 0;')
        self.emit('wgmma.fence.sync.aligned;')
        if not self.loop_prime:
            self.issue_mma_group(self.pending_pv_by_slot[1-self.loop_version])
        self.issue_mma_group(instructions)
        self.emit('wgmma.wait_group.sync.aligned 1;')
        self.publish_v_and_prefetch()
        self.wait_for_mma(op)

    def publish_v_and_prefetch(self):
        from native_online_ring import prefetch_next
        self.emit('cp.async.wait_group 0;')
        self.emit('bar.sync 0;')
        self.emit('fence.proxy.async.shared::cta;')
        prefetch_next(self, self.indices[self.iv])

    def drain_pv(self):
        # Ordinary attention has the same static trip count at every CTA.
        self.emit('setp.eq.u32 mma_acc, 0, 0;')
        self.emit('wgmma.fence.sync.aligned;')
        self.issue_mma_group(self.pending_pv_by_slot[(self.ktiles-1)%2])
        self.emit('wgmma.wait_group.sync.aligned 0;')

    def wait_for_mma(self, op):
        if getattr(self, 'kv_prefetch_edges', None) is not None and op is self.qk_op:
            from native_online_ring import issue_at_qk_edge
            issue_at_qk_edge(self, self.indices[self.iv], 'after')
        if getattr(self, 'prefetch_after_qk', False) and op is self.qk_op:
            # The previous PV was drained. Only the current QK can be in flight.
            self.publish_v_and_prefetch()
        if op is self.pv_op and self.pending_normalizer:
            # The PV accumulator remains untouched until wait_group. Keep the
            # original FP32 reduction tree and row-add operand order.
            for pending in self.pending_normalizer:
                super().operation(pending)
            self.pending_normalizer.clear()
        super().wait_for_mma(op)

    def bind_loop_accumulator(self, ops, loop, pv):
        # The old O, rescaled O and PV result have consecutive lifetimes.
        yield_op = ops[loop['body_terminator']]
        result = pv['results'][0]
        if yield_op['operands'].count(result) != 1:
            return
        carry = loop['body_arguments'][1+yield_op['operands'].index(result)]
        scaled = pv['operands'][2]
        users = [op for op in ops if carry in op['operands']]
        if (len(users) == 1 and users[0]['name'] == 'arith.mulf' and users[0]['results'] == [scaled]
                and [op for op in ops if scaled in op['operands']] == [pv]
                and [op for op in ops if result in op['operands']] == [yield_op]
                and self.tensor_slots(carry) == self.tensor_slots(scaled) == self.tensor_slots(result)):
            self.carry_alias = {scaled: carry, result: carry}

    def newregs(self, value):
        if value in self.carry_alias:
            self.reg[value] = self.regs(self.carry_alias[value])
            return self.reg[value]
        return super().newregs(value)

    def finalize_allocation(self, allocation):
        if self.ring_candidate:
            from native_online_ring import prepare_ring
            allocation = prepare_ring(self, allocation)
            if not self.ring_stride:
                if self.cross_iteration_pv or self.prefetch_after_qk or self.kv_prefetch_edges is not None:
                    raise ValueError('the requested schedule requires two allocated K and V slots')
                self.carry_alias.clear()
                self.positive_sum_ops.clear()
        if getattr(self, 'k_transport', 'cp.async') == 'tma':
            from online_k_tma import configure
            allocation = configure(self, allocation)
            if self.v_transport == 'tma':
                from online_k_tma import configure_v
                self.v_tma = configure_v(self, allocation)
        return allocation

    def loop_body_versions(self, op):
        return 2 if self.ring_stride and op is self.kv_loop else 1

    def shared(self, value, offset):
        addr = super().shared(value, offset)
        if self.copy_version and self.storage(value)['root'] in self.ring_roots:
            self.emit(f'add.u32 {addr}, {addr}, {self.ring_stride};')
        return addr

    def group_descriptor(self, value, desc, starts, k):
        if not self.ring_stride:
            return super().group_descriptor(value, desc, starts, k)
        displacement = self.loop_version*self.ring_stride if self.storage(value)['root'] in self.ring_roots else 0
        key = (value, id(desc), tuple(starts), k, displacement)
        if key not in self.descriptors:
            pos = len(self.code)
            self.descriptors[key] = super().group_descriptor(value, desc, starts, k, displacement)
            self.invariants.extend(self.code[pos:])
            del self.code[pos:]
        return self.descriptors[key]

    def sequence(self, ops, begin, end):
        if begin==0:
            cta,head,qtile,kvhead = [self.tmp() for _ in range(4)]
            self.qrow,self.krow = self.tmp(),self.tmp()
            self.emit(f'mov.u32 {cta}, %ctaid.x;')
            self.emit(f'div.u32 {head}, {cta}, {self.qtiles};')
            self.emit(f'rem.u32 {qtile}, {cta}, {self.qtiles};')
            self.emit(f'mul.lo.u32 {self.qrow}, {qtile}, {self.qtile};')
            if self.online_query_start:
                self.emit(f'add.u32 {self.qrow}, {self.qrow}, {self.online_query_start};')
            self.kv_head = kvhead
            self.emit(f'div.u32 {kvhead}, {head}, {self.group};')
            qbase,self.kbase = self.tmp(),self.tmp()
            qp,kp,vp,op=self.pitches
            self.emit(f'mad.lo.u32 {qbase}, {head}, {qp}, {self.qrow};')
            self.emit(f'mul.lo.u32 {self.kbase}, {kvhead}, {kp};')
            self.ptr = {r:self.tmp('b64') for r in (self.q,self.k,self.vroot,self.out)}
            self.set_pointer(self.q,qbase,2*self.dim)
            self.vbase=self.kbase
            if vp!=kp:
                self.vbase=self.tmp()
                self.emit(f'mul.lo.u32 {self.vbase}, {kvhead}, {vp};')
            if op!=qp:
                self.emit(f'mad.lo.u32 {qbase}, {head}, {op}, {self.qrow};')
            self.set_pointer(self.out,qbase,self.output_bytes*self.value_dim)
        super().sequence(ops,begin,end)
        if begin == 0 and self.invariants:
            self.code[self.preheader:self.preheader] = self.invariants

    def map(self, table):
        # A GF(2) column contributes one or more output bits. Group all input
        # bits with the same displacement into one shift and mask.
        from backend import linear
        if self.access_policy == 'legacy' or not self.packed_owners:
            return super().map(table)
        base, columns = linear(table)
        masks = {}
        for source, column in enumerate(columns):
            while column:
                bit = column & -column
                shift = bit.bit_length()-1-source
                masks[shift] = masks.get(shift,0) | (1<<source)
                column ^= bit
        out = self.tmp()
        self.emit(f'mov.u32 {out}, {base};')
        for shift,mask in masks.items():
            part = self.tmp()
            self.emit(f'and.b32 {part}, tid, {mask};')
            if shift:
                opcode = 'shl.b32' if shift > 0 else 'shr.u32'
                self.emit(f'{opcode} {part}, {part}, {abs(shift)};')
            self.emit(f'xor.b32 {out}, {out}, {part};')
        return out

    def global_io(self, op, store):
        if not store or (self.output_bytes==4 and (self.access_policy=='legacy' or not self.packed_owners)):
            return super().global_io(op,store)
        value,buffer = op['operands']
        dist = self.v[value]['distributed']
        regs = self.regs(value)
        slots = self.tensor_slots(value)
        root = self.storage(buffer)['root']
        offsets = self.offsets(buffer)
        width = self.output_bytes
        dtype = f'xf{width*8}'
        if any(dtype not in self.v[v]['type'].split(',')[0] for v in (value,buffer)):
            raise ValueError('packed attention requires matching store carrier and buffer types')
        # The producer may own adjacent pairs even when four-word groups
        # cross lanes. Validate alignment and election for every physical lane.
        for lanes in (4, 2, 1):
            if self.storage(buffer)["alignment"] < width*lanes or len(regs)%lanes:
                continue
            groups = []
            for i in range(0,len(regs),lanes):
                start = [offsets[x] for x in slots[i]]
                if any(base % (lanes*width) or offsets[x] != base+width*j or
                       dist['first'][x] != t*len(regs)+i+j
                       for j in range(lanes)
                       for t,(base,x) in enumerate(zip(start,slots[i+j],strict=True))):
                    break
                groups.append(start)
            else:
                for i,start in enumerate(groups):
                    off = self.map(start)
                    addr = self.address(root,off)
                    guard = self.access_guard(root,off)
                    vec = ', '.join(regs[lanes*i:lanes*(i+1)])
                    if lanes == 1:
                        self.emit(f'{guard}st.global.b{width*8} [{addr}], {vec};')
                    else:
                        self.emit(f'{guard}st.global.v{lanes}.b{width*8} [{addr}], '+'{'+vec+'};')
                return
        return super().global_io(op,store)

    def loop_upper_bound(self, op, static_bound):
        bound = self.active_kv_bound(op, static_bound)
        if self.ring_stride and op is self.kv_loop:
            from native_online_ring import issue_tile
            self.kv_bound = bound
            self.preheader = len(self.code)
            if self.k_tma:
                from online_k_tma import initialize
                initialize(self)
            issue_tile(self, self.indices[self.iv], 0)
        return bound

    def active_kv_bound(self, op, static_bound):
        if op is not self.pruned_loop:
            return static_bound
        last = self.tmp()
        self.emit(f'add.u32 {last}, {self.qrow}, {self.qtile+(self.kv_tile-1 if self.qtile%self.kv_tile else 0)};')
        self.emit(f'shr.u32 {last}, {last}, {self.kv_tile.bit_length()-1};')
        self.emit(f'min.u32 {last}, {last}, {static_bound};')
        return last

    def set_pointer(self,root,row,stride):
        addr=self.ptr[root]
        self.emit(f'mul.wide.u32 {addr}, {row}, {stride};')
        self.emit(f'add.u64 {addr}, {addr}, \u0024{self.roots[root]+1};')

    def address(self,root,offset):
        addr=self.tmp('b64')
        self.emit(f'cvt.u64.u32 {addr}, {offset};')
        self.emit(f'add.u64 {addr}, {addr}, {self.ptr[root]};')
        return addr

    def wide_copy(self,op):
        if self.ring_stride and op is self.vcopy:
            return
        if self.ring_stride and op is self.kcopy:
            from native_online_ring import enter_iteration
            self.batched_kv = True
            if self.loop_version == 0 and (not self.cross_iteration_pv or self.loop_prime):
                self.packed_copies.extend((self.k,self.vroot))
            index = self.indices[self.iv]
            self.emit(f'mul.lo.u32 {self.krow}, {index}, {self.kv_tile};')
            enter_iteration(self, index)
            return
        if op is self.kcopy:
            index=self.indices[self.iv];row=self.tmp()
            self.emit(f'mul.lo.u32 {self.krow}, {index}, {self.kv_tile};')
            self.emit(f'add.u32 {row}, {self.kbase}, {self.krow};')
            self.set_pointer(self.k,row,2*self.dim)
            if self.vbase!=self.kbase:
                self.emit(f'add.u32 {row}, {self.vbase}, {self.krow};')
            self.set_pointer(self.vroot,row,2*self.value_dim)
        from native_online_memory import emit_packed_copies
        if op is self.vcopy and self.batched_kv:
            return
        if op is self.kcopy and self.adjacent_kv and self.access_policy != "legacy":
            self.batched_kv = emit_packed_copies(self,[op,self.vcopy])
            if self.batched_kv:
                return
        if self.access_policy != 'legacy' and emit_packed_copies(self,[op]):
            return
        super().wide_copy(op)

    def access_guard(self,root,offset):
        query = root in (self.q,self.out)
        limit = self.rows if query else self.kv_rows
        if limit % (self.qtile if query else self.kv_tile)==0:
            return ''
        row=self.tmp();pred=self.tmp('pred')
        cols = self.value_dim if root in (self.out,self.vroot) else self.dim
        self.emit(f'shr.u32 {row}, {offset}, {(cols * (self.output_bytes if root==self.out else 2)).bit_length()-1};')
        self.emit(f'add.u32 {row}, {row}, {self.qrow if query else self.krow};')
        self.emit(f'setp.lt.u32 {pred}, {row}, {limit};')
        return '@'+pred+' '

    def operation(self,op):
        if self.ring_stride and any(op is pending for pending in self.normalizer_ops):
            self.pending_normalizer.append(op)
            return
        if op is self.pv_op and not self.ring_stride and getattr(self, 'defer_v_wait', False):
            self.emit('cp.async.wait_group 0;')
            self.emit('bar.sync 0;')
            self.emit('fence.proxy.async.shared::cta;')
        super().operation(op)
        if op is not self.logits or not (self.causal or self.kv_rows%self.kv_tile):
            return
        if self.ring_stride:
            last, interior = self.tmp(), self.tmp('pred')
            self.emit(f'add.u32 {last}, {self.krow}, {self.kv_tile-1};')
            self.emit(f'setp.lt.u32 {interior}, {last}, {self.kv_rows};')
            if self.causal:
                self.emit(f'setp.le.and.u32 {interior}, {last}, {self.qrow}, {interior};')
            mask_label = 'frisk_mask_prime' if self.loop_prime else f'frisk_mask_done_{self.loop_version}'
            self.emit(f'@{interior} bra.uni {mask_label};')
        v,=op['results']
        for reg,slots in zip(self.regs(v),self.tensor_slots(v),strict=True):
            k=self.map([x%self.kv_tile for x in slots]);pred=self.tmp('pred')
            self.emit(f'add.u32 {k}, {k}, {self.krow};')
            if self.causal:
                q=self.map([x//self.kv_tile for x in slots])
                self.emit(f'add.u32 {q}, {q}, {self.qrow};')
                self.emit(f'setp.le.u32 {pred}, {k}, {q};')
            if self.kv_rows%self.kv_tile:
                combo='.and' if self.causal else ''
                extra=', '+pred if self.causal else ''
                self.emit(f'setp.lt{combo}.u32 {pred}, {k}, {self.kv_rows}{extra};')
            self.emit(f'selp.f32 {reg}, {reg}, 0fff800000, {pred};')
        if self.ring_stride:
            self.emit(mask_label+':')
