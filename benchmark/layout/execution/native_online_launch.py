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
        self.ktiles = (self.kv_rows+63)//64
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
        ss,rs = [op for op in ops if op['name']=='frisk.mma']
        self.qtile, self.dim = ss['attrs']['m'], ss['attrs']['k']
        self.value_dim = rs['attrs']['n']
        if (self.value_dim not in (32,64,128) or self.qtile not in (64,128,256,512) or self.dim not in (64,128,256) or
                ss['attrs']['n'] != 64 or
                [rs['attrs'][k] for k in ('m','n','k')] != [self.qtile,self.value_dim,64]):
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
        for view,rows,cols,width in zip(views,(self.qtile,64,64,self.qtile),
                                       (self.dim,self.dim,self.value_dim,self.value_dim),(2,2,2,self.output_bytes),strict=True):
            offsets=self.offsets(view)
            if (len(offsets)!=rows*cols or
                    any(offsets[i]!=i*width for i in range(rows*cols))):
                raise ValueError('packed online attention requires row-major Q/K/V/O views')
        if any(op['name']=='frisk.tile_load' for op in ops):
            raise ValueError('unexpected global input in the online graph')
        from native_online_memory import identity_suffix
        self.pruned_loop = loop if self.causal and identity_suffix(ops,loop,ss,rs,self.logits) else None
        self.pv_op = rs
        if getattr(self, 'defer_v_wait', False):
            target = self.storage(self.vcopy['operands'][1])['root']
            consumers = [op for op in ops[ops.index(self.vcopy)+1:hi] if any(
                'memref<' in self.v[value]['type'] and self.storage(value)['root'] == target
                for value in op['operands'])]
            if not self.adjacent_kv or consumers != [rs] or self.storage(rs['operands'][1])['root'] != target:
                raise ValueError('the V tile must have only its certified PV consumer before reuse')
        self.packed_copies = []
        text, info = super().run()
        if getattr(self, 'defer_v_wait', False) and not self.batched_kv:
            raise ValueError('the deferred V wait needs two certified packed copies')
        info['defer_v_wait'] = getattr(self, 'defer_v_wait', False)
        info.update(batched_kv=self.batched_kv, causal_kv_pruned=self.pruned_loop is not None, packed_copy_roots=self.packed_copies)
        info.update(launch_grid=[self.qtiles*self.heads,1,1], query_start=start, head_pitches=list(self.pitches), kv_tiles=self.ktiles,
                    native_online_launch=cfg, q_tile=self.qtile, head_dim=self.dim)
        if self.value_dim != self.dim:
            info["value_dim"] = self.value_dim
        return text,info

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
        if op is not self.pruned_loop:
            return static_bound
        last = self.tmp()
        self.emit(f'add.u32 {last}, {self.qrow}, {self.qtile};')
        self.emit(f'shr.u32 {last}, {last}, 6;')
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
        if op is self.kcopy:
            index=self.indices[self.iv];row=self.tmp()
            self.emit(f'mul.lo.u32 {self.krow}, {index}, 64;')
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
        if limit % (self.qtile if query else 64)==0:
            return ''
        row=self.tmp();pred=self.tmp('pred')
        cols = self.value_dim if root in (self.out,self.vroot) else self.dim
        self.emit(f'shr.u32 {row}, {offset}, {(cols * (self.output_bytes if root==self.out else 2)).bit_length()-1};')
        self.emit(f'add.u32 {row}, {row}, {self.qrow if query else self.krow};')
        self.emit(f'setp.lt.u32 {pred}, {row}, {limit};')
        return '@'+pred+' '

    def operation(self,op):
        if op is self.pv_op and getattr(self, 'defer_v_wait', False):
            self.emit('cp.async.wait_group 0;')
            self.emit('bar.sync 0;')
            self.emit('fence.proxy.async.shared::cta;')
        super().operation(op)
        if op is not self.logits or not (self.causal or self.kv_rows%64):
            return
        v,=op['results']
        for reg,slots in zip(self.regs(v),self.tensor_slots(v),strict=True):
            k=self.map([x%64 for x in slots]);pred=self.tmp('pred')
            self.emit(f'add.u32 {k}, {k}, {self.krow};')
            if self.causal:
                q=self.map([x//64 for x in slots])
                self.emit(f'add.u32 {q}, {q}, {self.qrow};')
                self.emit(f'setp.le.u32 {pred}, {k}, {q};')
            if self.kv_rows%64:
                combo='.and' if self.causal else ''
                extra=', '+pred if self.causal else ''
                self.emit(f'setp.lt{combo}.u32 {pred}, {k}, {self.kv_rows}{extra};')
            self.emit(f'selp.f32 {reg}, {reg}, 0fff800000, {pred};')
