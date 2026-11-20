"""Native multi-CTA launch of Frisk's two-warpgroup online attention."""
from wide_execution import WideEmitter


class NativeOnlineEmitter(WideEmitter):
    def run(self):
        ops = self.data['operations']
        module, = [op for op in ops if op['name'] == 'builtin.module']
        cfg = module['attrs']['frisk.online_launch']
        if set(cfg)-{'kv_rows'} != {'rows', 'heads', 'kv_heads', 'causal'}:
            raise ValueError('invalid online launch fields')
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
        self.kcopy = copies[1]
        ss,rs = [op for op in ops if op['name']=='frisk.mma']
        self.qtile, self.dim = ss['attrs']['m'], ss['attrs']['k']
        self.value_dim = rs['attrs']['n']
        if (self.value_dim not in (32,64,128) or self.qtile not in (128,256,512) or self.dim not in (64,128,256) or
                ss['attrs']['n'] != 64 or
                [rs['attrs'][k] for k in ('m','n','k')] != [self.qtile,self.value_dim,64]):
            raise ValueError('incompatible online QK/PV tile geometry')
        self.qtiles = (self.rows+self.qtile-1)//self.qtile
        if not (lo < ops.index(ss) < ops.index(rs) < hi and
                [o['attrs']['frisk.mma_contract']['form'] for o in (ss,rs)]==['ss','rs']):
            raise ValueError('expected a certified QK/PV pair')
        self.logits, = [op for op in ops if op['name']=='arith.mulf' and ss['results'][0] in op['operands']]
        store, = [op for op in ops if op['name']=='frisk.tile_store']
        if self.storage(store['operands'][1])['root'] != self.out or ops.index(store) <= hi:
            raise ValueError('expected one post-loop output store')
        views=[op['operands'][0] for op in copies]+[store['operands'][1]]
        for view,rows,cols,width in zip(views,(self.qtile,64,64,self.qtile),
                                       (self.dim,self.dim,self.value_dim,self.value_dim),(2,2,2,4),strict=True):
            offsets=self.offsets(view)
            if (len(offsets)!=rows*cols or
                    any(offsets[i]!=i*width for i in range(rows*cols))):
                raise ValueError('packed online attention requires row-major Q/K/V/O views')
        if any(op['name']=='frisk.tile_load' for op in ops):
            raise ValueError('unexpected global input in the online graph')
        text, info = super().run()
        info.update(launch_grid=[self.qtiles*self.heads,1,1], kv_tiles=self.ktiles,
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
            self.emit(f'div.u32 {kvhead}, {head}, {self.group};')
            qbase,self.kbase = self.tmp(),self.tmp()
            self.emit(f'mad.lo.u32 {qbase}, {head}, {self.rows}, {self.qrow};')
            self.emit(f'mul.lo.u32 {self.kbase}, {kvhead}, {self.kv_rows};')
            self.ptr = {r:self.tmp('b64') for r in (self.q,self.k,self.vroot,self.out)}
            self.set_pointer(self.q,qbase,2*self.dim)
            self.set_pointer(self.out,qbase,4*self.value_dim)
        super().sequence(ops,begin,end)

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
            self.set_pointer(self.vroot,row,2*self.value_dim)
        super().wide_copy(op)

    def access_guard(self,root,offset):
        query = root in (self.q,self.out)
        limit = self.rows if query else self.kv_rows
        if limit % (self.qtile if query else 64)==0:
            return ''
        row=self.tmp();pred=self.tmp('pred')
        cols = self.value_dim if root in (self.out,self.vroot) else self.dim
        self.emit(f'shr.u32 {row}, {offset}, {(cols * (4 if root==self.out else 2)).bit_length()-1};')
        self.emit(f'add.u32 {row}, {row}, {self.qrow if query else self.krow};')
        self.emit(f'setp.lt.u32 {pred}, {row}, {limit};')
        return '@'+pred+' '

    def operation(self,op):
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
