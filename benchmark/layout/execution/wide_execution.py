"""Two-warpgroup SM90a execution of a verified Frisk SSA packet.

The emitter consumes copy owners/addresses, MMA descriptors, reduction trees
and conversions from the ordinary exporter. Static single-CTA programs may
contain uniform counted loops and result-free conditional copy regions.
"""

from backend import Emitter, splat_f32_bits
from redistribution import validate
from shared_allocation import plan_shared_allocation


class WideEmitter(Emitter):
    threads = 256

    def tensor_slots(self, value):
        layout = self.v[value]['distributed']
        nr, *topology = layout['topology']
        if topology != [32, 4, self.threads//128, 1] or len(layout['logical']) != self.threads * nr:
            raise ValueError('inconsistent two-warpgroup carrier')
        return [layout['logical'][r::nr] for r in range(nr)]

    def validate_route(self, op, values):
        return validate(op, values, threads=self.threads)

    def access_guard(self, root, offset):
        return ""

    def wide_copy(self, op):
        a, b = op['operands']
        layout = op['execution']
        attrs = op['attrs']
        contract = attrs['frisk.copy_contract']
        if (contract['path'] != 'vector' or contract['vector_bytes'] not in (1, 2)
                or contract['completion'] != 'synchronous' or contract.get('guarded')
                or attrs['frisk.writer_policy'] != 'first_owner'
                or layout['topology'][1:] != [32,4,self.threads//128,1]
                or any('xf16' not in self.v[x]['type'] for x in (a,b))
                or any(attrs[key] != 'affine_map<() -> ()>' for key in ('srcMap','dstMap'))
                or attrs['srcExtents'] != attrs['dstExtents'] or attrs['srcExtents'] != layout['shape']):
            raise ValueError('wide copy requires a synchronous whole f16 tile')
        src, dst = self.offsets(a), self.offsets(b)
        nr = layout['topology'][0]
        # A wait_group is warpgroup-local. Both groups must retire reads of
        # the previous K/V tile before either starts overwriting its storage.
        self.emit('bar.sync 0;')
        for r in range(nr):
            slots = layout['logical'][r::nr]
            if len(slots) != self.threads or any(layout['first'][x] != t*nr+r for t,x in enumerate(slots)):
                raise ValueError('unsupported replicated copy')
            root = self.storage(a)['root']
            offset = self.map([src[x] for x in slots])
            sa = self.address(root, offset)
            guard = self.access_guard(root, offset)
            da = self.shared(b, self.map([dst[x] for x in slots]))
            value = self.tmp()
            if contract['vector_bytes'] == 1:
                hi = self.tmp()
                if guard:
                    self.emit(f'mov.u32 {value}, 0;')
                    self.emit(f'mov.u32 {hi}, 0;')
                self.emit(f'{guard}ld.global.u8 {value}, [{sa}];')
                self.emit(f'{guard}ld.global.u8 {hi}, [{sa}+1];')
                self.emit(f'shl.b32 {hi}, {hi}, 8;')
                self.emit(f'or.b32 {value}, {value}, {hi};')
            else:
                value = self.tmp('b16')
                if guard: self.emit(f'mov.b16 {value}, 0;')
                self.emit(f'{guard}ld.global.b16 {value}, [{sa}];')
            self.emit(f'st.shared.b16 [{da}], {value};')
        self.emit('bar.sync 0;')
        self.emit('fence.proxy.async.shared::cta;')

    def group_descriptor(self, value, desc, starts, k):
        entries = desc['entries']
        table = {(mn, col): (offset, phase) for mn,col,offset,phase in zip(*[iter(entries)]*4)}
        if len(table)*4 != len(entries):
            raise ValueError('duplicate descriptor entry')
        mode = {0:0,32:3,64:2,128:1}[desc['swizzle']]
        values = []
        for start in starts:
            off, phase = table[start,k]
            base = self.smem[self.storage(value)['root']] + off
            if base % 16 or base >= 1<<18 or not 0 <= phase < 8:
                raise ValueError('invalid descriptor address')
            values.append((base,phase))
        addr = self.map([values[t//128][0] for t in range(self.threads)])
        phase = self.map([values[t//128][1] for t in range(self.threads)])
        packed = self.tmp('b64')
        high = self.tmp('b64')
        self.emit(f'add.u32 {addr}, smem, {addr};')
        self.emit(f'shr.u32 {addr}, {addr}, 4;')
        self.emit(f'cvt.u64.u32 {packed}, {addr};')
        self.emit(f'cvt.u64.u32 {high}, {phase};')
        self.emit(f'shl.b64 {high}, {high}, 49;')
        bits = (desc['leading']//16 << 16) | (desc['stride']//16 << 32) | (mode << 62)
        self.emit(f'or.b64 {packed}, {packed}, {high};')
        self.emit(f'or.b64 {packed}, {packed}, {bits};')
        return packed

    def wide_mma(self, op):
        a,b,c = op['operands']
        result, = op['results']
        d = op['attrs']['frisk.mma_contract']
        grid, atom = d['grid'], d['atom']
        if (d['target'] != 'sm_90a' or d['version'] != 1 or atom not in ([64,32,16], [64,64,16], [64,128,16])
                or grid != [self.threads//128,1] or d['repeats'][1] != 1
                or d['input_type'] != 'f16' or d['accumulator_type'] != 'f32'):
            raise ValueError('wide WGMMA requires a certified 2x1 m64nNk16 contract')
        rows, _, depth = d['repeats']
        cols = atom[1]
        width = cols // 2
        if rows < 1 or depth < 1 or depth % self.mma_group_size:
            raise ValueError('WGMMA wait group must divide each atom accumulation')
        rs = d['form'] == 'rs'
        if ((d['form'], d['packing']) not in [('ss','none'),('rs','f16x2-low-high')]
                or ('a_descriptor' in d) == rs
                or (not rs and (d['a_descriptor']['major'] != 'k' or not op['attrs'].get('trans_b')))
                or d['b_descriptor']['major'] != ('mn' if rs else 'k')):
            raise ValueError('unsupported WGMMA operand/descriptor convention')
        slots = self.tensor_slots(result)
        if slots != self.tensor_slots(c):
            raise ValueError('incompatible accumulator operand')
        interleaved = d.get('row_partition') == 'interleaved64'

        def start(r,w):
            return 64*(2*r+w if interleaved else rows*w+r)

        def element(t,r,i):
            return start(r,t//128) + ((t//32)%4)*16 + (t%32)//4 + ((i//2)%2)*8

        def column(t,i):
            return 2*(t%4)+i%2+(i//4)*8

        out = self.newregs(result)
        for i,(z,old) in enumerate(zip(out,self.regs(c),strict=True)):
            if slots[i] != [element(t,i//width,i%width)*cols+column(t,i%width) for t in range(self.threads)]:
                raise ValueError('WGMMA accumulator ownership mismatch')
            self.emit(f'mov.f32 {z}, {old};')
        source = self.tensor_slots(a) if rs else None
        for r in range(rows):
            pending = []
            for k in range(depth):
                bd = self.group_descriptor(b,d['b_descriptor'],[0]*(self.threads//128),16*k)
                if rs:
                    packed = []
                    for j in range(4):
                        i = (r*depth+k)*8 + j*2
                        for h in (0,1):
                            expected = [element(t,r,2*j+h)*(16*depth)+16*k+column(t,2*j+h) for t in range(self.threads)]
                            if source[i+h] != expected:
                                raise ValueError('WGMMA RS packing disagrees with holder map')
                        z = self.tmp()
                        self.emit('mov.b32 '+z+', {'+', '.join(self.regs(a)[i:i+2])+'};')
                        packed.append(z)
                    operands = '{'+', '.join(packed)+'}, '+bd+', mma_acc, 1, 1, 1'
                else:
                    ad = self.group_descriptor(a,d['a_descriptor'],[start(r,w) for w in range(self.threads//128)],16*k)
                    operands = f'{ad}, {bd}, mma_acc, 1, 1, 0, 0'
                pending.append(operands)
                if len(pending) == self.mma_group_size:
                    # All RS fragments and descriptors are ready before the fence.
                    # Only same-shape WGMMA touches this accumulator before wait 0.
                    self.emit('setp.eq.u32 mma_acc, 0, 0;')
                    self.emit('wgmma.fence.sync.aligned;')
                    for arguments in pending:
                        self.emit(f'wgmma.mma_async.sync.aligned.m64n{cols}k16.f32.f16.f16 {{'+
                                  ', '.join(out[r*width:(r+1)*width])+'}, '+arguments+';')
                    self.emit('wgmma.commit_group.sync.aligned;')
                    self.emit('wgmma.wait_group.sync.aligned 0;')
                    pending.clear()

    def local_mux(self, registers, indices):
        support = sorted(set(indices))
        if len(support)==1:
            return registers[support[0]]
        selector = self.map(indices)
        result = registers[support[0]]
        for i in support[1:]:
            pred = self.tmp('pred')
            self.emit(f'setp.eq.u32 {pred}, {selector}, {i};')
            other = self.tmp()
            self.emit(f'selp.b32 {other}, {registers[i]}, {result}, {pred};')
            result = other
        return result

    def wide_convert(self, op):
        a, = op['operands']
        v, = op['results']
        plan = self.validate_route(op,self.v)
        bits = plan['payload']['bits']
        width = bits//8
        out = self.newregs(v)
        source = self.regs(a)
        if plan['mode'] != 'Shared':
            if bits==16:
                words = []
                for reg in source:
                    word = self.tmp()
                    self.emit(f'cvt.u32.u16 {word}, {reg};')
                    words.append(word)
                source = words
            for r,z in enumerate(out):
                owners = [p['source'] for p in plan['destinations'][r::len(out)]]
                indices = [o['reg'] for o in owners]
                chosen = source
                if plan['mode'] == 'Shuffle':
                    lane = self.map([o['lane'] for o in owners])
                    chosen = {}
                    for i in sorted(set(indices)):
                        q = self.tmp()
                        self.emit(f'shfl.sync.idx.b32 {q}, {source[i]}, {lane}, 31, -1;')
                        chosen[i] = q
                word = self.local_mux(chosen,indices)
                self.emit(f'cvt.u16.u32 {z}, {word};' if bits==16 else f'mov.b32 {z}, {word};')
            return
        if 'stripe_elements' in plan:
            self.striped_exchange(plan, a, v, source, out, bits)
            return
        for r,slots in enumerate(self.tensor_slots(a)):
            addr = self.map([width*x for x in slots])
            elected = [int(plan['global_writers'][slot]==dict(reg=r,lane=t%32,warp=(t//32)%4,warp_group=t//128)) for t,slot in enumerate(slots)]
            if not any(elected):
                continue
            prefix = ''
            if not all(elected):
                predicate = self.tmp('pred')
                self.emit(f'setp.ne.u32 {predicate}, {self.map(elected)}, 0;')
                prefix = '@'+predicate+' '
            self.emit(f'add.u32 {addr}, scratch, {addr};')
            self.emit(f'{prefix}st.shared.b{bits} [{addr}], {source[r]};')
        self.emit('bar.sync 0;')
        for z,slots in zip(out,self.tensor_slots(v),strict=True):
            addr = self.map([width*x for x in slots])
            self.emit(f'add.u32 {addr}, scratch, {addr};')
            self.emit(f'ld.shared.b{bits} {z}, [{addr}];')
        self.emit('bar.sync 0;')

    def stripe_predicate(self, active):
        # The intersection of election and stripe membership can constrain
        # several thread bits at once. A GF(2) scalar map cannot encode AND.
        ids = [tid for tid, member in enumerate(active) if member]
        first = ids[0]
        fixed = self.threads - 1
        for tid in ids:
            fixed &= ~(tid ^ first)
        pred = self.tmp('pred')
        field = self.tmp()
        if len(ids) == self.threads >> fixed.bit_count():
            self.emit(f'and.b32 {field}, tid, {fixed};')
            self.emit(f'setp.eq.u32 {pred}, {field}, {first & fixed};')
            return pred
        # Finite non-cubical elections are general masks, one uint32 per warp.
        masks = [sum(active[w*32 + lane] << lane for lane in range(32))
                 for w in range(self.threads // 32)]
        warp = self.tmp()
        self.emit(f'shr.u32 {warp}, tid, 5;')
        self.emit(f'mov.u32 {field}, {masks[0]};')
        for w, mask in enumerate(masks[1:], 1):
            if mask != masks[0]:
                self.emit(f'setp.eq.u32 {pred}, {warp}, {w};')
                self.emit(f'@{pred} mov.u32 {field}, {mask};')
        self.emit(f'shr.u32 {field}, {field}, lane;')
        self.emit(f'and.b32 {field}, {field}, 1;')
        self.emit(f'setp.ne.u32 {pred}, {field}, 0;')
        return pred

    def striped_exchange(self, plan, a, v, source, out, bits):
        width = bits // 8
        # Owners and logical slots are proven globally. Publish and gather a
        # bounded interval of those slots, then retire all readers before reuse.
        stripe = plan['stripe_elements']
        sources = self.tensor_slots(a)
        destinations = self.tensor_slots(v)
        writers = [[int(plan['global_writers'][slot] == dict(reg=r, lane=t%32,
                       warp=t//32%4, warp_group=t//128)) for t, slot in enumerate(slots)]
                   for r, slots in enumerate(sources)]

        def transfer(slots, active, reg, store):
            if not any(active):
                return
            prefix = ''
            if not all(active):
                prefix = '@' + self.stripe_predicate(active) + ' '
            addr = self.tmp()
            offset = self.map([width * (slot % stripe) for slot in slots])
            self.emit(f'add.u32 {addr}, scratch, {offset};')
            if store:
                self.emit(f'{prefix}st.shared.b{bits} [{addr}], {reg};')
            else:
                self.emit(f'{prefix}ld.shared.b{bits} {reg}, [{addr}];')

        for start in range(0, plan['volume'], stripe):
            end = start + stripe
            for reg, slots, elected in zip(source, sources, writers, strict=True):
                transfer(slots, [int(e and start <= x < end) for e,x in zip(elected,slots)],
                         reg, True)
            self.emit('bar.sync 0;')
            for reg, slots in zip(out, destinations, strict=True):
                transfer(slots, [int(start <= x < end) for x in slots], reg, False)
            self.emit('bar.sync 0;')

    def global_io(self,op,store):
        value,buffer = op['operands'] if store else (op['results'][0],op['operands'][0])
        if ('xf32' not in self.v[value]['type'].split(',')[0] or
                'xf32' not in self.v[buffer]['type'].split(',')[0]):
            raise ValueError('wide global I/O requires matching f32 carriers')
        dist = self.v[value]['distributed']
        regs = self.regs(value) if store else self.newregs(value)
        offsets = self.offsets(buffer)
        for i,slots in enumerate(self.tensor_slots(value)):
            root = self.storage(buffer)['root']
            offset = self.map([offsets[x] for x in slots])
            addr = self.address(root, offset)
            guard = self.access_guard(root, offset)
            if store:
                elected = [dist['first'][x]==t*len(regs)+i for t,x in enumerate(slots)]
                if not all(elected):
                    raise ValueError('wide store requires unreplicated destinations')
                self.emit(f'{guard}st.global.b32 [{addr}], {regs[i]};')
            else:
                self.emit(f'ld.global.b32 {regs[i]}, [{addr}];')

    def operation(self, op):
        name = op['name']
        if name in ('builtin.module','frisk.kernel','frisk.parallel','frisk.end','memref.alloc',
                    'frisk.storage_contract','frisk.layout_view','tensor.empty','linalg.yield'):
            return
        if name == 'frisk.copy': return self.wide_copy(op)
        if name == 'frisk.mma': return self.wide_mma(op)
        if name == 'frisk.reduce_tensor': return self.reduce(op)
        if name == 'linalg.broadcast': return self.broadcast(op)
        if name == 'frisk.convert_layout': return self.wide_convert(op)
        if name == 'frisk.tile_store': return self.global_io(op,True)
        if name == 'frisk.tile_load': return self.global_io(op,False)
        args = op['operands']
        value, = op['results']
        if name == 'arith.constant':
            if self.v[value]['type'] == 'index':
                self.indices[value] = op['attrs']['value']
            else:
                bits = splat_f32_bits(op['attrs']['value'])
                for z in self.newregs(value): self.emit(f'mov.b32 {z}, {bits};')
            return
        if name == 'arith.cmpi':
            if op['attrs']['predicate'] not in ('eq',0,'0 : i64','#arith.cmpipredicate<eq>'):
                raise ValueError('only uniform equality supported')
            pred = self.tmp('pred')
            self.emit(f'setp.eq.s32 {pred}, {self.indices[args[0]]}, {self.indices[args[1]]};')
            self.conditions[value] = pred
            return
        instructions = {'arith.mulf':'mul.rn.f32','arith.addf':'add.rn.f32','arith.subf':'sub.rn.f32',
                        'arith.maximumf':'max.NaN.f32','arith.divf':'div.rn.f32',
                        'arith.truncf':'cvt.rn.f16.f32','math.rsqrt':'rsqrt.approx.f32'}
        if name not in instructions and name not in ('math.exp','math.exp2'): raise ValueError('unhandled wide op '+name)
        if any(self.tensor_slots(v)!=self.tensor_slots(value) for v in args):
            raise ValueError('elementwise operand holder mismatch')
        for i,z in enumerate(self.newregs(value)):
            operands = [self.regs(v)[i] for v in args]
            if name in ('math.exp','math.exp2'):
                x = operands[0]
                if name == 'math.exp':
                    x = self.tmp('f32')
                    self.emit(f'mul.rn.f32 {x}, {operands[0]}, 0f3fb8aa3b;')
                ftz = '.ftz' if getattr(self,'flush_exp_subnormals',False) else ''
                self.emit(f'ex2.approx{ftz}.f32 {z}, {x};')
            else:
                self.emit(instructions[name]+' '+z+', '+', '.join(operands)+';')

    def loop_upper_bound(self, op, static_bound):
        return static_bound

    def sequence(self, ops, begin, end):
        i = begin
        while i < end:
            op = ops[i]
            name = op['name']
            if name not in ('scf.for','scf.if'):
                self.operation(op)
                i += 1
                continue
            tail = op['body_terminator']
            if not i < tail < end or ops[tail]['name'] != 'scf.yield':
                raise ValueError('invalid structured region')
            after = 'frisk_done_'+str(i)
            if name == 'scf.if':
                pred = self.conditions[op['operands'][0]]
                self.emit(f'@!{pred} bra.uni {after};')
                self.sequence(ops,i+1,tail)
            else:
                low,high,step = [self.indices[v] for v in op['operands'][:3]]
                if not all(type(x) is int for x in (low,high,step)) or not 0<=low<=high<2**30 or not 0<step<2**30:
                    raise ValueError('wide loop requires bounded static positive step')
                iv,*carried = op['body_arguments']
                for arg,init in zip(carried,op['operands'][3:],strict=True):
                    for z,x in zip(self.newregs(arg),self.regs(init),strict=True):
                        self.emit(f'mov.b32 {z}, {x};')
                index = self.tmp()
                self.indices[iv] = index
                self.emit(f'mov.u32 {index}, {low};')
                high = self.loop_upper_bound(op,high)
                self.emit(f'frisk_loop_{i}:')
                pred = self.tmp('pred')
                self.emit(f'setp.ge.s32 {pred}, {index}, {high};')
                self.emit(f'@{pred} bra.uni {after};')
                self.sequence(ops,i+1,tail)
                updates = []
                for arg,src in zip(carried,ops[tail]['operands'],strict=True):
                    for z,x in zip(self.regs(arg),self.regs(src),strict=True):
                        t = self.tmp()
                        self.emit(f'mov.b32 {t}, {x};')
                        updates.append((z,t))
                for z,t in updates: self.emit(f'mov.b32 {z}, {t};')
                self.emit(f'add.u32 {index}, {index}, {step};')
                self.emit(f'bra.uni frisk_loop_{i};')
                for result,arg in zip(op['results'],carried,strict=True):
                    self.reg[result] = self.regs(arg)
            self.emit(after+':')
            i = tail+1

    def run(self):
        if (self.mma_group_size not in (1, 4) or self.copy_batch or self.store_policy!='direct' or
                self.access_policy not in ('legacy','vector') or
                self.shared_policy!='declared' or self.demand_query_observer is not None or
                self.shuffle_network_policy!='auto' or self.execution_trace is not None):
            raise ValueError('wide execution requires the declared synchronous schedule')
        ops = self.data['operations']
        allocation = plan_shared_allocation(self.data,redistribution_validator=self.validate_route,execution_threads=self.threads)
        kernel, = [o for o in ops if o['name']=='frisk.kernel']
        roots = kernel['arguments']
        if set(roots)!=set(allocation['global_roots']) or len(roots)!=len(set(roots)):
            raise ValueError('wide kernel ABI requires distinct declared global arguments')
        self.roots = {v:i for i,v in enumerate(roots)}
        self.smem = {item['root_id']:item['offset'] for item in allocation['allocations']}
        shared = allocation['static_shared_bytes']
        self.indices, self.conditions = {}, {}
        self.decl += ['.reg .pred mma_acc;', '.reg .u32 tid, lane, smem, scratch;']
        self.emit('mov.u32 tid, %tid.x;')
        self.emit('and.b32 lane, tid, 31;')
        self.emit(f'mov.u32 smem, ${len(roots)+1};')
        self.emit(f'add.u32 scratch, smem, {allocation["scratch_offset"]};')
        self.sequence(ops,0,len(ops))
        asm = '{\n'+'\n'.join(self.decl+self.code+['mov.u32 $0, 0;'])+'\n}'
        asm = asm.replace('\\','\\5C').replace('"','\\22').replace('\n','\\0A')
        names = ['%arg'+str(i) for i in range(len(roots))]
        ptr = '!tt.ptr<i8>'
        text = ('module attributes {"ttg.num-warps" = '+str(self.threads//32)+' : i32, "ttg.num-ctas" = 1 : i32, '
                '"ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:90"} {\n'
                ' llvm.mlir.global external @frisk_dynamic() {addr_space = 3 : i32, alignment = 1024 : i64} : !llvm.array<0 x i8>\n'
                ' tt.func public @frisk_execution('+', '.join(n+': '+ptr for n in names)+') {\n'
                '  %sm = llvm.mlir.addressof @frisk_dynamic : !llvm.ptr<3>\n'
                '  %addr = llvm.ptrtoint %sm : !llvm.ptr<3> to i32\n'
                '  %done = tt.elementwise_inline_asm "'+asm+'" {constraints = "=r'+',l'*len(names)+',r", '
                'pure = false, packed_element = 1 : i32} '+', '.join(names)+', %addr : '+
                ', '.join([ptr]*len(names))+', i32 -> i32\n tt.return\n }\n}\n')
        return text,dict(threads=self.threads,root_order=roots,dynamic_shared_bytes=shared,
                         shared_allocation_plan=allocation,emitted_operations=len(ops),mma_group_size=self.mma_group_size)
