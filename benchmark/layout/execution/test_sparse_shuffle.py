"""Exact-support Shuffle tests; no compiler or CUDA required."""
import copy
import random
import re
import unittest
import backend
from redistribution import shuffle_source_support
from demand_coverage_audit import conversion_structure


def gf2_case(register_bits=3, rank=1, source_replication=1, target_replication=1):
    """Invertible affine register/thread shear, including all seven thread bits.

    High register replicas have zero GF(2) columns. A register-controlled lane-bit shear guarantees the
    Shuffle route; warp coordinates stay fixed while source registers can
    depend on warp bits. No historical graph, shape or layout ID is used.
    """
    assert 0 <= rank <= 7 and rank < register_bits
    base=1 << register_bits;nr=base*source_replication;dr=base*target_replication
    source=[t*base+(r%base) for t in range(128) for r in range(nr)]
    destination=[((t//32)*32+((t%32)^((r%base)>>rank&1)))*base+((r%base)^(t&((1<<rank)-1)))
                 for t in range(128) for r in range(dr)]
    first={};local={};warp={}
    for i,slot in enumerate(source):
        first.setdefault(slot,i);local.setdefault((slot,i//nr),i);warp.setdefault((slot,i//nr//32),i)
    def owner(i):return dict(reg=i%nr,lane=(i//nr)%32,warp=i//nr//32,warp_group=0)
    entries=[dict(slot=slot,source=owner(local.get((slot,i//dr),warp[(slot,i//dr//32)]))) for i,slot in enumerate(destination)]
    plan=dict(mode='Shuffle',threads=128,warps=4,source_registers=nr,destination_registers=dr,
              volume=128*base,scratch_bytes=0,payload=dict(bits=32,words=[dict(bit_offset=0,live_bits=32)]),
              destinations=entries,source_slots=source,global_writers=[owner(first[s]) for s in range(128*base)])
    values={}
    for i,n,slots in [(0,nr,source),(1,dr,destination)]:
        elected={}
        for index,slot in enumerate(slots):elected.setdefault(slot,index)
        values[i]=dict(id=i,type='tensor<'+str(128*base)+'xf32>',distributed=dict(shape=[128*base],
                       topology=[n,32,4,1,1],logical=slots,first=[elected[s] for s in range(128*base)]))
    return dict(name='frisk.convert_layout',operands=[0],results=[1],redistribution=plan),values


def dense_reference(plan,source_words):
    """Original all-source-register shuffle plus receiver-mux CPU oracle."""
    nr=plan['source_registers'];dr=plan['destination_registers'];out=[]
    for t in range(128):
        for r in range(dr):
            o=plan['destinations'][t*dr+r]['source'];sender=(t//32)*32+o['lane']
            candidates=[source_words[sender*nr+i] for i in range(nr)]
            out.append(candidates[o['reg']])
    return out


def emitted_fragment(operation,values,policy='legacy'):
    e=backend.Emitter(dict(values=list(values.values())),shuffle_network_policy=policy);nr=operation['redistribution']['source_registers']
    e.reg[0]=['s'+str(i) for i in range(nr)];e.convert(operation)
    return e,'\n'.join(e.code)


def interpret_fragment(emitter,nr,words):
    """Interpret actual emitted integer/mov/shuffle PTX over all128 threads."""
    registers={'tid':list(range(128)),'lane':[t&31 for t in range(128)]};registers.update({'s'+str(r):words[r::nr] for r in range(nr)})
    def value(token):return registers[token] if token in registers else [int(token,0)&0xffffffff]*128
    for line in emitter.code:
        predicate=None
        if line.startswith('@'):predicate,line=line.split(' ',1);predicate=registers[predicate[1:]]
        opcode,args=line.rstrip(';').split(' ',1);args=[a.strip() for a in args.split(',')];dst=args[0]
        if opcode.startswith('mov.'):
            result=value(args[1]).copy()
            if predicate is not None:result=[x if p else old for x,p,old in zip(result,predicate,registers[dst])]
        elif opcode=='bfe.u32':result=[(x>>int(args[2]))&((1<<int(args[3]))-1) for x in value(args[1])]
        elif opcode=='neg.s32':result=[(-x)&0xffffffff for x in value(args[1])]
        elif opcode=='and.b32':result=[x&y for x,y in zip(value(args[1]),value(args[2]))]
        elif opcode=='xor.b32':result=[x^y for x,y in zip(value(args[1]),value(args[2]))]
        elif opcode=='popc.b32':result=[x.bit_count() for x in value(args[1])]
        elif opcode=='setp.eq.u32':result=[x==y for x,y in zip(value(args[1]),value(args[2]))]
        elif opcode=='setp.ne.u32':result=[x!=y for x,y in zip(value(args[1]),value(args[2]))]
        elif opcode=='selp.b32':result=[a if p else b for a,b,p in zip(value(args[1]),value(args[2]),value(args[3]))]
        elif opcode=='shfl.sync.bfly.b32':
            assert predicate is None and args[3:]==['31','-1']
            source=value(args[1]);masks=value(args[2]);result=[source[(t//32)*32+((t&31)^(masks[t]&31))] for t in range(128)]
        elif opcode=='shfl.sync.idx.b32':
            assert predicate is None and args[3:]==['31','-1']
            source=value(args[1]);lanes=value(args[2]);result=[source[(t//32)*32+(lanes[t]&31)] for t in range(128)]
        else:raise AssertionError('unexpected PTX '+line)
        registers[dst]=result
    output=emitter.reg[1]
    return [registers[r][t] for t in range(128) for r in output]


class SparseShuffleTest(unittest.TestCase):
    def test_gf2_image_support_rank_zero_to_seven_and_warp_union(self):
        for rank in range(8):
            op,values=gf2_case(max(3,rank+1),rank);p=backend.validate_redistribution(op,values);dr=p['destination_registers']
            for r in range(dr):
                support=shuffle_source_support(p,r)
                self.assertEqual(len(support),1<<rank)
                self.assertEqual(support,tuple(sorted({p['destinations'][t*dr+r]['source']['reg'] for t in range(128)})))
            if rank>5:
                one_warp={p['destinations'][t*dr]['source']['reg'] for t in range(32)}
                self.assertGreater(len(shuffle_source_support(p,0)),len(one_warp))

    def test_actual_ptx_bitwise_matches_original_dense_reference(self):
        rng=random.Random(20261106)
        special=[0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0x7fa00001,0xffffffff]
        for bits,rank,replica,target in [(3,0,1,1),(3,1,1,1),(4,3,1,1),(3,2,2,1),(4,3,1,2),(5,4,1,1)]:
            op,values=gf2_case(bits,rank,replica,target);nr=op['redistribution']['source_registers']
            words=[special[i%len(special)] if i%3 else rng.getrandbits(32) for i in range(128*nr)]
            emitter,code=emitted_fragment(op,values)
            self.assertEqual(interpret_fragment(emitter,nr,words),dense_reference(op['redistribution'],words))
            self.assertFalse(any('@' in x for x in code.splitlines() if 'shfl.sync' in x))

    def test_nonzero_singleton_indices_are_not_renumbered(self):
        op,v=gf2_case(3,0);p=op['redistribution'];self.assertEqual(shuffle_source_support(p,7),(7,))
        e,code=emitted_fragment(op,v);self.assertEqual(code.count('shfl.sync.idx.b32'),8)
        self.assertEqual(code.count('setp.eq.u32'),0)
        words=list(range(128*8));self.assertEqual(interpret_fragment(e,8,words),dense_reference(p,words))

    def test_codegen_and_independent_audit_count_actual_support(self):
        for bits,rank in [(3,0),(3,1),(6,5),(8,5)]:
            op,v=gf2_case(bits,rank);e,code=emitted_fragment(op,v);p=op['redistribution'];dr=p['destination_registers'];count=dr*(1<<rank)
            self.assertEqual(code.count('shfl.sync.idx.b32'),count)
            self.assertEqual(code.count('setp.eq.u32'),count-dr)
            audit=conversion_structure(e,0,1,p,[(r,0,0,0,0,0,0,0) for r in range(dr)],False)
            self.assertEqual(audit['shuffle_warp_issues'],4*count)
            self.assertEqual(audit['bytes'],count*128*4)
            self.assertEqual(audit['address_alu_warp_issues'],4*(len(e.code)-count))

    def test_512_boundary_all65536_holders_and_rank7_work_cap(self):
        op,v=gf2_case(9,7);p=backend.validate_redistribution(op,v)
        self.assertEqual(p['volume'],65536)
        self.assertEqual(sum(len(shuffle_source_support(p,r)) for r in range(512)),65536)
        words=[(i*2654435761)&0xffffffff for i in range(65536)]
        expected=dense_reference(p,words)
        for i,entry in enumerate(p['destinations']):
            owner=entry['source'];self.assertEqual(owner['warp'],i//512//32)
            index=(owner['warp']*32+owner['lane'])*512+owner['reg']
            self.assertEqual(v[0]['distributed']['logical'][index],v[1]['distributed']['logical'][i])
            self.assertEqual(expected[i],words[index])
        too_large,values=gf2_case(9,7,source_replication=2)
        with self.assertRaisesRegex(ValueError,'bounded domain'):backend.validate_redistribution(too_large,values)

    def test_irregular_finite_support_and_malformed_inputs(self):
        op,v=gf2_case(3,2);p=op['redistribution'];q=copy.deepcopy(p)
        q['destinations'][3*q['destination_registers']]['source']['reg']=7
        self.assertEqual(set(shuffle_source_support(q,0)),{q['destinations'][t*q['destination_registers']]['source']['reg'] for t in range(128)})
        for register in [-1,True,p['destination_registers']]:
            with self.assertRaises(ValueError):shuffle_source_support(p,register)
        for reg in [True,-1,p['source_registers']]:
            q=copy.deepcopy(p);q['destinations'][0]['source']['reg']=reg
            with self.assertRaises(ValueError):shuffle_source_support(q,0)
        q=copy.deepcopy(p);q['destinations'].pop()
        with self.assertRaises(ValueError):shuffle_source_support(q,0)
        q=copy.deepcopy(p);q['mode']='Shared'
        with self.assertRaises(ValueError):shuffle_source_support(q,0)

if __name__=='__main__':unittest.main()
