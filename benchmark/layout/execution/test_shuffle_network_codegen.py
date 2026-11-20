"""Actual emitted PTX versus independent elected-owner bit payload reference."""
import random
import unittest
import backend
from demand_coverage_audit import conversion_structure
from shuffle_network import plan_shuffle_network, shuffle_network_candidates
from test_sparse_shuffle import gf2_case, emitted_fragment, interpret_fragment


def permutation_case(register_bits, pairs):
    registers=1<<register_bits
    source=list(range(128*registers));dest=[]
    for thread in range(128):
        for reg in range(registers):
            lane=thread&31;rr=reg
            for register_bit,lane_bit in pairs:
                delta=((rr>>register_bit)^(lane>>lane_bit))&1
                rr^=delta<<register_bit;lane^=delta<<lane_bit
            dest.append(((thread//32)*32+lane)*registers+rr)
    def owner(index):return dict(reg=index%registers,lane=(index//registers)&31,warp=index//(32*registers),warp_group=0)
    plan=dict(mode='Shuffle',threads=128,warps=4,source_registers=registers,destination_registers=registers,
              volume=len(source),scratch_bytes=0,payload=dict(bits=32,words=[dict(bit_offset=0,live_bits=32)]),
              destinations=[dict(slot=slot,source=owner(slot)) for slot in dest],source_slots=source,
              global_writers=[owner(i) for i in source])
    values={}
    for ident,slots in [(0,source),(1,dest)]:
        first=[0]*len(source)
        for index,slot in enumerate(slots):first[slot]=index
        values[ident]=dict(id=ident,type=f'tensor<{len(source)}xf32>',distributed=dict(shape=[len(source)],
                            topology=[registers,32,4,1,1],logical=slots,first=first))
    return dict(name='frisk.convert_layout',operands=[0],results=[1],redistribution=plan),values


def linear_case(register_bits,columns,offsets):
    """Independent full finite affine fixture, preserving warp identities."""
    registers=1<<register_bits;source=list(range(128*registers));dest=[]
    for thread in range(128):
        for reg in range(registers):
            x=(thread&31)*registers+reg;y=offsets[thread//32]
            for bit,column in enumerate(columns):
                if x>>bit&1:y^=column
            dest.append((thread//32)*32*registers+y)
    def owner(index):return dict(reg=index%registers,lane=(index//registers)&31,warp=index//(32*registers),warp_group=0)
    plan=dict(mode='Shuffle',threads=128,warps=4,source_registers=registers,destination_registers=registers,
              volume=len(source),scratch_bytes=0,payload=dict(bits=32,words=[dict(bit_offset=0,live_bits=32)]),
              destinations=[dict(slot=slot,source=owner(slot)) for slot in dest],source_slots=source,
              global_writers=[owner(i) for i in source])
    values={}
    for ident,slots in [(0,source),(1,dest)]:
        first=[0]*len(source)
        for index,slot in enumerate(slots):first[slot]=index
        values[ident]=dict(id=ident,type=f'tensor<{len(source)}xf32>',distributed=dict(shape=[len(source)],
                            topology=[registers,32,4,1,1],logical=slots,first=first))
    return dict(name='frisk.convert_layout',operands=[0],results=[1],redistribution=plan),values


def expected_words(plan,words):
    registers=plan['source_registers']
    return [words[((x['source']['warp']*32+x['source']['lane'])*registers)+x['source']['reg']]
            for x in plan['destinations']]


class ShuffleNetworkCodegenTest(unittest.TestCase):
    def words(self,n):
        rng=random.Random(20261104+n)
        # NaN payloads, signed zero, infinities and unstructured words are
        # transported as bits, without a float comparison hiding corruption.
        special=[0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0x7fa00001,0xffffffff]
        return [special[i%len(special)] if i%3 else rng.getrandbits(32) for i in range(n)]

    def check(self,operation,values,all_alternatives=False):
        p=backend.validate_redistribution(operation,values);nr=p['source_registers'];words=self.words(128*nr)
        emitter,code=emitted_fragment(operation,values,policy='auto')
        self.assertEqual(interpret_fragment(emitter,nr,words),expected_words(p,words))
        shfl=sum(line.startswith('shfl.sync.') for line in emitter.code)
        self.assertFalse(any('@' in line for line in emitter.code if 'shfl.sync.' in line))
        audit=conversion_structure(emitter,0,1,p,[(r,0,0,0,0,0,0,0) for r in range(p['destination_registers'])],False)
        self.assertEqual(audit['shuffle_warp_issues'],4*shfl)
        self.assertEqual(audit['address_alu_warp_issues'],4*(len(emitter.code)-shfl))
        if all_alternatives:
            for network in shuffle_network_candidates(p):
                trace=backend.Emitter({'values':list(values.values())});trace.reg[1]=trace._emit_shuffle_network(network,['s'+str(i) for i in range(nr)])
                self.assertEqual(interpret_fragment(trace,nr,words),expected_words(p,words),network['kind'])
        return emitter,plan_shuffle_network(p)

    def test_actual_shear_network_and_warp_parameter_bits(self):
        for bits,rank in [(3,0),(3,1),(4,3),(6,5),(8,6),(9,7)]:
            op,values=gf2_case(bits,rank)
            emitter,network=self.check(op,values)
            self.assertIsNotNone(network)
            self.assertTrue(emitter.shuffle_networks)

    def test_actual_ship_and_fused_permutation_networks(self):
        for bits,pairs in [(2,[(0,0)]),(3,[(0,2),(2,0)]),(5,[(i,4-i) for i in range(5)]),(9,[(i,i) for i in range(5)])]:
            op,values=permutation_case(bits,pairs)
            self.check(op,values,all_alternatives=True)

    def test_replication_remains_bitwise_exact_support_fallback(self):
        for a,b in [(2,1),(1,2),(2,2)]:
            op,values=gf2_case(3,2,a,b)
            emitter,network=self.check(op,values)
            self.assertIsNone(network)
            self.assertFalse(emitter.shuffle_networks)

    def test_dense_counterexample_does_not_increase_actual_shuffle_count(self):
        op,values=linear_case(3,[1,38,226,7,21,249,120,225],[0]*4)
        emitter,network=self.check(op,values)
        self.assertLessEqual(sum(x.startswith('shfl.sync.') for x in emitter.code),64)

    def test_exported_network_declaration_cannot_be_tampered(self):
        import copy
        op,values=permutation_case(3,[(0,0)])
        op['redistribution']['shuffle_network']=plan_shuffle_network(op['redistribution'])
        backend.validate_redistribution(op,values)
        bad=copy.deepcopy(op);bad['redistribution']['shuffle_network']['counts']['shfl']+=1
        with self.assertRaisesRegex(ValueError,'declaration differs'):
            backend.validate_redistribution(bad,values)

    def test_unknown_policy_and_false_counts_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'policy'):backend.Emitter({'values':[]},shuffle_network_policy='guess')
        op,values=permutation_case(3,[(0,0)])
        network=plan_shuffle_network(op['redistribution']);network['counts']['shfl']+=1
        with self.assertRaisesRegex(ValueError,'differs from proof'):
            backend.Emitter({'values':[]})._emit_shuffle_network(network,['s'+str(i) for i in range(8)])


if __name__=='__main__':unittest.main()
