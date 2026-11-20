import unittest
from unittest.mock import patch

from backend import emit
from h100_paired_mainloop import paired_native_mainloop, _memory_work
from native_store_selection import _semantic_ir, _within_gap, _compile_online_image
from shared_allocation import SharedBudgetExceeded


class NativeStoreTest(unittest.TestCase):
    def test_guarded_drain_counts_only_terminal_address_issues(self):
        from h100_segment_cost import guarded_drain,GuardedDrain
        ops=['MOV R0, RZ','HGMMA.64x64x16.F32 R5, R2, R4','BRA 0x0',
             'CALL.REL.NOINC 0xc0','BSYNC B1','IMAD R1, R2, R3, R4',
             '@P0 STG.E desc[UR4][R12.64], R6','LOP3.LUT R2, R1, R3, RZ, 0xc0, !PT',
             'IADD3 R9, R2, RZ, RZ','@P0 EXIT','STG.E desc[UR4][R8.64], R6','EXIT']
        fmt=lambda ls:chr(10).join(f' /*{i*16:04x}*/ {op} ;' for i,op in enumerate(ls))
        self.assertEqual(guarded_drain(fmt(ops)),GuardedDrain(3,2))
        self.assertIsNone(guarded_drain(fmt([op.replace('@P0 STG','STG') for op in ops])))
        self.assertIsNone(guarded_drain(fmt(ops[:8]+['BRA 0x60']+ops[9:])))
        self.assertIsNone(guarded_drain(fmt([op.replace('STG.E ','STG.E.128 ') for op in ops])))

    def test_native_value_width_preserves_default_and_changes_pv_only(self):
        from native_online_fixture import online_fixture
        self.assertEqual(online_fixture(256,64,193),online_fixture(256,64,193,value_dim=64))
        ir=online_fixture(256,64,193,kv_rows=241,value_dim=128)
        self.assertIn('memref<64x128xf16,1>',ir)
        self.assertIn('memref<256x128xf32,1>',ir)
        self.assertIn('n=128:i64',next(z for z in ir.splitlines() if z.startswith('%out1 =')))
        self.assertIn('n=64:i64',next(z for z in ir.splitlines() if z.startswith('%s1 =')))
        with self.assertRaises(ValueError):online_fixture(256,64,193,value_dim=21)

    def test_wide_qk_with_independent_value_dimension(self):
        from native_online_fixture import online_fixture
        ir = online_fixture(128, 256, 109, kv_rows=217, value_dim=128)
        qk = next(line for line in ir.splitlines() if line.startswith('%s1 ='))
        pv = next(line for line in ir.splitlines() if line.startswith('%out1 ='))
        self.assertIn('k=256:i64', qk)
        self.assertIn('n=128:i64', pv)
        self.assertIn('memref<128x128xf32,1>', ir)
        with self.assertRaises(ValueError):
            online_fixture(128, 256, 109)

    def test_segment_work_respects_native_region_boundaries(self):
        from h100_segment_cost import segment_work
        sass = ''' /*0000*/ MOV R1, RZ ;
 /*0010*/ LDL.64 R2, [R1+0x4] ;
 /*0020*/ STL [R1+0x8], R7 ;
 /*0030*/ HGMMA.64x64x16.F32 R2, R4, R6 ;
 /*0040*/ BRA 0x10 ;
 /*0050*/ LOP3.LUT R1, R2, R3, R4 ;
 /*0060*/ CALL.REL.NOINC 0xb0 ;
 /*0070*/ @P0 EXIT ;
 /*0080*/ LOP3.LUT R1, R2, R3, R4 ;
 /*0090*/ STG [R1], R2 ;
 /*00a0*/ EXIT ;
 /*00b0*/ IMAD R1, R2, R3, R4 ;'''
        resource = 'Function frisk_execution: REG:255 STACK:256 SHARED:1024'
        result = segment_work(sass,resource,98304,256)
        self.assertEqual((result.carveout_bytes,result.loop_local_bytes,result.tail_address_ops),
                         (100*1024,2*4*256,1))
        self.assertTrue(result.single_cta)
        self.assertIsNone(segment_work(sass.replace('[R1+0x4]','[R77]'),resource,98304,256).loop_local_bytes)

    def test_tail_spill_slots_count_width_and_overlap(self):
        from h100_segment_cost import _local_words
        code=[(0,'LDL.64','R8, [R1+0x10]'),(16,'STL','[R1+0x14], R2'),
              (32,'LDL','R3, [R1+0x14]')]
        self.assertEqual(_local_words(code),({4,5},3))
        self.assertEqual(_local_words([(0,'LDL','R3, [R8]')]),(None,0))

    def test_tail_capacity_paid_once_outside_loop(self):
        from dataclasses import replace
        from h100_paired_mainloop import NativeLongMemoryLoopPair, LongLoopMemory
        from h100_segment_cost import WriterWork, attach_segments, attach_writer
        a=WriterWork(100*1024,255,True,65536,12,65536,32)
        b=replace(a,carveout_bytes=228*1024)
        mem=LongLoopMemory(0,0,10,10,((0,0),(0,0)))
        def bill(a,b,iters):
            base=NativeLongMemoryLoopPair('#frisk.distributed<x>',0,0,mem,mem,iters)
            return attach_writer(attach_segments(base,a,b,8),a,b,8).tail_cache_miss_sectors_delta
        self.assertEqual(bill(a,b,8),576)
        self.assertEqual(bill(a,b,608),576)
        self.assertEqual(bill(b,a,8),-576)
        self.assertEqual(bill(a,replace(b,tail_live_bytes=None),8),0)

    def test_segment_charges_incremental_carveout_and_tail_once(self):
        from dataclasses import replace
        from h100_paired_mainloop import NativeLongMemoryLoopPair, LongLoopMemory
        from h100_segment_cost import SegmentWork, attach_segments
        work = LongLoopMemory(0,0,64,25,((0,0),(0,0)))
        loop = NativeLongMemoryLoopPair('#frisk.distributed<x>',0,0,work,work,608)
        a = SegmentWork(100*1024,255,True,65536,800)
        b = replace(a,carveout_bytes=228*1024,tail_address_ops=900)
        f = attach_segments(loop,a,b,8)
        self.assertEqual((f.cache_miss_sectors_delta,f.tail_address_issues_delta),(87552,800))
        self.assertEqual(attach_segments(loop,b,a,8).cache_miss_sectors_delta,-87552)
        self.assertEqual(attach_segments(loop,a,a,8).cache_miss_sectors_delta,0)
        self.assertEqual(attach_segments(loop,replace(a,single_cta=False),b,8).cache_miss_sectors_delta,0)
        self.assertEqual(attach_segments(loop,a,replace(b,loop_local_bytes=0),8).cache_miss_sectors_delta,0)

    def test_long_distance_raw_and_one_local_charge(self):
        from h100_paired_mainloop import _long_memory_work, NativeLongMemoryLoopPair
        fillers=[('MOV','R8, RZ')]*95
        read=('IADD3','R4, R3, RZ, RZ')
        exposed=_long_memory_work([('LDG.E','R3, [R8]'),*fillers,read])
        covered=_long_memory_work([('LDG.E','R3, [R8]'),('MOV','R3, RZ'),*fillers,read])
        self.assertEqual((exposed.global_wait,exposed.longer_waits),(0,((32,0),(160,0))))
        self.assertEqual(covered.longer_waits,((0,0),(0,0)))
        def pair(a,b):
            return NativeLongMemoryLoopPair('#frisk.distributed<test>',0,0,a,b,8)
        self.assertEqual(pair(covered,exposed).warp_load_wait_delta,512)
        self.assertEqual(pair(exposed,covered).warp_load_wait_delta,-512)
        local=_long_memory_work([('LDL.64','R3, [R8]'),*fillers,read])
        self.assertEqual(pair(covered,local).warp_local_word_delta,16)
        self.assertEqual(pair(local,local).warp_load_wait_delta,0)

    def test_rounding_after_warp_scaling_is_symmetric(self):
        from h100_paired_mainloop import LongLoopMemory, NativeLongMemoryLoopPair
        a=LongLoopMemory(0,0,0,0,((0,0),(0,0)))
        b=LongLoopMemory(0,0,0,0,((0,0),(1,0)))
        def delta(x,y):
            return NativeLongMemoryLoopPair('#frisk.distributed<test>',0,0,x,y,8).warp_load_wait_delta
        self.assertEqual((delta(a,b),delta(b,a)),(3,-3))

    def test_load_first_use_and_overwrite(self):
        ld=('LDG.E','R3, [R8.64]')
        use=('IADD3','R4, R3, RZ, RZ')
        self.assertEqual(_memory_work([ld,use,use],8).global_wait,7)
        self.assertEqual(_memory_work([ld,('MOV','R3, RZ'),use],8).global_wait,0)
        # The high address word is also a consumer of the load.
        self.assertEqual(_memory_work([ld,('STS','[R2.64], R10')],8).global_wait,7)
        self.assertEqual(_memory_work([('LDG','R5, [R1]'),('STS.64','[R1], R4')],8).global_wait,7)
        wide=_memory_work([('LDL.64','R4, [R1]'),('STL.64','[R1], R5')],8)
        self.assertEqual((wide.local_wait,wide.local_load_words,wide.local_store_words),(7,2,2))

    def test_rectangular_online_fixture_uses_kv_trip_count(self):
        from native_online_fixture import online_fixture
        ir=online_fixture(256,64,513,kv_rows=129)
        self.assertIn('rows=513:i64,kv_rows=129:i64,',ir)
        self.assertIn('%bound = arith.constant 3 :',ir)
        self.assertEqual(online_fixture(256,64,513),online_fixture(256,64,513,kv_rows=513))
        self.assertNotIn('kv_rows=',online_fixture(256,64,513))

    def test_rectangular_online_fixture_rejects_bad_lengths(self):
        from native_online_fixture import online_fixture
        for n in (0,-1,True,3.5):
            with self.assertRaises(ValueError):
                online_fixture(256,64,513,kv_rows=n)

    def test_gate_uses_register_pressure_and_loop_trips(self):
        from native_store_selection import _memory_gate
        line='frisk.spill_pressure = array<i64: 128, 128, 0, 0>'
        args=(line,[(64,16),(128,8)])
        costs=(6000000,4000000)
        self.assertFalse(_memory_gate(*args,128,costs,10)['open'])
        self.assertTrue(_memory_gate(*args,4096,costs,10)['open'])
        small=line.replace('128','64')
        self.assertFalse(_memory_gate(small,args[1],4096,costs,10)['open'])
        self.assertTrue(_memory_gate(small,args[1],128,(91,100),10)['open'])


    def listing(self, extra=(), after=(), wait='gsb0, 0x0', groups=(1,)*8):
        instructions=['MOV R1, RZ', *extra]
        for form in ('ss','rs'):
            a='gdesc[UR4], R0' if form=='ss' else 'R4, gdesc[UR4], R0'
            for size in groups:
                instructions += ['HGMMA.64x64x16.F32 R0, '+a]*size
                instructions.append('WARPGROUP.DEPBAR.LE '+wait)
        instructions+=['@!P0 BRA 0x0',*after,'EXIT']
        return 'code for sm_90a\nFunction : frisk_execution\n'+'\n'.join(
            f' /*{i*16:04x}*/ {op} ;' for i,op in enumerate(instructions))

    def test_single_kv_backedge(self):
        base=self.listing()
        route=self.listing(['IMAD R0, R1, R2, R3','IMAD.SHL.U32 R0, R1, 2, RZ'],
                           after=['STS [R0], R1','BAR.SYNC 0x0'])
        pair=paired_native_mainloop(base,route,source='#frisk.distributed<test>',kv_tiles=16,warps=8)
        self.assertEqual((pair.warp_issues_delta,pair.warp_imads_delta),(256,128))
        self.assertIn('backend = "native"',pair.mlir_field())
        with self.assertRaises(ValueError):
            paired_native_mainloop(base,self.listing(wait='gsb0, 0x1'),source=pair.source,kv_tiles=16,warps=8)

    def test_four_issues_per_wait_in_each_stage(self):
        grouped=self.listing(groups=(4,4))
        kwargs=dict(source='#frisk.distributed<test>',kv_tiles=16,warps=8,mma_group_size=4)
        self.assertEqual(paired_native_mainloop(grouped,grouped,**kwargs).warp_issues_delta,0)
        for other in (self.listing(),self.listing(groups=(3,5)),self.listing(groups=(4,4),wait='gsb0, 0x1')):
            with self.assertRaises(ValueError):
                paired_native_mainloop(grouped,other,**kwargs)
        with self.assertRaises(ValueError):
            paired_native_mainloop(grouped,grouped,**dict(kwargs,mma_group_size=2))

    def test_asymmetric_qk_pv_and_instruction_delta(self):
        base=self.listing(groups=(4,4))
        # Keep all eight SS issues; the wider PV has one four-issue N128 group.
        lines=base.splitlines()
        before_pv=next(i for i,x in enumerate(lines) if 'HGMMA.' in x and ', R4, gdesc' in x)
        for i in range(before_pv,before_pv+4):
            lines[i]=lines[i].replace('64x64','64x128')
        del lines[before_pv+5:before_pv+10]
        base='\n'.join(lines)
        extra=base.replace('MOV R1, RZ ;','MOV R1, RZ ;\n /*0008*/ IMAD R0, R1, R2, R3 ;',1)
        args=dict(source='#frisk.distributed<x>',kv_tiles=3,warps=8,mma_group_size=4,stages=((64,8),(128,4)))
        pair=paired_native_mainloop(base,extra,**args)
        self.assertEqual((pair.warp_issues_delta,pair.warp_imads_delta),(24,24))
        with self.assertRaises(ValueError):
            paired_native_mainloop(base,base.replace('64x128','64x64'),**args)

    def test_device_opt_in_respects_explicit_allocation_limit(self):
        with patch('native_store_selection.compile_execution',side_effect=[SharedBudgetExceeded(), ('kernel',{})]) as compile_image, \
             patch('native_store_selection._device_shared_limit',return_value=232448):
            self.assertEqual(_compile_online_image({},4),('kernel',{}))
            self.assertEqual(compile_image.call_args.args,({'shared_limit_bytes':232448},))
        with patch('native_store_selection.compile_execution',side_effect=SharedBudgetExceeded()), \
             patch('native_store_selection._device_shared_limit') as device:
            with self.assertRaises(SharedBudgetExceeded):
                _compile_online_image({'shared_limit_bytes':65536},4)
            device.assert_not_called()

    def test_gate_boundaries_are_symmetric_and_inclusive(self):
        for a,b in ((100,90),(90,100),(13,13),(0,0)):
            self.assertTrue(_within_gap(a,b,10))
        self.assertFalse(_within_gap(89,100,10))
        self.assertFalse(_within_gap(100,1,0))

    def test_native_launch_requires_two_warpgroups(self):
        with self.assertRaisesRegex(ValueError,'256-thread'):
            emit({'operations':[{'name':'builtin.module','attrs':{'frisk.online_launch':{}}}]})

    def test_ignore_receipt_but_preserve_store_operand_and_encoding(self):
        def store(operand,extra,layout):
            return f'  frisk.tile_store %{operand}, %1 {{flag = {extra}}} : tensor<2xf32, {layout}>, memref<2xf32>'
        base=store(3,'true','#before')
        self.assertEqual(_semantic_ir(base),_semantic_ir(store(3,'dense<true> : tensor<8xi1>','#before')))
        self.assertNotEqual(_semantic_ir(base),_semantic_ir(store(2,'true','#before')))
        self.assertNotEqual(_semantic_ir(base),_semantic_ir(store(3,'true','#after')))


if __name__=='__main__':
    unittest.main()
