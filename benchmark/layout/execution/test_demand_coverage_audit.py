import copy
import unittest
import demand_coverage_audit as audit

ZERO=(0,0,0,0,0,0,0,0)
ONE=(1,0,0,0,0,0,0,0)

class FootprintEmitter:
    def __init__(self,alignment=32,replicated=False):
        logical=[t%32 if replicated else t for t in range(128)]
        self.v={10:{'id':10,'distributed':{'topology':[1,32,4,1,1],'logical':logical,
                 'first':list(range(32 if replicated else 128))}}}
        self.data={'operations':[{'name':'frisk.tile_load','operands':[0],'results':[10]},
                                 {'name':'frisk.tile_store','operands':[10,1],'results':[]}]}
        self.align=alignment
    def storage(self,value):return {'root':value,'alignment':self.align}
    def offsets(self,value):return {slot:4*slot for slot in range(128)}

class DemandCoverageAuditTest(unittest.TestCase):
    def test_full_thread_domain_affine_signature(self):
        query=[7^(3 if t&1 else 0)^(16 if t&8 else 0) for t in range(128)]
        self.assertEqual(audit.canonical_query(query),(7,3,0,0,16,0,0,0))
    def test_non_affine_query_rejected_by_full_domain(self):
        query=list(range(128));query[3]^=1
        with self.assertRaises(ValueError):audit.canonical_query(query)
    def test_invalid_query_indices_and_domain_rejected(self):
        for query in [[0]*127,[True]*128,[-1]*128]:
            with self.assertRaises(ValueError):audit.canonical_query(query)
    def test_alias_queries_union_not_sum(self):
        values=[{'id':9,'distributed':{},'layout_variable_ids':[1,2]}]
        rows,mapped=audit.compare_queries(values,{1:{ZERO},2:{ZERO,ONE}},{9:{ZERO,ONE}})
        self.assertEqual(len(mapped[9]),2);self.assertEqual(rows[0]['expected_queries'],2)
    def test_same_count_wrong_query_fails(self):
        values=[{'id':9,'distributed':{},'layout_variable_ids':[1]}]
        with self.assertRaisesRegex(ValueError,'query mismatch'):audit.compare_queries(values,{1:{ZERO}},{9:{ONE}})
    def test_unmapped_actual_query_fails(self):
        for ids in [None,[],[True]]:
            values=[{'id':9,'distributed':{},'layout_variable_ids':ids}]
            with self.assertRaises(ValueError):audit.compare_queries(values,{}, {9:{ZERO}})
    def test_constraint_keyed_boundary_can_alias_same_actual_value(self):
        values=[{'id':9,'distributed':{},'layout_variable_ids':[1],'layout_boundary_ids':[14]}]
        rows,mapped=audit.compare_queries(values,{1:{ZERO}},{9:{ZERO,ONE}},{14:{ZERO,ONE}})
        self.assertEqual(len(mapped[9]),2);self.assertEqual(rows[0]['boundaries'],[14])
    def test_two_operand_boundaries_are_independent_despite_same_target_variable(self):
        values=[{'id':9,'distributed':{},'layout_variable_ids':[],'layout_boundary_ids':[14]},
                {'id':10,'distributed':{},'layout_variable_ids':[],'layout_boundary_ids':[15]}]
        _,mapped=audit.compare_queries(values,{}, {9:{ZERO},10:{ONE}},{14:{ZERO},15:{ONE}})
        self.assertEqual(mapped,{9:{ZERO},10:{ONE}})
    def test_boundary_only_export_may_omit_definition_ids(self):
        values=[{'id':9,'distributed':{},'layout_boundary_ids':[14]}]
        rows,mapped=audit.compare_queries(values,{}, {9:{ZERO}},{14:{ZERO}})
        self.assertEqual(rows[0]['variables'],[]);self.assertEqual(mapped,{9:{ZERO}})
    def test_nonempty_orphan_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError,'boundary has no exported SSA'):audit.compare_queries([],{}, {},{14:{ZERO}})
    def test_nonempty_orphan_cpp_variable_fails(self):
        with self.assertRaisesRegex(ValueError,'no exported SSA'):audit.compare_queries([], {1:{ZERO}}, {})
    def test_graph_variable_cannot_be_mapped_to_distinct_ssa(self):
        values=[{'id':i,'distributed':{},'layout_variable_ids':[1]} for i in [9,10]]
        with self.assertRaisesRegex(ValueError,'multiple actual SSA'):audit.compare_queries(values,{1:{ZERO}},{9:{ZERO},10:{ZERO}})
    def test_duplicate_ledger_variables_or_queries_rejected(self):
        row={'variable':1,'queries':[{'terms':list(ZERO)}]}
        for rows in [[row,row],[dict(row,queries=row['queries']*2)]]:
            with self.assertRaises(ValueError):audit.parse_values({'values':rows})
    def test_query_coefficients_are_strict_unsigned_integers(self):
        for terms in [[0]*7,[0]*7+[True],[0]*7+[-1]]:
            with self.assertRaises(ValueError):audit.parse_values({'values':[{'variable':1,'queries':[{'terms':terms}]}]})
    @staticmethod
    def byte_request():
        return ['ld.global.u8 %r, [%p+0];','shl.b32 %r, %r, 8;','or.b32 %v, %v, %r;']*4
    def test_actual_load_run_preserves_address_assembly_and_literal_moves(self):
        request=self.byte_request()
        code=request+['bfe.u32 %r, tid, 0, 1;','mul.lo.u32 %r, %r, 4;','mov.b32 %f, 0;','cvt.u64.u32 %a, %r;']+request
        result=audit.global_load_issue_groups(code)
        self.assertEqual((result['global_load_issue_groups'],result['f32_load_requests'],result['global_byte_loads']),(1,2,8))
    def test_actual_compute_communication_sync_and_store_cut_load_run(self):
        for opcode in ['mul.rn.f32','sub.rn.f32','div.rn.f32','ex2.approx.f32','shfl.sync.idx.b32','ld.shared.b32','@%p st.shared.b32','bar.sync','@%p st.global.u8']:
            code=self.byte_request()+[opcode+' %r, %v;']+self.byte_request()
            self.assertEqual(audit.global_load_issue_groups(code)['global_load_issue_groups'],2,opcode)
    def test_empty_compute_and_register_alias_do_not_invent_load_groups(self):
        self.assertEqual(audit.global_load_issue_groups(['mul.rn.f32 %f, %a, %b;'])['global_load_issue_groups'],0)
        self.assertEqual(audit.global_load_issue_groups(self.byte_request()+['mov.b32 %x, %y;']+self.byte_request())['global_load_issue_groups'],1)
    def test_partial_or_wrong_width_global_request_fails_closed(self):
        for code in [['ld.global.u8 %r, [%p];'],['ld.global.u8 %r, [%p];','bar.sync 0;'],['ld.global.u32 %r, [%p];']]:
            with self.assertRaises(ValueError):audit.global_load_issue_groups(code)
    def test_new_eighteenth_feature_is_strict_but_legacy_seventeen_remain_compatible(self):
        expected=dict.fromkeys(audit.FEATURE_NAMES,0);expected['global_load_issue_groups']=3
        priced=dict(expected);audit.check_priced_features(priced,expected)
        priced['global_load_issue_groups']=2
        with self.assertRaisesRegex(ValueError,'global_load_issue_groups'):audit.check_priced_features(priced,expected)
        del priced['global_load_issue_groups'];audit.check_priced_features(priced,expected,legacy_17=True)
        with self.assertRaisesRegex(ValueError,'global_load_issue_groups'):audit.check_priced_features(priced,expected)
    def test_constant_moves_are_charged_per_observed_query(self):
        class Emitter:pass
        emitter=Emitter();emitter.data={'operations':[{'name':'arith.constant','results':[9]}]}
        totals=dict(warp_issues=0,global_sectors32=0,shared_bank_excess=0,address_alu_warp_issues=0)
        features,work=audit.expected_priced_features(emitter,{9:{ZERO,ONE}},totals,[])
        self.assertEqual(features['alu_warp_issues'],8);self.assertEqual(work,8)
        corrupted=dict(features);corrupted['alu_warp_issues']=0
        with self.assertRaisesRegex(ValueError,'alu_warp_issues'):audit.check_priced_features(corrupted,features)
    def test_affine_map_grammar_counts_all_nonzero_thread_columns(self):
        self.assertEqual(audit.map_work([0]*128),1)
        self.assertEqual(audit.map_work([t*4 for t in range(128)]),29)
        malformed=list(range(128));malformed[3]^=1
        with self.assertRaises(ValueError):audit.map_work(malformed)
    def test_variable_receiver_mux_is_independent_and_mandatory(self):
        class Emitter:
            def tensor_slots(self,value):return [[((t^32) if value==2 else t)*2+r for t in range(128)] for r in range(2)]
        plan=dict(mode='Shared',source_registers=2,destination_registers=2,
                  global_writers={slot:{'warp':(slot//2)//32,'lane':(slot//2)%32,'reg':slot%2} for slot in range(256)})
        q=(0,1,0,0,0,0,0,0)
        value=audit.conversion_structure(Emitter(),1,2,plan,{q},True)
        self.assertEqual(value['receiver_mux_alu_warp_issues'],32)
        self.assertEqual(value['address_alu_warp_issues'],1000)
        self.assertEqual((value['bytes'],value['barriers']),(2048,2))
        audit.check_boundary(value,value,7)
        for field in ('receiver_mux_alu_warp_issues','address_alu_warp_issues','bytes','barriers'):
            corrupted=copy.deepcopy(value);corrupted[field]=0
            with self.assertRaisesRegex(ValueError,field):audit.check_boundary(corrupted,value,7)
        missing=dict(value);del missing['receiver_mux_alu_warp_issues']
        with self.assertRaisesRegex(ValueError,'receiver_mux'):audit.check_boundary(missing,value,7)
    def test_all_eighteen_priced_features_include_explicit_inactive_zero(self):
        expected=dict.fromkeys(audit.FEATURE_NAMES,0);expected['alu_warp_issues']=17
        audit.check_priced_features({'alu_warp_issues':17},expected)
        for field in audit.FEATURE_NAMES:
            corrupted=dict(expected);corrupted[field]+=1
            with self.assertRaisesRegex(ValueError,field):audit.check_priced_features(corrupted,expected)
    def test_dead_declared_shared_still_reserves_scratch(self):
        record={'cost':{'cost':[0,0,0,0,0,4096,0,0,0]}}
        audit.check_scratch({'scratch_max_bytes':4096},{'scratch_max_bytes':4096},4096,record,True)
        with self.assertRaisesRegex(ValueError,'dead conversions'):audit.check_scratch({'scratch_max_bytes':0},{'scratch_max_bytes':4096},4096,record,True)
        record['cost']['cost'][5]=0
        with self.assertRaisesRegex(ValueError,'priced Shared scratch'):audit.check_scratch({'scratch_max_bytes':4096},{'scratch_max_bytes':4096},4096,record,True)
    def test_aligned_contiguous_access_footprint_closed_form(self):
        rows=audit.access_coverage(FootprintEmitter(),{10:{ZERO}},{'root_order':[0,1]})
        self.assertEqual([r['warp_issues'] for r in rows],[4,4])
        self.assertEqual([r['global_sectors32'] for r in rows],[16,16])
    def test_unknown_alignment_uses_worst_root_residue(self):
        rows=audit.access_coverage(FootprintEmitter(1),{10:{ZERO}},{'root_order':[0,1]})
        self.assertEqual([r['global_sectors32'] for r in rows],[20,20])
    def test_replicated_store_elects_only_canonical_owner_warp(self):
        rows=audit.access_coverage(FootprintEmitter(replicated=True),{10:{ZERO}},{'root_order':[0,1]})
        self.assertEqual([r['warp_issues'] for r in rows],[4,1])
        self.assertEqual([r['global_sectors32'] for r in rows],[16,4])
if __name__=='__main__':unittest.main()
