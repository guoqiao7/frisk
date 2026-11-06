import unittest
import native_surviving_audit as a


def op(name,operands,results):return dict(name=name,operands=operands,results=results)

class SurvivalTests(unittest.TestCase):
    def sample(self):
        return [op('frisk.tile_load',[],[1]),op('frisk.tile_load',[],[2]),op('frisk.tile_load',[],[3]),
                op('arith.mulf',[1,2],[4]),op('frisk.convert_layout',[3],[5]),op('arith.mulf',[4,4],[6]),op('frisk.tile_store',[6,99],[])]
    def test_output_backward_closure_removes_unused_input(self):
        rows=self.sample();keep,comm=a.surviving_ssa(rows,set(range(1,7)),{i:{(0,)*8} for i in range(1,7)},[(rows[4],{'mode':'Register'})])
        self.assertEqual(keep,{1,2,4,6});self.assertEqual(comm,[])
    def test_emitted_shared_keeps_dead_scaffolding(self):
        rows=self.sample();keep,comm=a.surviving_ssa(rows,set(range(1,7)),{i:{(0,)*8} for i in range(1,7)},[(rows[4],{'mode':'Shared'})])
        self.assertEqual(keep,set(range(1,7)));self.assertEqual(comm,[rows[4]])
    def test_unemitted_shared_not_a_root(self):
        rows=self.sample();keep,_=a.surviving_ssa(rows,set(range(1,7)),{i:{(0,)*8} for i in (1,2,3,4,6)},[(rows[4],{'mode':'Shared'})])
        self.assertEqual(keep,{1,2,4,6})
    def test_shuffle_is_root(self):
        rows=self.sample();keep,_=a.surviving_ssa(rows,set(range(1,7)),{i:{(0,)*8} for i in range(1,7)},[(rows[4],{'mode':'Shuffle'})])
        self.assertIn(3,keep);self.assertIn(5,keep)
    def test_shared_source_closure_recursive(self):
        rows=self.sample();rows[4]['operands']=[4];rows[-1]['operands']=[3,99]
        keep,_=a.surviving_ssa(rows,set(range(1,7)),{i:{(0,)*8} for i in range(1,7)},[(rows[4],{'mode':'Shared'})])
        self.assertEqual(keep,{1,2,3,4,5})
    def test_no_store_rejected(self):
        with self.assertRaisesRegex(ValueError,'one final store'):a.surviving_ssa([],set(),{},[])
    def test_unknown_root_rejected(self):
        with self.assertRaisesRegex(ValueError,'lacks'):a.surviving_ssa([op('frisk.tile_store',[1,99],[])],{1},{1:{(0,)*8}},[])
    def test_communication_fee_tamper_rejected(self):
        expected=dict.fromkeys(a.FEATURES,0);expected['shared_warp_issues']=4;actual=dict(expected);actual['shared_warp_issues']=0
        with self.assertRaisesRegex(ValueError,'shared_warp_issues'):a._check_features(actual,expected)
    def test_removed_order_feature_rejected(self):
        expected=dict.fromkeys(a.FEATURES,0)
        with self.assertRaisesRegex(ValueError,'unknown'):a._check_features(dict(expected,global_load_issue_groups=0),expected)

if __name__=='__main__':unittest.main()
