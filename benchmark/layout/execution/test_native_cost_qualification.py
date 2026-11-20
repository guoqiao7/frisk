import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import native_cost_qualification as q
import native_cost_fixture as f

MODELS=[dict(name=n,id=n) for n in f.MODEL_NAMES]
ARMS=[a['arm'] for a in f.arms(MODELS)]


def feedback():
    return {name:dict(status='supported',cost=10,uses_gpu_timing=False) for name in ARMS}


class SelectionTests(unittest.TestCase):
    def test_native_minimum(self):
        x=feedback();x['v5d-LI']['cost']=1;r=q.choose_native(f.specifications()[0],x,ARMS)
        self.assertEqual(r['selected_source'],'v5d-LI');self.assertEqual(len(set(r['selections'].values())),1);self.assertFalse(r['uses_gpu_timing'])
    def test_decision_tamper_rejected(self):
        x=feedback();d=q.choose_native(f.specifications()[0],x,ARMS);d['selections']['native-LI-opportunity']='legacy-LI'
        with self.assertRaisesRegex(ValueError,'frozen feedback'):q.verify_decision(f.specifications()[0],x,ARMS,d)
    def test_tie_deterministic(self):
        self.assertEqual(q.choose_native(f.specifications()[0],feedback(),list(reversed(ARMS)))['selected_source'],'surviving-J-new')
    def test_one_unsupported_fallbacks_whole_pool(self):
        x=feedback();x['legacy-LI']=dict(status='unsupported',reason='unknown-opcode',cost=None)
        r=q.choose_native(f.specifications()[0],x,ARMS);self.assertEqual(r['status'],'common-fallback');self.assertEqual(r['selected_source'],'surviving-J-new');self.assertEqual(len(r['candidates']),6)
    def test_one_unavailable_fallbacks_whole_pool(self):
        x=feedback();x['v5d-LI']=dict(status='unavailable',cost=None)
        self.assertEqual(q.choose_native(f.specifications()[0],x,ARMS)['status'],'common-fallback')
    def test_div_forced_fallback(self):
        self.assertEqual(q.choose_native(f.specifications()[4],feedback(),ARMS)['status'],'common-fallback')
    def test_mixed_forced_fallback(self):
        self.assertEqual(q.choose_native(f.specifications()[3],feedback(),ARMS)['status'],'common-fallback')
    def test_anchor_forced_fallback(self):
        self.assertEqual(q.choose_native(f.specifications()[6],feedback(),ARMS)['status'],'common-fallback')
    def test_missing_candidate_rejected(self):
        x=feedback();x.pop(ARMS[0])
        with self.assertRaisesRegex(ValueError,'complete'):q.choose_native(f.specifications()[0],x,ARMS)
    def test_timing_leak_rejected(self):
        x=feedback();x[ARMS[0]]['uses_gpu_timing']=True
        with self.assertRaisesRegex(ValueError,'no-timing'):q.choose_native(f.specifications()[0],x,ARMS)
    def test_noninteger_score_rejected(self):
        x=feedback();x[ARMS[0]]['cost']=1.5
        with self.assertRaisesRegex(ValueError,'integer'):q.choose_native(f.specifications()[0],x,ARMS)


class PortfolioTests(unittest.TestCase):
    def rows(self):return [dict(arm=a['arm'],path='/'+a['arm'],treatment=a) for a in f.arms(MODELS)]
    def test_six_solves_total_not_single(self):
        record=dict(total=dict(probes=65536,refinement=256),search_components=[dict(seed_probes=1,seed_scores=2)])
        with patch.object(q,'read',return_value=record),patch.object(q.demand,'check_search') as check:r=q.portfolio_budget(self.rows(),MODELS)
        self.assertEqual(r['total_probes'],393216);self.assertEqual(r['total_refinement'],1536);self.assertEqual(check.call_count,6)
    def test_missing_source_rejected(self):
        with self.assertRaisesRegex(ValueError,'six-source'):q.portfolio_budget(self.rows()[:-1],MODELS)
    def test_duplicate_source_rejected(self):
        rows=self.rows();rows[-1]=rows[0]
        with self.assertRaisesRegex(ValueError,'six-source'):q.portfolio_budget(rows,MODELS)
    def test_totals_rejected_even_with_mocked_per_source(self):
        record=dict(total=dict(probes=65537,refinement=256),search_components=[])
        with patch.object(q,'read',return_value=record),patch.object(q.demand,'check_search'):
            with self.assertRaisesRegex(ValueError,'total budget'):q.portfolio_budget(self.rows(),MODELS)
    def test_audit_cannot_write_old_stage(self):
        with self.assertRaisesRegex(ValueError,'new stage'):q.audit_preserved(Path('/home/baopeihua/frisk/guoqiao/optimize_plan/demand_cost_search_repair_artifacts/new.json'))

class OverheadTests(unittest.TestCase):
    def test_sums_all_sources_and_deduplicates_preparation(self):
        spec=f.specifications()[0];key='natural-'+spec['key'];rows=[]
        for a in f.arms(MODELS):
            phases=[dict(stage='preparation',command_metadata='/prep-'+a['model'],command_metadata_sha256='ok'),
                    dict(stage='native-selection',command_metadata='/solve-'+a['arm'],command_metadata_sha256='ok')]
            rows.append(dict(spec=spec,arm=a['arm'],path='/'+a['arm'],phases=phases,native_analysis_wall_ns=2))
        groups={key:dict(selections={'legacy-J-new':dict(median_us=2.),'native-pool':dict(median_us=1.)})}
        def load(path):
            if Path(path).name=='compile.json':return dict(wall_ns=30)
            return dict(exit_code=0,elapsed_ns=10 if str(path).startswith('/prep-') else 20)
        with patch.object(q,'sha',return_value='ok'),patch.object(q,'read',side_effect=load):r=q.feedback_overhead(rows,groups)[key]
        self.assertEqual(r['portfolio_setup_wall_ns'],342);self.assertEqual(r['observed_legacy_components_wall_ns'],60);self.assertEqual(r['additional_observed_component_work_ns'],282)
        self.assertEqual(r['native_compile_requests'],6);self.assertAlmostEqual(r['estimated_break_even_launches'],.282)

if __name__=='__main__':unittest.main()
