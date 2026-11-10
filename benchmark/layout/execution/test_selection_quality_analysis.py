import copy
import hashlib
import json
import unittest
import numpy as np
import selection_quality_analysis as analysis


def candidate(arm,identity,legacy,calibrated=None):
    return dict(arm=arm,execution_identity=identity,cubin_sha256="cubin-"+identity,
                model_scores=dict(legacy=legacy,calibrated=legacy if calibrated is None else calibrated))


def records(candidates,durations):
    result=[]
    for b in range(10):
        content=dict(batch=b,key="fixed-abi",input_hashes=["q"+str(b),"k","v"],reference_sha256="ref"+str(b))
        pair=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        for c in candidates:
            value=durations[c["arm"]]
            us=value[b] if isinstance(value,list) else value
            result.append(dict(batch=b,arm=c["arm"],valid=True,pair_id=pair,pair_content=content,
                               launch_count=100,actual_graph_nodes=100,
                               samples_ms=[us/1000]*5,graph_elapsed_ms=[us/10]*5,cubin_sha256=c["cubin_sha256"]))
    return result


class SelectionQuality(unittest.TestCase):
    def fixture(self):
        cs=[candidate("A","image-A",[1,20],[5,20]),candidate("B","image-B",[2,0],[1,0])]
        return cs,records(cs,dict(A=2.,B=4.)),dict(legacy="A",calibrated="B")

    def test_known_inversion_and_regret(self):
        cs,rows,sel=self.fixture();r=analysis.analyze_group(rows,cs,sel)
        self.assertEqual(r["models"]["legacy"]["correct_pairs"],1)
        self.assertEqual(r["models"]["calibrated"]["inversions"],1)
        self.assertEqual(r["selections"]["calibrated"]["observed_regret_ratio"],2.)
        self.assertEqual(r["selections"]["calibrated"]["regret_ci95"],[2.,2.])
        self.assertTrue(r["selections"]["legacy"]["model_optimum_membership"]["legacy"])

    def test_duplicate_images_do_not_increase_blocks(self):
        cs,rows,sel=self.fixture();base=analysis.analyze_group(rows,cs,sel)
        cs.append(candidate("alias-A","image-A",[0,0],[0,0]))
        for row in list(rows):
            if row["arm"]=="A":row=copy.deepcopy(row);row["arm"]="alias-A";rows.append(row)
        r=analysis.analyze_group(rows,cs,sel)
        self.assertEqual(r["paired_batches"],10)
        self.assertEqual(r["unique_execution_identities"],2)
        self.assertEqual(r["candidate_assignments"],3)
        self.assertEqual(r["selections"]["calibrated"]["regret_ci95"],base["selections"]["calibrated"]["regret_ci95"])
        self.assertEqual(r["models"]["calibrated"]["assignment_score_disagreements_within_image"],["image-A"])
        self.assertEqual(r["models"]["calibrated"]["correct_pairs"],1)

    def test_lexicographic_scores_are_not_scalar_sums(self):
        cs=[candidate("A","a",[1,999]),candidate("B","b",[2,0])]
        r=analysis.analyze_group(records(cs,dict(A=2.,B=4.)),cs,dict(chosen="A"))
        self.assertEqual(r["models"]["legacy"]["best_execution_identities"],["a"])

    def test_same_image_selection_is_exact_tie(self):
        cs=[candidate("A","a",[1]),candidate("alias","a",[5])]
        r=analysis.analyze_group(records(cs,dict(A=2.,alias=2.01)),cs,dict(first="A",second="alias"))
        self.assertEqual(r["image_comparisons"],[])
        self.assertEqual(r["policy_comparisons"]["first/second"]["ratio_ci95"],[1.,1.])
        self.assertEqual(r["selections"]["second"]["observed_regret_ratio"],1.)

    def test_practical_tie_not_ranking_success(self):
        cs,_,sel=self.fixture();r=analysis.analyze_group(records(cs,dict(A=2.,B=2.02)),cs,sel)
        self.assertEqual(r["models"]["legacy"]["distinguishable_pairs"],0)
        self.assertEqual(r["models"]["legacy"]["correct_pairs"],0)

    def test_model_tie_on_real_difference_is_explicit(self):
        cs,rows,sel=self.fixture();cs[1]["model_scores"]=copy.deepcopy(cs[0]["model_scores"])
        r=analysis.analyze_group(rows,cs,sel)
        self.assertEqual(r["models"]["legacy"]["model_ties_on_distinguishable_pairs"],1)

    def test_inverse_orientation_recomputes_even_median(self):
        cs,_,sel=self.fixture()
        a=[1.]*10;b=[1.,1.,1.,1.,2.,4.,9.,9.,9.,9.]
        r=analysis.analyze_group(records(cs,dict(A=a,B=b)),cs,sel)
        inverse=analysis.policy_ratio(r,"legacy","calibrated")
        self.assertAlmostEqual(inverse["ratio"],np.median(np.asarray(a)/b))
        self.assertNotAlmostEqual(inverse["ratio"],1/analysis.policy_ratio(r,"calibrated","legacy")["ratio"])

    def test_bootstrap_deterministic_and_order_independent(self):
        cs,rows,sel=self.fixture();r1=analysis.analyze_group(rows,cs,sel)
        r2=analysis.analyze_group(list(reversed(rows)),list(reversed(cs)),sel)
        self.assertEqual(r1,r2)

    def test_missing_or_invalid_batch_fails_closed(self):
        cs,rows,sel=self.fixture()
        with self.assertRaises(ValueError):analysis.analyze_group(rows[:-1],cs,sel)
        rows[0]["valid"]=False
        with self.assertRaises(ValueError):analysis.analyze_group(rows,cs,sel)

    def test_unmatched_inputs_fail_closed(self):
        cs,rows,sel=self.fixture();row=rows[0];row["pair_content"]=dict(row["pair_content"],reference_sha256="other")
        row["pair_id"]=hashlib.sha256(json.dumps(row["pair_content"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
        with self.assertRaises(ValueError):analysis.analyze_group(rows,cs,sel)

    def test_raw_identity_tampering_fails(self):
        cs,rows,sel=self.fixture()
        for key,value in (("pair_id","bad"),("cubin_sha256","other"),("actual_graph_nodes",99),("batch",True)):
            changed=copy.deepcopy(rows);changed[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):analysis.analyze_group(changed,cs,sel)

    def test_duration_equation_and_nonfinite_fail(self):
        cs,rows,sel=self.fixture()
        for key,value in (("samples_ms",[.001]*5),("graph_elapsed_ms",[float("nan")]*5),("samples_ms",[True]*5)):
            changed=copy.deepcopy(rows);changed[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):analysis.analyze_group(changed,cs,sel)

    def test_bad_candidate_domains_fail(self):
        cs,rows,sel=self.fixture()
        variants=[cs+[cs[0]], [dict(cs[0],execution_identity=""),cs[1]],
                  [dict(cs[0],model_scores=dict(legacy=[True],calibrated=[0])),cs[1]],
                  [dict(cs[0],model_scores=dict(legacy=[1],calibrated=[0])),cs[1]],
                  [cs[0],dict(cs[1],execution_identity=cs[0]["execution_identity"])]]
        for changed in variants:
            with self.subTest(changed=changed),self.assertRaises(ValueError):analysis.analyze_group(rows,changed,sel)
        with self.assertRaises(ValueError):analysis.analyze_group(rows,cs,dict(unknown="Z"))

    def test_natural_union_can_omit_scores_explicitly(self):
        cs,rows,sel=self.fixture()
        for c in cs:c.pop("model_scores")
        r=analysis.analyze_group(rows,cs,sel)
        self.assertEqual(r["models"],{})
        self.assertEqual(r["unavailable_models"],{})
        self.assertEqual(r["selections"]["calibrated"]["observed_regret_ratio"],2.)

    def test_partial_scores_never_form_a_subset_ranking(self):
        cs,rows,sel=self.fixture();cs[0]["model_scores"].pop("legacy")
        r=analysis.analyze_group(rows,cs,sel)
        self.assertNotIn("legacy",r["models"])
        self.assertEqual(r["unavailable_models"]["legacy"]["missing_arms"],["A"])
        self.assertIn("calibrated",r["models"])

    def test_fixed_suite_known_geomean_and_max_regret(self):
        cs,rows,sel=self.fixture();first=analysis.analyze_group(rows,cs,sel)
        second=analysis.analyze_group(records(cs,dict(A=3.,B=9.)),cs,sel)
        suite=analysis.summarize_suite(dict(first=first,second=second))
        self.assertAlmostEqual(suite["policies"]["calibrated"]["geomean_regret_ratio"],np.sqrt(6.))
        self.assertEqual(suite["policies"]["calibrated"]["max_case_regret_ratio"],3.)
        np.testing.assert_allclose(suite["policies"]["calibrated"]["geomean_regret_ci95"],[np.sqrt(6.)]*2)
        self.assertEqual(suite["policies"]["legacy"]["max_case_regret_ci95"],[1.,1.])
        self.assertAlmostEqual(suite["policy_comparisons"]["calibrated/legacy"]["ratio"],np.sqrt(6.))

    def test_fixed_suite_recomputes_best_image_inside_each_resample(self):
        cs,_,sel=self.fixture()
        # B changes from faster to slower than A across blocks; all blocks remain
        # paired. Regret uncertainty must include selection of the reference.
        group=analysis.analyze_group(records(cs,dict(A=[2.]*10,B=[1.]*5+[4.]*5)),cs,sel)
        suite=analysis.summarize_suite(dict(program=group))
        ci=suite["policies"]["legacy"]["max_case_regret_ci95"]
        self.assertEqual(ci,[1.,2.])
        self.assertGreater(ci[1],suite["policies"]["legacy"]["max_case_regret_ratio"])

    def test_fixed_suite_is_order_independent(self):
        cs,rows,sel=self.fixture();g=analysis.analyze_group(rows,cs,sel)
        one=analysis.summarize_suite(dict(a=g,b=g),["legacy","calibrated"])
        two=analysis.summarize_suite(dict(b=g,a=g),["legacy","calibrated"])
        self.assertEqual(one,two)

    def test_fixed_suite_rejects_missing_policies_or_bad_observations(self):
        cs,rows,sel=self.fixture();g=analysis.analyze_group(rows,cs,sel)
        with self.assertRaises(ValueError):analysis.summarize_suite({})
        with self.assertRaises(ValueError):analysis.summarize_suite(dict(case=g),["unmeasured"])
        bad=copy.deepcopy(g);bad["image_batch_medians_us"]["image-A"].pop()
        with self.assertRaises((ValueError,TypeError)):analysis.summarize_suite(dict(case=bad))
        bad=copy.deepcopy(g);bad["status"]="incomplete"
        with self.assertRaises(ValueError):analysis.summarize_suite(dict(case=bad))

    def test_declared_familywise_guard_interval(self):
        cs,_,sel=self.fixture()
        g=analysis.analyze_group(records(cs,dict(A=[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.],B=5.)),cs,sel)
        nominal=analysis.paired_policy_interval(g,"legacy","calibrated")
        guarded=analysis.paired_policy_interval(g,"legacy","calibrated",alpha=.05/9)
        self.assertLessEqual(guarded["ratio_ci"][0],nominal["ratio_ci"][0])
        self.assertGreaterEqual(guarded["ratio_ci"][1],nominal["ratio_ci"][1])
        self.assertEqual(guarded["confidence"],1-.05/9)
        self.assertEqual(guarded["paired_batches"],10)
        with self.assertRaises(ValueError):analysis.paired_policy_interval(g,"legacy","calibrated",alpha=0)

    def test_missing_input_binding_fails(self):
        cs,rows,sel=self.fixture();rows[0]["pair_content"]={}
        with self.assertRaises(ValueError):analysis.analyze_group(rows,cs,sel)


if __name__=="__main__":unittest.main()
