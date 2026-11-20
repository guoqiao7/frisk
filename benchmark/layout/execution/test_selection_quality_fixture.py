"""CPU-only leakage, semantic and preregistration checks for the quality stage."""
import copy
from pathlib import Path
import unittest
import numpy as np
import selection_quality_fixture as fixture

ROOT=Path(__file__).resolve().parents[3]

class SelectionQualityFixtureTests(unittest.TestCase):
    def test_previous_holdout_cannot_be_relabelled(self):
        old=fixture.historical_specifications()[0]
        old.update(key="not-really-new",split="holdout")
        with self.assertRaisesRegex(ValueError,"previously observed"):
            fixture.validate_specifications([old])

    def test_entire_program_split_is_disjoint(self):
        rows=fixture.specifications()
        old={fixture.program_identity(x) for x in fixture.historical_specifications()}
        holdout=[x for x in rows if x["split"]=="holdout"]
        anchors=[x for x in rows if x["split"]=="observed-anchor"]
        self.assertEqual(len(holdout),6);self.assertEqual(len(anchors),3)
        self.assertFalse({fixture.program_identity(x) for x in holdout}&old)
        self.assertTrue(all(fixture.program_identity(x) in old for x in anchors))
        self.assertTrue(all(x["case_id"]=="attention" for x in anchors))

    def test_duplicate_identity_cannot_inflate_coverage(self):
        rows=fixture.specifications()
        duplicate=copy.deepcopy(rows[0]);duplicate["key"]="other-key"
        with self.assertRaisesRegex(ValueError,"duplicate"):
            fixture.validate_specifications(rows+[duplicate])

    def test_controlled_and_natural_sources_preserve_actual_algorithm(self):
        import re
        for spec in fixture.specifications()[:6]:
            controlled=fixture.source(spec,ROOT,controlled=True)
            natural=fixture.source(spec,ROOT)
            self.assertIn("joint.A=#A,joint.B=#B",controlled)
            self.assertNotIn("joint.A",natural);self.assertNotIn("joint.B",natural)
            erased=controlled.replace(",joint.A=#A,joint.B=#B","")
            erased="\n".join(x for x in erased.splitlines() if not re.match(r"#(?:A|B)=",x))+"\n"
            self.assertEqual(natural,erased)
            self.assertIn("tensor<"+"x".join(map(str,spec["shape"]))+"xf32>",natural)
            self.assertIn("frisk.tile_store",natural)

    def test_attention_cannot_accidentally_receive_controlled_injection(self):
        for spec in fixture.specifications()[6:]:
            with self.assertRaisesRegex(ValueError,"natural domains only"):
                fixture.source(spec,ROOT,controlled=True)
            text=fixture.source(spec,ROOT)
            self.assertNotIn("joint.A",text)
            self.assertIn("frisk.mma",text)

    def test_all_independent_inputs_have_finite_references(self):
        for spec in fixture.specifications()[:6]:
            cases=fixture.numerical_cases(spec)
            self.assertEqual(len(cases),7)
            for name,q,k,v in cases:
                self.assertTrue(np.all(k!=0),(spec["key"],name))
                expected=fixture.reference(spec,q,k,v)
                self.assertEqual(expected.shape,q.shape)
                self.assertTrue(np.isfinite(expected).all(),(spec["key"],name))
                self.assertTrue(fixture.check_numerics(spec,expected,expected)["passed"])

    def test_ten_timing_batches_have_finite_independent_references(self):
        from calibration_fixture import inputs as previous_inputs
        for spec in fixture.specifications()[:6]:
            digests=set()
            for batch in range(10):
                actual=fixture.inputs(spec,batch)
                previous=previous_inputs(spec,batch)
                self.assertTrue(all(np.array_equal(a,b) for a,b in zip(actual,previous)))
                q,k,v=actual
                self.assertTrue(np.all(k!=0))
                expected=fixture.reference(spec,q,k,v)
                self.assertTrue(np.isfinite(expected).all(),(spec["key"],batch))
                self.assertTrue(fixture.check_numerics(spec,expected,expected)["passed"])
                digests.add(fixture.hashlib.sha256(q.tobytes()+k.tobytes()+v.tobytes()).hexdigest())
            self.assertEqual(len(digests),10)

    def test_promotion_excludes_anchors_from_primary_and_controls_guard_family(self):
        protocol=fixture.protocol()
        gate=protocol["acceptance"]["calibrated_promotion"]
        self.assertAlmostEqual(gate["geomean_calibrated_over_legacy_ci95_upper_lt"],1/1.03)
        self.assertEqual(gate["guard_comparison_count"],9)
        self.assertEqual(gate["guard_familywise_alpha"],.05)
        self.assertIn("six fresh",gate["primary"])
        self.assertIn("Attention anchors excluded",gate["primary"])
        self.assertIn("union of four chosen",protocol["experiment"]["natural"])
        self.assertIn("only in controlled",protocol["experiment"]["oracle_scope"])

    def test_exp_tolerance_is_frozen_and_no_relaxation_for_ordinary(self):
        rows=fixture.specifications();one=np.array([1.0],np.float32)
        exp=next(x for x in rows if x["tail"]=="exp1")
        self.assertTrue(fixture.check_numerics(exp,one+np.float32(1e-6),one)["passed"])
        self.assertFalse(fixture.check_numerics(exp,one+np.float32(1e-3),one)["passed"])
        ordinary=rows[0]
        self.assertFalse(fixture.check_numerics(ordinary,one+np.float32(1e-6),one)["passed"])
        self.assertFalse(fixture.check_numerics(ordinary,np.array([0.0],np.float32),np.array([-0.0],np.float32))["passed"])

    def test_resource_and_model_statuses_do_not_silently_promote(self):
        protocol=fixture.protocol()
        self.assertFalse(protocol["profile"]["fit"])
        self.assertIn("otherwise retain legacy",protocol["acceptance"]["calibrated_promotion"]["requirements"])
        self.assertIn("not-qualified",protocol["resources"]["physical_registers"])
        self.assertEqual(protocol["experiment"]["conversion"],"C")
        self.assertIn("LI-O",protocol["experiment"]["excluded"])
        self.assertIn("no mandatory speedup",protocol["acceptance"]["joint_superiority"])
        self.assertEqual(fixture.digest(protocol),fixture.digest(fixture.protocol()))

    def test_safety_identity_changes_with_complete_spec_and_abi(self):
        spec=fixture.specifications()[0]
        original=fixture.safety_input_identity(spec,{"copy_path":"vector"},["q","k","v","o"])
        different=fixture.safety_input_identity(fixture.specifications()[1],{"copy_path":"vector"},["q","k","v","o"])
        self.assertNotEqual(fixture.digest(original),fixture.digest(different))
        other=fixture.safety_input_identity(spec,{"copy_path":"vector"},["q","k","v","other"])
        self.assertNotEqual(fixture.digest(original),fixture.digest(other))

if __name__=="__main__":unittest.main()
