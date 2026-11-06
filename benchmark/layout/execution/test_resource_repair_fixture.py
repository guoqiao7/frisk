import unittest
from resource_repair_fixture import specifications,program_identity,arms,protocol,fixture,inputs,reference,canonical_digest
from calibration_fixture import specifications as previous
import numpy as np


class ResourceRepairFixtureTest(unittest.TestCase):
    def test_all_prior_holdouts_are_known_and_disjoint_from_new_programs(self):
        specs=specifications();known=[x for x in specs if x["split"]=="known"];new=[x for x in specs if x["split"]=="prospective"]
        self.assertEqual(len(known),12);self.assertEqual(len(new),6)
        self.assertEqual({program_identity(x) for x in known},{program_identity(x) for x in previous()})
        self.assertFalse({program_identity(x) for x in known}&{program_identity(x) for x in new})
        self.assertEqual(sum(x["previous_split"]=="holdout" for x in known),3)
    def test_all_variants_stay_in_their_whole_program_split(self):
        specs=specifications()
        self.assertEqual(sum(len(arms(s)) for s in specs if s["split"]=="known"),42)
        self.assertEqual(sum(len(arms(s)) for s in specs if s["split"]=="prospective"),24)
        for spec in specs:
            self.assertTrue(all(a["selection"]=="J" for a in arms(spec)))
    def test_shapes_addresses_and_reference_stay_finite(self):
        for spec in specifications():
            if spec["split"]!="prospective":continue
            text=fixture(spec);rows,cols=spec["shape"]
            self.assertIn(f"memref<{rows}x{cols}xf32,1>",text)
            self.assertIn(f"{cols}*i+j",text)
            self.assertIn(f"output_extents=[{4*rows*cols},8]",text)
            q,k,v=inputs(spec);result=reference(spec,q,k,v)
            self.assertEqual(result.size,rows*cols);self.assertTrue(np.all(np.isfinite(result)))
    def test_registration_contains_frozen_thresholds_and_all_programs(self):
        p=protocol();self.assertEqual(len(p["specifications"]),18)
        self.assertEqual(p["acceptance"]["structural_shared"]["supported_error_bytes"],0)
        self.assertEqual(p["acceptance"]["physical_register_estimate"]["prospective_mae_max"],8)
        self.assertEqual(p["acceptance"]["physical_register_estimate"]["prospective_max_underprediction"],16)
        self.assertTrue(p["no_kernel_timing"])
        altered=dict(p);altered["seed"]=p["seed"]+1
        self.assertNotEqual(canonical_digest(p),canonical_digest(altered))


if __name__=="__main__":unittest.main()
