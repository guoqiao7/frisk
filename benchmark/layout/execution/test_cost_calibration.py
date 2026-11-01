import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import calibration_fixture as fixture
import calibration_analysis as analysis
import cost_calibration as driver

class FrozenFixtures(unittest.TestCase):
    def test_complete_spec_split_and_finite_references(self):
        rows=fixture.specifications()
        self.assertEqual(len(rows),12)
        self.assertEqual(len({r["key"] for r in rows}),12)
        self.assertFalse(set(r["shape"][0] for r in rows if r["split"]=="holdout")<={64})
        for spec in rows:
            if spec["case_id"]=="attention":continue
            text=fixture.fixture(spec);q,k,v=fixture.inputs(spec)
            self.assertTrue(np.isfinite(fixture.reference(spec,q,k,v)).all())
            self.assertIn("x".join(map(str,spec["shape"]))+"xf32",text)
            self.assertIn("joint.A=#A,joint.B=#B",text)
    def test_source_value_extraction_and_tail(self):
        specs=fixture.specifications()
        for tail,n in (("mul8",8),("div6",6),("exp1",1),("mixed",4)):
            spec=next(r for r in specs if r["tail"]==tail)
            text=fixture.fixture(spec)
            self.assertEqual(text.count("%cal"),2*n)
            self.assertIn("frisk.tile_store %cal"+str(n-1),text)
    def test_shapes_mapping_and_seed(self):
        for spec in fixture.specifications():
            if spec["case_id"]=="attention":continue
            shape=spec["shape"];bits=sum(int(np.log2(x)) for x in shape)
            self.assertIn(f"tensor<{bits}x{bits}xi1>",fixture.encoding(shape))
            self.assertNotEqual(fixture.encoding(shape),fixture.encoding(shape,True))
            self.assertFalse(np.array_equal(fixture.inputs(spec,0)[0],fixture.inputs(spec,1)[0]))
    def test_protocol_tampering_fails(self):
        with tempfile.TemporaryDirectory() as td:
            p=driver.init(Path(td)/"run");driver.verify_protocol(p)
            value=driver.read(p/"protocol.json");value["seed"]+=1;driver.write(p/"protocol.json",value)
            with self.assertRaises(ValueError):driver.verify_protocol(p)
    def test_no_import_time_gpu(self):
        self.assertTrue(callable(driver.measure))

class Fit(unittest.TestCase):
    def test_nonnegative_fit_known_solution(self):
        x=np.array([[0,0],[1,0],[0,1],[1,1],[2,1]],float)
        y=3+x@np.array([2.,4.])
        fit=analysis.coordinate_fit(x,y,ridge=0)
        self.assertTrue(fit["converged"])
        self.assertLess(fit["kkt_violation"],1e-8)
        np.testing.assert_allclose(fit["coefficients"],[2,4],rtol=1e-9)
        self.assertAlmostEqual(fit["intercept"],3)
    def test_correlated_features_disclose_rank(self):
        fit=analysis.coordinate_fit([[1,2,0],[2,4,0],[3,6,0]],[2,3,4])
        self.assertTrue(fit["converged"])
        self.assertLess(fit["design_rank"],fit["design_columns"])
        self.assertEqual(fit["active_columns"],[True,True,False])
    def test_negative_slope_is_bounded(self):
        fit=analysis.coordinate_fit([[1],[2],[3]],[5,4,3])
        self.assertTrue(fit["converged"]);self.assertEqual(fit["coefficients"],[0])
        self.assertAlmostEqual(fit["intercept"],4)
    def rows(self):
        rows=[]
        for i in range(4):
            f={k:0 for k in analysis.FEATURES};f["alu_warp_issues"]=i+1
            rows.append(dict(spec=dict(key="train"+str(i),split="train"),candidate="A",
                             features=f,median_us=2.+i,cubin_sha256="c"+str(i),measurement_sha256="m"+str(i)))
        rows.append(dict(spec=dict(key="holdout",split="holdout"),candidate="B",features={"invalid":object()},median_us=-100))
        return rows
    def test_holdout_cannot_affect_fit(self):
        rows=self.rows();p1,f1=analysis.fit_profile(rows,fixture.protocol())
        rows[-1]["median_us"]=object();rows[-1]["features"]=None
        p2,f2=analysis.fit_profile(rows,fixture.protocol())
        self.assertEqual(p1,p2);self.assertEqual(f1,f2)
        self.assertNotIn("holdout",f1["training_specs"])
        self.assertEqual(set(p1["coefficients"]),set(analysis.FEATURES))
    def test_bad_input_fails_closed(self):
        for x,y in (([[1],[np.nan]],[1,2]),([[1],[-1]],[1,2]),([[1],[2]],[-1,1])):
            with self.assertRaises(ValueError):analysis.coordinate_fit(x,y)
        with self.assertRaises(ValueError):analysis.features(dict(cost=dict(features={"unknown":1})))
        with self.assertRaises(ValueError):analysis.features(dict(cost=dict(features={"alu_warp_issues":True})))

class MachineEvidence(unittest.TestCase):
    def test_predicated_static_instructions(self):
        value=driver.parse_sass("/*0000*/ LDC R1, c[0x0][0x28];\n/*0010*/ @!P0 BAR.SYNC 0;\n/*0020*/ @UP1 LDL.64 R1,[R2];\n/* 0x0000000000 */")
        self.assertEqual(value,dict(LDC=1,BAR=1,LDL=1))
        with self.assertRaises(ValueError):driver.parse_sass("no machine instructions")

    def test_all_predicate_spellings_are_static_instructions(self):
        predicates=("P0","!P7","UP2","!UP3","PT","!PT","UPT","!UPT")
        for predicate in predicates:
            with self.subTest(predicate=predicate):
                value=driver.parse_sass("/*0010*/ @"+predicate+" LDS RZ, [RZ];")
                self.assertEqual(value,dict(LDS=1))
        value=driver.parse_sass("/*0010*/ @!PT LDS RZ, [RZ];\n/*0020*/ @PT LDS R1, [R2];\n/*0030*/ @!UPT STS [RZ], RZ;\n/*0040*/ @UPT STS [R1], R2;")
        self.assertEqual(value,dict(LDS=2,STS=2))

class ResourceFit(unittest.TestCase):
    def rows(self):
        result=[]
        for i in range(4):
            scope=dict(scope="parallel",status="available",saturated=False,logical_live_carrier_bytes_per_thread=4*(i+1),
                       logical_live_carrier_slots_per_thread=i+1,logical_live_shared_root_bytes=i*1024,declared_shared_root_bytes=i*1024)
            result.append(dict(spec=dict(key="t"+str(i),split="train"),candidate="A",resource_comparison=dict(scopes=[scope]),
                       physical_resources=dict(registers_per_thread=16+2*i,shared_bytes=i*1024,local_bytes=0,residency=dict(kind="theoretical-resource-ceiling"))))
        return result
    def test_train_mapping_frozen_for_holdout(self):
        rows=self.rows();model=analysis.legacy_ssa_resource_calibration(rows)["models"]
        new=dict(rows[-1]);new["spec"]=dict(key="h",split="holdout")
        new["physical_resources"]=dict(new["physical_resources"],registers_per_thread=200,shared_bytes=80000)
        assessed=analysis.legacy_ssa_resource_calibration(rows+[new],model)
        self.assertEqual(assessed["models"],model)
        self.assertEqual(assessed["errors"]["holdout"]["registers_per_thread"]["n"],1)
        self.assertGreater(assessed["errors"]["holdout"]["registers_per_thread"]["mae"],100)
    def test_unconverged_resource_model_unavailable(self):
        from unittest.mock import patch
        with patch.object(analysis,"coordinate_fit",return_value=dict(converged=False,kkt_violation=1.)):
            result=analysis.legacy_ssa_resource_calibration(self.rows())
        self.assertEqual(result["models"]["registers_per_thread"]["status"],"unavailable")
        self.assertNotIn("registers_per_thread_predicted",result["rows"][0])
    def test_unavailable_not_fabricated_zero(self):
        row=self.rows()[0];row["resource_comparison"]=dict(scopes=[dict(status="unavailable")])
        result=analysis.resource_calibration([row])
        self.assertEqual(len(result["unavailable"]),1);self.assertFalse(result["rows"])
    def test_default_requires_lowering_not_truth_fitted_physical_point(self):
        rows=self.rows()
        for row in rows: row["physical_resources"]["registers_per_thread"]=99999
        result=analysis.resource_calibration(rows)
        self.assertEqual(result["resource_model_version"],"lowering-compiler-v2")
        self.assertFalse(result["physical_prediction_qualified"])
        self.assertEqual(len(result["unavailable"]),len(rows))
        self.assertFalse(result["rows"])
    def test_legacy_frozen_profile_is_explicitly_unqualified(self):
        rows=self.rows();model=analysis.legacy_ssa_resource_calibration(rows)["models"]
        result=analysis.resource_calibration(rows,model)
        self.assertEqual(result["resource_model_version"],"legacy-ssa-regression")
        self.assertFalse(result["physical_prediction_qualified"])
    def test_full_source_identity_includes_new_files(self):
        files=driver.source_identity()["source_manifest"]
        self.assertIn("benchmark/layout/execution/calibration_analysis.py",files)
        self.assertIn("lib/Dialect/Frisk/Target/SM90/SM90CostModel.cpp",files)
        self.assertIn("CMakeLists.txt",files)

class PairedStatistics(unittest.TestCase):
    def rows(self):
        rows=[]
        for b in range(10):
            content=dict(batch=b,input_offset=1,output_offset=2)
            identity=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            for arm,ms in (("A",.004),("B",.002)):
                rows.append(dict(batch=b,arm=arm,pair_content=content.copy(),pair_id=identity,valid=True,
                                 launch_count=100,actual_graph_nodes=100,samples_ms=[ms]*5,graph_elapsed_ms=[ms*100]*5))
        return rows
    def test_paired_known_ratio(self):
        result=analysis.paired_analysis(self.rows(),["A","B"])
        self.assertTrue(result["formal"]);self.assertEqual(result["n"],10)
        self.assertEqual(result["comparisons"]["A/B"]["ratio_ci95"],[2.,2.])
    def test_no_pseudo_replication(self):
        rows=self.rows()
        for r in rows:r["samples_ms"]*=10;r["graph_elapsed_ms"]*=10
        self.assertEqual(analysis.paired_analysis(rows,["A","B"])["n"],10)
    def test_invalid_pair_not_replaced(self):
        rows=self.rows();rows[1]["valid"]=False
        result=analysis.paired_analysis(rows,["A","B"])
        self.assertFalse(result["formal"]);self.assertEqual(result["invalid_batches"],[0])
    def test_pair_and_graph_integrity(self):
        rows=self.rows();rows[0]["actual_graph_nodes"]=99
        with self.assertRaises(ValueError):analysis.paired_analysis(rows,["A","B"])
        rows=self.rows();rows[0]["pair_content"]["output_offset"]=9
        with self.assertRaises(ValueError):analysis.paired_analysis(rows,["A","B"])

class MeasurementBinding(unittest.TestCase):
    def fake(self,base,phase="initial",profile=None):
        run=driver.init(base/"run");measurement=base/"measurement";measurement.mkdir()
        tool=base/"fake-tool";tool.write_bytes(b"test-only native identity")
        guards=driver.input_hashes(run,profile)
        rows=[] if profile is None else [dict(arm="calibrated-J",profile_sha256=driver.sha(profile))]
        manifest=dict(status="validated",identity=driver.source_identity(),tools=dict(hashes={str(tool):driver.sha(tool)}),input_hashes=guards,rows=rows)
        driver.write(run/(phase+"-manifest.json"),manifest)
        frozen=dict(run=str(run.resolve()),phase=phase,protocol_sha256=driver.sha(run/"protocol.json"),
                    manifest_sha256=driver.sha(run/(phase+"-manifest.json")),source=manifest["identity"],tools=manifest["tools"],input_hashes=guards)
        driver.write(measurement/"freeze.json",frozen);driver.write(measurement/"observations.json",[]);driver.write(measurement/"analysis.json",[])
        self.seal(run,measurement,phase)
        return run,measurement
    def seal(self,run,measurement,phase):
        evidence={str(p.relative_to(measurement)):driver.sha(p) for p in measurement.rglob("*") if p.is_file() and p.name!="completion.json"}
        driver.write(measurement/"completion.json",dict(status="complete",run=str(run.resolve()),phase=phase,
                     protocol_sha256=driver.sha(run/"protocol.json"),manifest_sha256=driver.sha(run/(phase+"-manifest.json")),
                     freeze_sha256=driver.sha(measurement/"freeze.json"),evidence_hashes=evidence))
    def test_initial_and_other_run_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);run,m=self.fake(base)
            self.assertEqual(driver.verify_measurement(run,m,"initial"),[])
            another=driver.init(base/"another")
            # Even an identical protocol is insufficient: run identity must match.
            (another/"initial-manifest.json").write_bytes((run/"initial-manifest.json").read_bytes())
            with self.assertRaises(ValueError):driver.verify_measurement(another,m,"initial")
    def test_phase_and_payload_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            run,m=self.fake(Path(td))
            with self.assertRaises(ValueError):driver.verify_measurement(run,m,"selected",run/"unit-profile.json")
            (m/"analysis.json").write_text("[1]")
            with self.assertRaises(ValueError):driver.verify_measurement(run,m,"initial")
    def test_selected_profile_binding(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);profile=base/"profile.json";profile.write_text('{"test-only":"profile A"}')
            run,m=self.fake(base,"selected",profile)
            self.assertEqual(driver.verify_measurement(run,m,"selected",profile),[])
            another=base/"other-profile.json";another.write_text('{"test-only":"profile B"}')
            with self.assertRaises(ValueError):driver.verify_measurement(run,m,"selected",another)
    def test_freeze_and_source_identity_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            run,m=self.fake(Path(td));freeze=driver.read(m/"freeze.json")
            freeze["source"]["source_digest"]="wrong";driver.write(m/"freeze.json",freeze);self.seal(run,m,"initial")
            with self.assertRaises(ValueError):driver.verify_measurement(run,m,"initial")

class DocumentedFailures(unittest.TestCase):
    def value(self):
        return dict(status="failed",failure_kind="unsupported",fatal=False,reason="LI-O local/future sharing unavailable",limits=dict(probes=65536,refinement=256))
    def test_only_public_code3_is_retained(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);driver.write(d/"failure.json",self.value())
            value=driver.documented_outcome(3,d)
            self.assertEqual(value["status"],"documented-failure")
            self.assertEqual(value["failure_kind"],"unsupported")
            for code in (0,1,2,-9,-15):
                with self.assertRaises(RuntimeError):driver.documented_outcome(code,d)
    def test_missing_malformed_and_fatal_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            with self.assertRaises(FileNotFoundError):driver.documented_outcome(3,d)
            (d/"failure.json").write_text("{bad")
            with self.assertRaises(ValueError):driver.documented_outcome(3,d)
            for change in (dict(fatal=True),dict(fatal=0),dict(failure_kind="integrity"),dict(reason=""),dict(limits=dict(probes=True,refinement=1)),dict(extra="unknown")):
                value=self.value();value.update(change);driver.write(d/"failure.json",value)
                with self.assertRaises(ValueError):driver.documented_outcome(3,d)
    def test_validation_retains_documented_row_without_launch(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            run=driver.init(Path(td)/"run")
            failure=run/"unavailable";failure.mkdir();driver.write(failure/"failure.json",self.value())
            row=dict(driver.documented_outcome(3,failure),path=str(failure),artifact_hashes={"failure.json":driver.sha(failure/"failure.json")})
            manifest=dict(status="generated",input_hashes=driver.input_hashes(run),tools=dict(hashes={}),
                          identity=driver.source_identity(),rows=[row])
            driver.write(run/"initial-manifest.json",manifest)
            with patch.object(driver,"invoke",side_effect=AssertionError("must not launch unavailable row")):
                driver.validate(run,"initial")
            after=driver.read(run/"initial-manifest.json")
            self.assertEqual(after["status"],"validated");self.assertEqual(after["rows"],[row])
            with self.assertRaises(ValueError):driver.measurement_groups(run,"initial")

if __name__=="__main__":unittest.main()
