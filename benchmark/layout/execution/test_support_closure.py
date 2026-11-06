"""CPU-only contract tests for the current support closure driver."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import support_closure as driver
import support_closure_fixture as fixture


class ClosureFixtureTests(unittest.TestCase):
    def test_declared_matrix_preserves_large_and_small_shapes(self):
        specs=fixture.specifications()
        self.assertEqual(len(specs),11)
        self.assertEqual(len({x["key"] for x in specs}),11)
        self.assertEqual({tuple(x["shape"]) for x in specs if x["graph_name"]=="mixed"},{(128,32),(32,16)})
        self.assertEqual(len([(x,m) for x in specs for m in fixture.protocol()["models"]]),22)
        self.assertTrue(fixture.protocol()["no_kernel_timing"])

    def test_natural_sources_have_no_controlled_layout_metadata(self):
        for spec in fixture.specifications():
            text=fixture.source(spec,driver.ROOT)
            self.assertNotIn("joint.A",text);self.assertNotIn("joint.B",text)
            self.assertNotIn("force-producer",text)
            self.assertIn("frisk.tile_store",text)
            if spec["case_id"]!="attention":
                self.assertIn("tensor<"+"x".join(map(str,spec["shape"]))+"xf32>",text)

    def test_f32_inputs_and_reference_are_finite_with_nonzero_divisors(self):
        for spec in fixture.specifications()[:8]:
            cases=fixture.numerical_cases(spec)
            self.assertEqual(len(cases),7)
            self.assertEqual([x[0] for x in cases],["seed0","seed1","seed2","zeroQ","constant","signedcoords","cancellation"])
            for name,q,k,v in cases:
                self.assertEqual(q.dtype,np.float32);self.assertTrue(np.all(k!=0))
                ref=fixture.reference(spec,q,k,v)
                self.assertTrue(fixture.check_numerics(spec,ref,ref)["passed"])
                self.assertTrue(np.isfinite(ref).all())

    def test_bitexact_checker_rejects_signed_zero_change(self):
        spec=fixture.specifications()[0]
        positive=np.array([0.0],np.float32);negative=np.array([-0.0],np.float32)
        self.assertTrue(np.array_equal(positive,negative))
        self.assertFalse(fixture.check_numerics(spec,positive,negative)["passed"])
        self.assertFalse(fixture.check_numerics(spec,np.array([np.nan],np.float32),positive)["passed"])

    def test_mixed_threshold_stays_frozen(self):
        spec=next(x for x in fixture.specifications() if x["tail"]=="mixed")
        reference=np.array([1.0],np.float32)
        self.assertTrue(fixture.check_numerics(spec,reference+np.float32(1e-6),reference)["passed"])
        self.assertFalse(fixture.check_numerics(spec,reference+np.float32(1e-3),reference)["passed"])

    def test_strong_offsets_cross_input_and_output_independently(self):
        spec=fixture.specifications()[-1]
        self.assertEqual(set(fixture.offsets(spec,"tma")),{(i,o) for i in (0,128,256) for o in (1,2,3)})
        self.assertEqual(fixture.offsets(spec,"vector"),[(1,1),(2,2),(3,3)])

    def test_safety_identity_includes_spec_input_protocol_and_abi(self):
        spec=fixture.specifications()[0]
        info={"copy_path":"vector"};a=fixture.safety_input_identity(spec,info,["q","k","v","o","global"])
        b=fixture.safety_input_identity(fixture.specifications()[4],info,["q","k","v","o","global"])
        c=fixture.safety_input_identity(spec,info,["q","k","v","o","other"])
        self.assertNotEqual(fixture.digest(a),fixture.digest(b));self.assertNotEqual(fixture.digest(a),fixture.digest(c))

    def test_attention_reference_retains_existing_case_set(self):
        spec=fixture.specifications()[-1]
        self.assertEqual(len(fixture.numerical_cases(spec)),15)
        self.assertEqual(fixture.reference(spec,*fixture.numerical_cases(spec)[0][1:]).shape,(64,16))


class ClosureIntegrityTests(unittest.TestCase):
    def test_cuda_lazy_initialization_default_does_not_change_effective_identity(self):
        with patch.dict(driver.os.environ,{},clear=True):
            initial=driver.execution_environment()
            self.assertEqual(initial["CUDA_MODULE_LOADING"],"LAZY")
            driver.os.environ["CUDA_MODULE_LOADING"]="LAZY"
            self.assertEqual(initial,driver.execution_environment())
            driver.os.environ["CUDA_MODULE_LOADING"]="EAGER"
            self.assertNotEqual(initial,driver.execution_environment())

    def test_out_of_place_abi_allows_adjacent_output_and_overlapping_inputs(self):
        result=driver.validate_runtime_regions([100,100,104,120],[16,20,4,16])
        self.assertTrue(result["runtime_abi_checked"])
        with self.assertRaisesRegex(ValueError,"overlap"):
            driver.validate_runtime_regions([100,100,104,119],[16,20,4,16])

    def test_runtime_pointer_ranges_reject_invalid_and_uint64_overflow(self):
        for pointer,length in [(0,4),(-1,4),(True,4),(1,0),(1,-1),(1,1.5),((1<<64)-1,1),(1<<64,1)]:
            with self.assertRaises(ValueError):driver.validate_runtime_regions([pointer,100,200,300],[length,4,4,4])

    def test_sanitizer_environment_accepts_only_the_exact_frozen_prefix(self):
        with patch.dict(driver.os.environ,{"LD_LIBRARY_PATH":"/original/lib::/another"},clear=True):
            expected=driver.execution_environment();prefix="/frozen/cuda/compute-sanitizer"
            driver.os.environ["LD_LIBRARY_PATH"]=prefix+":"+expected["LD_LIBRARY_PATH"]
            evidence=driver.verify_environment(expected,sanitizer_prefix=prefix)
            self.assertEqual(evidence["canonical"],expected)
            self.assertEqual(evidence["raw"]["LD_LIBRARY_PATH"],prefix+":/original/lib::/another")
            with self.assertRaises(ValueError):driver.verify_environment(expected)
            for unexpected in ["/evil:"+expected["LD_LIBRARY_PATH"],prefix+":/evil:"+expected["LD_LIBRARY_PATH"],prefix+":/another:/original/lib",prefix+":"+prefix+":"+expected["LD_LIBRARY_PATH"]]:
                driver.os.environ["LD_LIBRARY_PATH"]=unexpected
                with self.assertRaises(ValueError):driver.verify_environment(expected,sanitizer_prefix=prefix)

    def test_sanitizer_environment_preserves_unset_original_path(self):
        with patch.dict(driver.os.environ,{},clear=True):
            expected=driver.execution_environment();prefix="/frozen/runtime"
            driver.os.environ["LD_LIBRARY_PATH"]=prefix
            self.assertIsNone(driver.verify_environment(expected,sanitizer_prefix=prefix)["canonical"]["LD_LIBRARY_PATH"])
            driver.os.environ["CUDA_VISIBLE_DEVICES"]="different"
            with self.assertRaises(ValueError):driver.verify_environment(expected,sanitizer_prefix=prefix)

    def test_empty_successful_gpu_process_query_is_idle(self):
        device=SimpleNamespace(returncode=0,stderr="",stdout=fixture.GPU_UUID+", NVIDIA H100, 570.0\n")
        empty=SimpleNamespace(returncode=0,stderr="",stdout="")
        self.assertEqual(driver.parse_gpu_state(device,empty)["foreign_pids"],[])
        occupied=SimpleNamespace(returncode=0,stderr="",stdout=fixture.GPU_UUID+", 123456789, python\n")
        self.assertEqual(driver.parse_gpu_state(device,occupied)["foreign_pids"],[123456789])
        with self.assertRaises(ValueError):driver.parse_gpu_state(device,SimpleNamespace(returncode=0,stderr="warning",stdout=""))
        with self.assertRaises(ValueError):driver.parse_gpu_state(device,SimpleNamespace(returncode=0,stderr="",stdout="bad"))

    def test_exact_artifact_guard_rejects_changed_or_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);(d/"kernel.cubin").write_bytes(b"one");identity={"kernel.cubin":driver.sha(d/"kernel.cubin")}
            driver.verify_files(d,identity);(d/"kernel.cubin").write_bytes(b"two")
            with self.assertRaises(ValueError):driver.verify_files(d,identity)
            (d/"kernel.cubin").unlink()
            with self.assertRaises(ValueError):driver.verify_files(d,identity)

    def test_accepted_artifacts_remain_bound_after_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            spec={"key":"fixture"}
            driver.write(directory/"configuration.json",dict(spec=spec,model="legacy"))
            driver.write(directory/"calibration_spec.json",spec)
            (directory/"selected.mlir").write_text("selected original")
            (directory/"kernel.cubin").write_bytes(b"native original")
            driver.write(directory/"generation-inputs.json",{"selected.mlir":driver.sha(directory/"selected.mlir")})
            driver.write(directory/"image-identity.json",{"kernel.cubin":driver.sha(directory/"kernel.cubin"),"generation-inputs.json":driver.sha(directory/"generation-inputs.json")})
            driver.write(directory/"correctness.json",dict(status="passed",cubin_sha256=driver.sha(directory/"kernel.cubin"),gpu_uuid=fixture.GPU_UUID,image_identity_sha256=driver.sha(directory/"image-identity.json")))
            row=dict(status="validated",spec=spec,model="legacy",generation_inputs_sha256=driver.sha(directory/"generation-inputs.json"),image_identity_sha256=driver.sha(directory/"image-identity.json"),correctness_sha256=driver.sha(directory/"correctness.json"),artifact_hashes={x.name:driver.sha(x) for x in directory.iterdir()})
            driver.verify_accepted_row(directory,row)
            for path in list(directory.iterdir()):
                original=path.read_bytes();path.write_bytes(original+b"changed")
                with self.assertRaises(ValueError):driver.verify_accepted_row(directory,row)
                path.write_bytes(original)

    def test_matrix_cannot_duplicate_or_substitute_a_predeclared_case(self):
        protocol=fixture.protocol();rows=[dict(spec=x,model=m) for x in protocol["specifications"] for m in protocol["models"]]
        driver.verify_declared_rows(rows,protocol)
        with self.assertRaises(ValueError):driver.verify_declared_rows(rows[:-1],protocol)
        with self.assertRaises(ValueError):driver.verify_declared_rows(rows[:-1]+[rows[0]],protocol)
        bad=[dict(x) for x in rows];bad[0]=dict(bad[0],spec=dict(bad[0]["spec"],shape=[1,1]))
        with self.assertRaises(ValueError):driver.verify_declared_rows(bad,protocol)

    def test_manifest_config_lookup_rejects_missing_and_duplicate(self):
        row={"path":"/tmp/closure-a"}
        self.assertEqual(driver.row_for(Path(row["path"]),{"rows":[row]}),row)
        with self.assertRaises(ValueError):driver.row_for(Path(row["path"]),{"rows":[]})
        with self.assertRaises(ValueError):driver.row_for(Path(row["path"]),{"rows":[row,row]})

    def test_target_device_is_selected_before_native_compiler(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/"run"/"cases"/"case";directory.mkdir(parents=True)
            driver.write(directory/"generation-inputs.json",{})
            row=dict(path=str(directory.resolve()),generation_inputs_sha256=driver.sha(directory/"generation-inputs.json"))
            order=[]
            def compiler(path):
                order.append("compile");raise ValueError("intentional CPU stop before native work")
            with patch.object(driver,"verify_run",return_value=({"rows":[row]},{})),patch.object(driver,"activate_target_device",side_effect=lambda:order.append("device") or dict(gpu_uuid=fixture.GPU_UUID)),patch("cost_calibration.compile_exact",side_effect=compiler):
                with self.assertRaises(driver.ClosureFailure):driver.execute(directory)
            self.assertEqual(order,["device","compile"])
            self.assertTrue((directory/"target-device-before-native.json").is_file())

    def test_safety_requires_accepted_named_load_only_worker(self):
        with self.assertRaisesRegex(ValueError,"named sanitizer"):
            driver.execute(Path("/tmp/unread"),load_only=True,safety=True)
        with self.assertRaisesRegex(ValueError,"accepted load-only"):
            driver.execute(Path("/tmp/unread"),safety=True,safety_tool="memcheck")

if __name__=="__main__":unittest.main()
