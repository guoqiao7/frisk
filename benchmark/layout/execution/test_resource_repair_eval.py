import copy
import unittest
import json,hashlib,tempfile,time,types,sys
from pathlib import Path
from unittest.mock import patch
from resource_repair_eval import point_value,summarize_evidence,checked_run


def row(key="prospective-x",arm="A-C",conversion="C",prediction=None,actual=64):
    return dict(spec=dict(key=key,split="prospective"),arm=arm,assignment_sha256="same",
        conversion=conversion,post_opt=conversion=="O",shared_error_bytes=0,register_prediction=prediction,
        actual_registers=actual,register_error=prediction-actual if prediction is not None else None,
        compiler_driver_verified=True,local_bytes=0)


class ResourceRepairEvalTest(unittest.TestCase):
    threshold=dict(prospective_mae_max=8.,prospective_max_underprediction=16.)
    def test_missing_point_is_not_zero_error_or_qualified(self):
        result=summarize_evidence([row()],self.threshold)
        self.assertIsNone(result["physical_register_estimator"]["mae"])
        self.assertEqual(result["physical_register_estimator"]["unavailable"],1)
        self.assertFalse(result["physical_register_estimator"]["qualified"])
        self.assertTrue(result["compilation_feedback_required_for_unqualified"])
    def test_duplicate_arms_collapse_but_conversion_policies_do_not(self):
        a=row(prediction=64);duplicate=dict(a,arm="alias");different=row(arm="A-O",conversion="O",prediction=64)
        result=summarize_evidence([a,duplicate,different],self.threshold)
        self.assertEqual(result["observations"],3);self.assertEqual(result["unique_configurations"],2)
    def test_duplicate_contradictory_evidence_rejected(self):
        a=row(prediction=64);b=dict(a,actual_registers=80)
        with self.assertRaisesRegex(ValueError,"inconsistent"):summarize_evidence([a,b],self.threshold)
    def test_accuracy_and_underprediction_both_gate_qualification(self):
        for cases in ([row(prediction=40)], [row(prediction=80)]):
            result=summarize_evidence(cases,self.threshold)
            self.assertFalse(result["physical_register_estimator"]["qualified"])
        result=summarize_evidence([row(prediction=60)],self.threshold)
        self.assertTrue(result["physical_register_estimator"]["qualified"])
    def test_structural_and_feedback_gates_do_not_hide_inaccurate_points(self):
        a=row(prediction=64);a["shared_error_bytes"]=16;a["compiler_driver_verified"]=False
        result=summarize_evidence([a],self.threshold)
        self.assertFalse(result["structural_shared_qualified"]);self.assertFalse(result["compiler_driver_feedback_verified"])
    def test_unknown_point_and_malformed_numeric_points(self):
        self.assertIsNone(point_value(dict(physical_registers=None)))
        self.assertIsNone(point_value(dict(physical_registers=dict(status="unavailable",requires_compilation=True))))
        for value in (-1,True,float("nan"),float("inf")):
            with self.subTest(value=value),self.assertRaises(ValueError):point_value(dict(physical_registers=value))
    def test_output_root_prevents_old_artifact_controller_misrouting(self):
        with self.assertRaises(ValueError):checked_run("/home/baopeihua/frisk/guoqiao/optimize_plan/cost_resource_calibration_artifacts/old")
        with self.assertRaises(ValueError):checked_run("/home/baopeihua/frisk/guoqiao/optimize_plan/resource_prediction_repair_artifacts")

    def test_gpu_preflight_accepts_successful_empty_process_query(self):
        from resource_repair_eval import check_idle_state,VALIDATION_GPU_UUID
        state=dict(device=dict(exit_code=0,stdout=VALIDATION_GPU_UUID+"\n",stderr=""),
                   processes=dict(exit_code=0,stdout="",stderr=""))
        self.assertEqual(check_idle_state(state),[])
        state["processes"]["stdout"]="   "
        with self.assertRaises(ValueError):check_idle_state(state)

    def test_gpu_preflight_rejects_untrusted_queries_and_foreign_work(self):
        from resource_repair_eval import check_idle_state,VALIDATION_GPU_UUID
        empty=dict(device=dict(exit_code=0,stdout=VALIDATION_GPU_UUID+"\n",stderr=""),
                   processes=dict(exit_code=0,stdout="",stderr=""))
        for field,value in (("exit_code",1),("stderr","query warning"),("stdout","bad, row\n")):
            state=copy.deepcopy(empty);state["processes"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):check_idle_state(state)
        for name in ("device","processes"):
            state=copy.deepcopy(empty)
            state[name]["stdout"]="GPU-other\n" if name=="device" else "GPU-other, 42, python, 10\n"
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,"UUID mismatch"):check_idle_state(state)
        state=copy.deepcopy(empty);state["processes"]["stdout"]=VALIDATION_GPU_UUID+", 42, python, 10\n"
        with self.assertRaisesRegex(RuntimeError,"foreign GPU workload"):check_idle_state(state)
        state["processes"]["stdout"]=VALIDATION_GPU_UUID+", no-pid, python, 10\n"
        with self.assertRaises(ValueError):check_idle_state(state)

    def test_gpu_preflight_retains_raw_failed_query_evidence(self):
        import resource_repair_eval as evaluation
        output=types.SimpleNamespace(returncode=0,stdout="GPU-other\n",stderr="")
        with patch.object(evaluation.subprocess,"run",return_value=output),self.assertRaises(evaluation.GpuPreflightError) as caught:
            evaluation.idle_preflight()
        self.assertEqual(caught.exception.state["status"],"failed")
        self.assertEqual(caught.exception.state["device"]["stdout"],"GPU-other\n")
        self.assertIn(evaluation.VALIDATION_GPU_UUID,caught.exception.state["device"]["command"])

    def test_worker_reuses_first_ptx_event_not_after_feedback_reanalysis(self):
        import resource_repair_eval as evaluation
        with tempfile.TemporaryDirectory(dir=evaluation.ART,prefix="unit-first-event-") as name:
            d=Path(name);(d/"execution.json").write_text("{}")
            prediction=dict(shared=dict(physical_shared_bytes=4),physical_registers=None)
            first_event={}
            def normal_compile(directory):
                (directory/"kernel.ptx").write_text("fake PTX, no compiler invocation")
                (directory/"kernel.cubin").write_bytes(b"fake image")
                image=hashlib.sha256((directory/"kernel.cubin").read_bytes()).hexdigest()
                first_event.update(created_ns=time.time_ns(),prediction=prediction,reads_native_resources=False,
                    information_stage="PTX-only-pre-RA-information",wallclock_stage="after-combined-Triton-compile")
                (directory/"resource-pressure-ptx-only.json").write_text(json.dumps(first_event))
                resolved=dict(compiler_exact=dict(identity=dict(cubin_sha256=image)),raw_native_shared_bytes=1028)
                (directory/"resolved-compiler-resources.json").write_text(json.dumps(resolved))
            fake=types.SimpleNamespace(compile_exact=normal_compile)
            def normal_execute(directory,*args,**kwargs):
                fake.compile_exact(directory)
                image=hashlib.sha256((directory/"kernel.cubin").read_bytes()).hexdigest()
                return dict(physical_resources=dict(registers_per_thread=8,shared_bytes=4,local_bytes=0),
                    compiler_resource_confirmation=dict(status="confirmed",registers_per_thread=8,shared_bytes=4,local_bytes=0,
                                                        loaded_cubin_sha256=image))
            fake.execute=normal_execute
            with patch.dict(sys.modules,{"cost_calibration":fake}), patch("resource_prediction.predict_resources",return_value=prediction) as predictor:
                evaluation.worker(d)
                self.assertEqual(predictor.call_count,1,"wrapper must not redo PTX estimation after internal native feedback")
            evidence=json.loads((d/"resource-repair-evidence.json").read_text())
            self.assertEqual(evidence["estimate_ptx_only"]["created_ns"],first_event["created_ns"])
            self.assertTrue(evidence["estimate_ptx_only"]["reused_first_pre_feedback_record"])
            self.assertTrue(evidence["exact_feedback_is_not_a_prediction"])


if __name__=="__main__":unittest.main()
