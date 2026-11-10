"""Focused CPU regressions for the fixed native selection and calibration gate."""

import copy
import hashlib
import json
import unittest

from execution_plan_cost import extract_features
from fixed_execution_decision import describe, fit_profile, fixed_choice, prediction


_GRID = [240, 1, 1]
_BLOCK = [128, 1, 1]
_POLICY = {"access": "factored", "mma_group": 1, "reduction": "auto", "shared": "declared", "store": "direct"}
_WIDTH = {4: "f32", 8: "v2.f32", 16: "v4.f32"}


def features(loads, stores, *, publication=0, batched=False, operand='x'):
    instructions = ["{"] + [f"ld.global.{_WIDTH[width]} x, [a];" for width in loads]
    instructions += ["bar.sync 0;", "fence.proxy.async.shared::cta;"] * publication
    instructions += [f"st.global.{_WIDTH[width]} [b], {operand};" for width in stores] + ["}"]
    asm = r"\0A".join(instructions)
    text = (
        'module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {\n'
        'tt.func public @frisk_execution() {\n'
        f'%v = tt.elementwise_inline_asm "{asm}" {{pure = false}}\n'
        'tt.return\n}\n}\n'
    )
    access = [
        {"operation": operation, "element_bytes": 4,
         "groups_by_width": {width: widths.count(width) for width in set(widths)},
         "sector_issues32_upper": 32}
        for operation, widths in (("load", loads), ("store", stores))
    ]
    metadata = {"global_access": access, "global_alignments": [16] * 4}
    if publication:
        metadata.update(copy_path="vector", global_access=[dict(access[0], operation="copy"), access[1]],
                        execution_trace=[dict(operation="frisk.copy", barriers=publication,
                                              shuffles=0, wgmma=0, waits=0)])
        if batched:
            metadata["copy_batches"] = [{"operations": [0, 1, 2]}]
    return extract_features(text, metadata, grid=_GRID, block=_BLOCK)


def admitted(arm, *, loads=(4, 4), stores=(8,), access="factored", blocks=3, local=0):
    image = {
        "cubin_sha256": hashlib.sha256(arm.encode()).hexdigest(), "grid": _GRID, "block": _BLOCK,
        "entry": "frisk_execution", "dynamic_shared_bytes": 0,
    }
    identity = hashlib.sha256(json.dumps(image, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(image, image_identity=identity, status="admitted", case="prospective", arm=arm,
                program_sha256=arm, packet_sha256="same-packet", spec={"operator": "mul"},
                policy=dict(_POLICY, access=access), preview_features=features(loads, stores),
                resources={"device_ordinal": 1, "device_limits": {"multiprocessor_count": 114},
                           "residency": {"active_blocks_per_sm": blocks, "launchable_by_resources": True},
                           "registers_per_thread": 32, "local_bytes": local})


def trial(name, contrast, ratio=.90, *, shape=None, family=None):
    return {"id": name, "shape": shape or name, "family": family or contrast["family"],
            "mechanism": contrast["mechanism"], "contrast": contrast,
            "ratios": [ratio] * 12, "offsets": [0, 16, 32] * 4, "provenance": {"synthetic": True}}


class FixedExecutionDecisionTest(unittest.TestCase):
    def setUp(self):
        self.base = admitted("incumbent")
        # Fewer loads and more stores creates an unresolved structural tradeoff.
        self.variant = admitted("variant", loads=(8,), stores=(4, 4), access="vector")
        self.contrast = describe(self.variant, self.base)

    def profile(self, ratios=(.90, .90, .90)):
        return fit_profile([trial(f"dev-{i}", self.contrast, ratio) for i, ratio in enumerate(ratios)])

    def test_supported_intervention_can_override_structural_tradeoff(self):
        profile = self.profile()
        estimate = prediction(profile, self.contrast, "prospective")
        self.assertEqual((estimate["shapes"], estimate["lower"], estimate["upper"]), (3, .90, .90))
        selected, receipt = fixed_choice([self.variant], self.base, profile)
        self.assertIs(selected, self.variant)
        self.assertEqual(receipt["reason"], "calibrated-win")
        self.assertEqual(receipt["candidates"]["variant"]["structural"], "incomparable")

    def test_no_support_or_no_leave_shape_fold_retains_fallback(self):
        for profile in (fit_profile([]), self.profile((.90,)), self.profile((.90, .90))):
            with self.subTest(trials=len(profile["evidence"])):
                self.assertIsNone(prediction(profile, self.contrast, "prospective"))
                selected, receipt = fixed_choice([self.variant], self.base, profile)
                self.assertIs(selected, self.base)
                self.assertEqual(receipt["reason"], "pinned-fallback")
        duplicates = fit_profile([trial(f"replica-{i}", self.contrast, shape="one-shape") for i in range(3)])
        self.assertIsNone(prediction(duplicates, self.contrast, "prospective"))

    def test_extrapolation_is_rejected(self):
        profile = self.profile()
        for field in ("coordinates", "key"):
            changed = copy.deepcopy(self.contrast)
            changed[field][0] = 999
            with self.subTest(field=field):
                self.assertIsNone(prediction(profile, changed, "prospective"))

    def test_leave_one_shape_drops_the_entire_shape_and_can_veto(self):
        profile = self.profile((.90, .90, .97))
        folds = {fold["id"]: fold for fold in profile["leave_one_shape_out"]}
        self.assertEqual(folds["dev-2"]["prediction"]["support"], ["dev-0", "dev-1"])
        self.assertFalse(folds["dev-2"]["covers"])
        self.assertIsNone(prediction(profile, self.contrast, "prospective"))
        self.assertTrue(all(fold["prediction"] is None for fold in profile["leave_one_family_out"]))

    def test_resource_guards_take_precedence(self):
        profile = self.profile()
        for parameters in ({"blocks": 2}, {"local": 4}):
            candidate = admitted("variant", loads=(8,), stores=(4, 4), access="vector", **parameters)
            with self.subTest(parameters=parameters):
                selected, receipt = fixed_choice([candidate], self.base, profile)
                self.assertIs(selected, self.base)
                self.assertEqual(receipt["candidates"]["variant"]["reason"], "resource-regression")

    def test_structural_preference_requires_dominance(self):
        fast = admitted("vector", loads=(8,), access="vector")
        selected, receipt = fixed_choice([self.variant, fast], self.base, fit_profile([]))
        self.assertIs(selected, fast)
        self.assertEqual(receipt["reason"], "structural-preference")
        self.assertEqual(receipt["candidates"]["variant"]["reason"], "unresolved-tradeoff")
        tie = admitted("tie")
        selected, receipt = fixed_choice([tie], self.base, fit_profile([]))
        self.assertIs(selected, self.base)
        self.assertEqual(receipt["reason"], "pinned-fallback")

    def test_calibrated_regression_vetoes_structural_preference(self):
        fast = admitted("vector", loads=(8,), access="vector")
        contrast = describe(fast, self.base)
        profile = fit_profile([trial(f"regression-{i}", contrast, 1.08) for i in range(3)])
        selected, receipt = fixed_choice([fast], self.base, profile)
        self.assertIs(selected, self.base)
        self.assertEqual(receipt["candidates"]["vector"]["reason"], "calibrated-regression")

    def test_copy_publication_requires_checked_control_and_resource_band(self):
        old = admitted("old", blocks=4)
        new = admitted("new", blocks=4)
        old["preview_features"] = features((4, 4), (8,), publication=3)
        new["preview_features"] = features((4, 4), (8,), publication=1, batched=True)
        new["policy"]["copy_batch"] = True
        self.assertEqual(describe(new, old)["mechanism"], "copy-publication")
        self.assertEqual(describe(new, old)["key"][1], ["copy_batch", False, True])
        selected, audit = fixed_choice([new], old, fit_profile([]))
        self.assertIs(selected, new)
        self.assertEqual(audit["reason"], "copy-publication-preference")
        self.assertEqual(audit["candidates"]["new"]["control_arm"], "old")

        changed = copy.deepcopy(new)
        changed["preview_features"] = features((4, 4), (8,), publication=1, batched=True, operand="y")
        selected, audit = fixed_choice([changed], old, fit_profile([]))
        self.assertIs(selected, old)
        self.assertEqual(audit["candidates"]["new"]["reason"], "no-matched-publication-control")

        lower_residency = copy.deepcopy(new)
        lower_residency["resources"]["residency"]["active_blocks_per_sm"] = 3
        selected, audit = fixed_choice([lower_residency], old, fit_profile([]))
        self.assertIs(selected, old)
        self.assertEqual(audit["candidates"]["new"]["reason"], "publication-resource-regression")

    def test_copy_publication_cannot_use_a_failed_or_different_layout_control(self):
        base, control, batch = (admitted(arm) for arm in ("base", "control", "batch"))
        for row, n, batched in ((base, 3, False), (control, 3, False), (batch, 1, True)):
            row["preview_features"] = features((4, 4), (8,), publication=n, batched=batched)
        for row in (control, batch):
            row["policy"]["mma_group"] = 16
        batch["policy"]["copy_batch"] = True
        for mismatch in ("failed", "other-packet"):
            candidate = copy.deepcopy(control)
            if mismatch == "failed":
                candidate["status"] = mismatch
            else:
                candidate["packet_sha256"] = mismatch
            selected, audit = fixed_choice([candidate, batch], base, fit_profile([]))
            self.assertIs(selected, base)
            self.assertEqual(audit["candidates"]["batch"]["reason"], "no-matched-publication-control")

    def test_access_classification_checks_the_width_histogram(self):
        original = admitted("mixed-width", loads=(4, 4, 16))
        repacked = admitted("different-widths", loads=(8, 8, 8), access="vector")
        self.assertEqual(describe(repacked, original)["mechanism"], "access-width")
        unchanged_widths = admitted("new-addresses", loads=(4, 4, 16), access="vector")
        self.assertEqual(describe(unchanged_widths, original)["mechanism"], "address-only")


if __name__ == "__main__":
    unittest.main()
