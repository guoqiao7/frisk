import unittest

from normalization_schedule import instruction_stream, wave_service, grid_service, rank_grid


class NormalizationScheduleTest(unittest.TestCase):
    def test_wait_mask_consumes_issuing_slot_once(self):
        # Two outstanding loads share one barrier. Only the later deadline is exposed.
        stream = [("LDG", 0, 2, 1), ("LDG", 0, 2, 1), ("FADD", 4, 7, 1),
                  ("FADD", 4, 7, 1), ("EXIT", 0, 7, 1)]
        one = wave_service(stream, 1)
        four = wave_service(stream, 4)
        self.assertEqual((one["issue"], one["global_wait"]), (5, 283))
        self.assertEqual((four["issue"], four["global_wait"]), (20, 280))
        self.assertEqual(wave_service([("LDL", 0, 1, 1), ("IADD3", 1, 7, 1)], 1)["local_wait"], 0)

    def test_fallthrough_and_function_bounds(self):
        def listing(lines):
            return "Function : full\n" + "\n".join(
                f"/*{16*i:04x}*/ {op}; /* 0x{'0'*16} */ /* 0x{'0'*16} */"
                for i, op in enumerate(lines))
        self.assertEqual(len(instruction_stream(listing(["FADD R2,R2,R3", "EXIT", "BRA 0x20"]), "full")), 2)
        for lines in (["BRA 0x10", "EXIT"], ["@P0 EXIT", "EXIT"], ["FADD R2,R2,R3"]):
            self.assertIsNone(instruction_stream(listing(lines), "full"))
        self.assertIsNone(instruction_stream(listing(["EXIT"]), "other"))

    def test_partial_waves_and_incomplete_evidence(self):
        def record(issue):
            return dict(physical_total=100, schedule=dict(resident_ctas=2, waves=[
                dict(issue=issue), dict(issue=issue+4)]), read_sectors=4, write_sectors=4,
                spill_sectors=0, ldg=(1, 0, 0), interior_reloads=0, conversions=0)
        a, b = record(100), record(97)
        self.assertEqual(grid_service(a, 229, 114), 204)
        rs = dict(incumbent=a, wide=b)
        self.assertEqual(rank_grid(rs, "incumbent", 114, 114)[0], "wide")
        b["schedule"]["waves"][0]["issue"] = 98
        self.assertEqual(rank_grid(rs, "incumbent", 114, 114)[0], "incumbent")
        b.pop("schedule")
        self.assertEqual(rank_grid(rs, "incumbent", 114, 114)[1]["status"], "incomplete-schedule")
        with self.assertRaises(ValueError):
            rank_grid(rs, "incumbent", 0, 114)


if __name__ == "__main__":
    unittest.main()
