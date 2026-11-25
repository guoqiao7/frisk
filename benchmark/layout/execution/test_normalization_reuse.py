import itertools
import unittest
from normalization_reuse import NormalizationReuse, miss_fractions
from normalization_schedule import rank_grid


class NormalizationReuseTest(unittest.TestCase):
    def test_distinct_input_output_cycles(self):
        for nx, ny in itertools.product((1, 4), repeat=2):
            launches = [((100*i, 32), (2000+200*j, 64))
                        for i, j in [(k % nx, k % ny) for k in range(max(nx,ny))]]
            for k in range(len(launches)):
                reuse = NormalizationReuse.from_launches(launches, k, cyclic=True)
                self.assertEqual(reuse.input_distance_bytes, nx*32+min(nx,ny)*64)
                self.assertEqual(reuse.output_distance_bytes, ny*64+min(nx,ny)*32)

    def test_history_alias_and_first_touch(self):
        launch = ((128, 32), (512, 64))
        self.assertEqual(NormalizationReuse.from_launches([launch], 0), NormalizationReuse())
        self.assertEqual(NormalizationReuse.from_launches([launch], 0, cold_start=True),
                         NormalizationReuse("cold","cold"))
        # A buffer written in a previous launch can now feed an input view.
        subview = ((528, 16), (768, 32))
        self.assertEqual(NormalizationReuse.from_launches([launch,subview], 1, cold_start=True),
                         NormalizationReuse(96,"cold"))
        overlapping = ((144, 32), (512,64))
        self.assertEqual(NormalizationReuse.from_launches([launch,overlapping], 1, cold_start=True),
                         NormalizationReuse(None,96))
        self.assertEqual(NormalizationReuse.from_launches([((1,16),(8,16))], 0, cyclic=True),
                         NormalizationReuse())

    def test_validation_and_capacity_curve(self):
        for bad in (0, True, -3, "hot", 2.5):
            with self.assertRaises(ValueError):
                NormalizationReuse(bad, None)
        with self.assertRaises(ValueError):
            NormalizationReuse.from_launches([((0,1),(8,1))], 0, cyclic=True, cold_start=True)
        curve=[miss_fractions(i,100)[0] for i in range(1,120)]
        self.assertEqual(curve,sorted(curve))
        self.assertEqual(miss_fractions(None,100),(0.,1.))

    def test_unknown_operand_evaluates_corners_and_keeps_legacy_choice_on_disagreement(self):
        def record(issue, instructions):
            wave=dict(issue=issue,global_wait=0,local_wait=0,shared_wait=0)
            return dict(physical_total=1, schedule=dict(resident_ctas=1,waves=[wave]),
                        read_sectors=32,write_sectors=32,spill_sectors=0,ldg=[instructions],
                        stg=[1],interior_reloads=0,conversions=0)
        records={"a":record(90,20),"b":record(100,1)}
        choose=lambda reuse:rank_grid(records,"a",8,8,reuse=reuse,l2_bytes=100)
        w,r=choose(NormalizationReuse(None,1))
        self.assertEqual((w,r["reuse"]["status"]),("a","reuse-sensitive-grid-fallback"))
        self.assertEqual(len(r["reuse"]["scenarios"]),2)
        w,r=choose(NormalizationReuse("cold",None))
        self.assertEqual((w,r["reuse"]["status"]),("b","stable-across-unknown-reuse"))
        self.assertEqual(choose(NormalizationReuse(1,1))[0],"a")
        self.assertEqual(choose(NormalizationReuse("cold",1))[0],"b")
        with self.assertRaises(ValueError):
            rank_grid(records,"a",8,8,l2_bytes=10)


if __name__=="__main__":
    unittest.main()
