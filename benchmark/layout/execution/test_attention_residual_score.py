import unittest

from attention_residual_score import emit_residual_score_attention, residual_score_holders
from attention_tile_mma import THREADS, build_d64_wgmma_tiles, fragment_coordinate


class ResidualScoreProofTests(unittest.TestCase):
    def test_every_residual_prefix_matches_independent_native_holders(self):
        columns = tuple(build_d64_wgmma_tiles(kv_tile=128).score_fragment_columns)
        for valid in range(65,128):
            masks = residual_score_holders(valid,columns)
            self.assertEqual(len(masks),64)
            self.assertEqual(sum(mask.bit_count() for mask in masks),64*valid)
            for register,mask in enumerate(masks):
                for thread in range(THREADS):
                    row,column=fragment_coordinate(thread,register,kv_tile=128)
                    self.assertEqual(bool(mask>>thread&1),column<valid)
                    self.assertLess(row,64)
            self.assertTrue(all(mask==(1<<THREADS)-1 for mask in masks[:32]))

    def test_invalid_domains_and_wrong_native_map_are_rejected(self):
        columns=tuple(build_d64_wgmma_tiles(kv_tile=128).score_fragment_columns)
        for valid in (True,0,1,63,64,128,129,65.0):
            with self.assertRaises(ValueError):
                residual_score_holders(valid,columns)
        wrong=list(columns)
        wrong[0],wrong[1]=wrong[1],wrong[0]
        with self.assertRaises(ValueError):
            residual_score_holders(127,tuple(wrong))
        with self.assertRaises(ValueError):
            residual_score_holders(127,columns[:-1])

    def test_nonresidual_shapes_fail_before_layout_packet_access(self):
        for length in (64,65,127,128,129,191,192,256,512,513):
            spec=dict(B=1,H=2,N=length,D=64,q_block=64,kv_block=64)
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    emit_residual_score_attention(spec,tile_packet=None)


if __name__=='__main__':
    unittest.main()

