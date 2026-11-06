import struct
import unittest
from backend import splat_f32_bits

class SplatConstantBitsTest(unittest.TestCase):
    def test_decimal_bytes_are_unchanged(self):
        for value in ('0.000000e+00','2.000000e+00','-3.125','1.234567e-10','1.0e38'):
            expected=struct.unpack('<I',struct.pack('<f',float(value)))[0]
            self.assertEqual(splat_f32_bits('dense<'+value+'> : tensor<4xf32>'),expected)
    def test_tensor_layout_dense_matrix_is_not_the_constant_payload(self):
        encoding=' : tensor<64x64xf32, #layout<matrix = dense<\"0x800010\"> : tensor<12x12xi1>>>'
        self.assertEqual(splat_f32_bits('dense<2.500000e-01>'+encoding),0x3e800000)
        self.assertEqual(splat_f32_bits('dense<0x80000000>'+encoding),0x80000000)
        with self.assertRaisesRegex(ValueError,'splat'):splat_f32_bits('dense<[1.0,2.0]>'+encoding)
    def test_negative_zero_decimal_and_hex_preserve_sign(self):
        self.assertEqual(splat_f32_bits('dense<-0.000000e+00>'),0x80000000)
        self.assertEqual(splat_f32_bits('dense<0x80000000>'),0x80000000)
    def test_finite_hex_and_subnormal_bits_are_exact(self):
        for value in (0,1,0x3f800000,0x7f7fffff,0xff7fffff):
            self.assertEqual(splat_f32_bits('dense<0x'+format(value,'08X')+'>'),value)
    def test_infinity_decimal_and_hex(self):
        self.assertEqual(splat_f32_bits('dense<inf>'),0x7f800000)
        self.assertEqual(splat_f32_bits('dense<-inf>'),0xff800000)
        self.assertEqual(splat_f32_bits('dense<0x7F800000>'),0x7f800000)
    def test_nan_payload_and_sign_are_preserved(self):
        for bits in (0x7fc00000,0x7f800001,0xffc01234):
            self.assertEqual(splat_f32_bits('dense<0x'+format(bits,'08X')+'>'),bits)
        self.assertEqual(splat_f32_bits('dense<nan>'),struct.unpack('<I',struct.pack('<f',float('nan')))[0])
    def test_non_splat_and_malformed_literals_are_explicitly_rejected(self):
        for value in ('dense<[1.0,2.0]>','dense<[[1.0]]>','dense<0x123>','dense<0x100000000>','1.0','dense<foo>'):
            with self.assertRaisesRegex(ValueError,'f32'):splat_f32_bits(value)
if __name__=='__main__':unittest.main()
