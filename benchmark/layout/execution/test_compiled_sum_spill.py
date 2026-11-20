import unittest
from hashlib import sha256
from types import SimpleNamespace

from compiled_sum_spill import CompiledSumArm, LocalTraffic, local_traffic, selected_image


def listing(*instructions):
    return "Function : sum\n" + "\n".join(
        f" /*{i * 16:04x}*/ {op} ;" for i, op in enumerate(instructions)
    )


class CompiledSumSpillTest(unittest.TestCase):
    def test_full_width_qualified_and_predicated(self):
        code = listing("LDL.LU.64 R2, [R1]", "@P0 BRA 0x30",
                       "STL.128 [R1], R4", "EXIT", "BRA 0x40", "NOP")
        self.assertEqual(local_traffic(code, "sum"), LocalTraffic(2, 4, 1, 1))

    def test_local_access_in_repeat_is_unsupported(self):
        code = listing("LDL R4, [R1]", "@P0 BRA 0x0", "EXIT")
        with self.assertRaisesRegex(ValueError, "loop trip count"):
            local_traffic(code, "sum")

    def test_counter_above_four_hex_digits(self):
        code = listing(*(["NOP"] * 4096), "LDL R4, [R1]", "EXIT")
        self.assertEqual(local_traffic(code, "sum").load_words, 1)

    def test_incomplete_listing_is_rejected(self):
        code = listing("NOP", "LDL R4, [R1]", "EXIT").replace("/*0010*/", "/*0120*/")
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            local_traffic(code, "sum")

    def test_receipt_revalidates_binary(self):
        image = SimpleNamespace(asm={"cubin": b"first"})
        digest = sha256(b"first").hexdigest()
        arms = {"#R": CompiledSumArm(image, digest, LocalTraffic(0, 0, 0, 0), "")}
        receipt = f'compiled_spill_cubin = "{digest}"'
        self.assertIs(selected_image(arms, receipt), image)
        image.asm["cubin"] = b"other"
        with self.assertRaisesRegex(ValueError, "stale"):
            selected_image(arms, receipt)


if __name__ == "__main__":
    unittest.main()
