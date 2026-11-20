"""Independent full-domain network, fallback and exact grammar tests; CPU only."""
import copy
import itertools
import random
import unittest
from shuffle_network import (IDENTITY_LANE, instruction_counts, plan_shuffle_network,
                             shuffle_network_candidates, simulate_network)


def _linear(columns, value):
    result = 0
    for bit, column in enumerate(columns):
        if value >> bit & 1:
            result ^= column
    return result


def owner_plan(register_bits, columns, base=0, warp_columns=(0, 0)):
    """Direct oracle owner function; source slots are unique holder IDs."""
    registers = 1 << register_bits
    dimension = register_bits + 5
    entries = []
    for thread in range(128):
        warp, lane = divmod(thread, 32)
        offset = base ^ _linear(warp_columns, warp)
        for register in range(registers):
            source = _linear(columns, register | (lane << register_bits)) ^ offset
            entries.append(dict(slot=warp * 32 * registers + source,
                                source=dict(reg=source % registers, lane=source // registers,
                                            warp=warp, warp_group=0)))
    return dict(mode='Shuffle', threads=128, warps=4, source_registers=registers,
                destination_registers=registers, volume=128 * registers,
                scratch_bytes=0, payload=dict(bits=32, words=[dict(bit_offset=0, live_bits=32)]),
                destinations=entries)


def expected_holders(plan):
    registers = plan['source_registers']
    return [(e['source']['warp'] * 32 + e['source']['lane']) * registers + e['source']['reg']
            for e in plan['destinations']]


def old_grammar(plan):
    """Direct per-destination accounting independent of affine rank shortcut."""
    registers = plan['source_registers']
    work = shfl = 0
    def mapped(table):
        return 1 + 4 * sum(table[1 << bit] != table[0] for bit in range(7))
    for register in range(registers):
        owners = [plan['destinations'][thread * registers + register]['source'] for thread in range(128)]
        source_registers = [o['reg'] for o in owners]
        lanes = [o['lane'] for o in owners]
        support = len(set(source_registers))
        shfl += support
        work += mapped(source_registers) + mapped(lanes) + support + 1 + 2 * (support - 1)
    return work, shfl


def random_matrix(rng, dimension):
    rows = [1 << i for i in range(dimension)]
    for _ in range(4 * dimension):
        i, j = rng.sample(range(dimension), 2)
        if rng.randrange(3):
            rows[i] ^= rows[j]
        else:
            rows[i], rows[j] = rows[j], rows[i]
    return [sum(((rows[i] >> j) & 1) << i for i in range(dimension)) for j in range(dimension)]


class ShuffleNetworkTest(unittest.TestCase):
    def check_plan(self, plan):
        alternatives = shuffle_network_candidates(plan)
        self.assertTrue(alternatives)
        expected = expected_holders(plan)
        legacy_work, legacy_shfl = old_grammar(plan)
        for candidate in alternatives:
            self.assertEqual(simulate_network(candidate), expected)
            self.assertEqual(candidate['legacy_work'], legacy_work)
            self.assertEqual(candidate['legacy_shfl'], legacy_shfl)
            self.assertEqual(candidate['counts'], instruction_counts(candidate['steps'], candidate['registers']))
        eligible = [p for p in alternatives if p['counts']['total_work'] <= legacy_work and p['counts']['shfl'] <= legacy_shfl]
        chosen = plan_shuffle_network(plan)
        if eligible:
            self.assertLessEqual(chosen['counts']['shfl'],legacy_shfl)
            self.assertEqual(chosen, min(eligible, key=lambda p:(p['counts']['shfl'], p['counts']['total_work'], p['kind'])))
        else:
            self.assertIsNone(chosen)
        return alternatives

    def test_exact_five_bit_swap_counts_all65536_holders(self):
        register_bits = 9
        columns = [1 << i for i in range(register_bits + 5)]
        for i in range(5):
            columns[i], columns[register_bits + i] = columns[register_bits + i], columns[i]
        p = owner_plan(register_bits, columns)
        alternatives = {n['kind']:n for n in self.check_plan(p)}
        fused = alternatives['bit-permutation-fused']; ship = alternatives['bit-permutation-ship']
        self.assertEqual((fused['counts']['shfl'], fused['counts']['selects']), (496, 5120))
        self.assertEqual((ship['counts']['shfl'], ship['counts']['selects']), (1280, 3840))
        self.assertEqual(fused['legacy_shfl'], 16384)
        self.assertEqual(plan_shuffle_network(p)['kind'], 'bit-permutation-fused')

    def test_random_bit_permutations_with_affine_and_warp_offsets(self):
        rng = random.Random(2026110401)
        for case in range(80):
            bits = 1 + case % 6; dimension = bits + 5
            columns = [1 << i for i in range(dimension)];rng.shuffle(columns)
            plan = owner_plan(bits, columns, rng.randrange(1 << dimension),
                              (rng.randrange(1 << dimension), rng.randrange(1 << dimension)))
            with self.subTest(case=case):
                self.check_plan(plan)

    def test_dense_invertible_gf2_shears_full_domains(self):
        rng = random.Random(2026110402)
        dense = 0
        for case in range(100):
            bits = 1 + case % 4; dimension = bits + 5
            columns = random_matrix(rng, dimension)
            plan = owner_plan(bits, columns, rng.randrange(1 << dimension),
                              (rng.randrange(1 << dimension), rng.randrange(1 << dimension)))
            with self.subTest(case=case):
                candidates = self.check_plan(plan)
                if any(c & (c-1) for c in columns):
                    self.assertEqual([c['kind'] for c in candidates], ['linear-shear']);dense += 1
        self.assertGreater(dense, 90)

    def test_lane_only_permutation_all120_and_zero_register_bits(self):
        for positions in itertools.permutations(range(5)):
            self.check_plan(owner_plan(0, [1 << i for i in positions]))

    def test_arbitrary_word_payload_preserved_including_ieee_patterns(self):
        bits = 4;dimension = bits + 5
        columns = [1 << i for i in range(dimension)]
        for i in range(4):columns[i], columns[bits+i] = columns[bits+i], columns[i]
        p = owner_plan(bits, columns, base=17, warp_columns=(9, 65))
        special = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x7fa00001, 0xffffffff]
        values = [special[i % len(special)] if i % 3 else (i * 2654435761) & 0xffffffff
                  for i in range(128 * (1 << bits))]
        expected = [values[i] for i in expected_holders(p)]
        for candidate in shuffle_network_candidates(p):
            self.assertEqual(simulate_network(candidate, values), expected)

    def test_non_affine_replication_crosswarp_and_malformed_fallback(self):
        columns = [1 << i for i in range(8)]
        columns[0], columns[3] = columns[3], columns[0]
        p = owner_plan(3, columns)
        mutations = []
        q = copy.deepcopy(p);q['destinations'][100]['source']['reg'] ^= 1;mutations.append(q)
        q = copy.deepcopy(p);q['destinations'][0]['source']['warp'] = 1;mutations.append(q)
        q = copy.deepcopy(p);q['destination_registers'] = 4;mutations.append(q)
        q = copy.deepcopy(p);q['source_registers'] = 3;mutations.append(q)
        q = copy.deepcopy(p);q['payload']['bits'] = 16;mutations.append(q)
        q = copy.deepcopy(p);q['destinations'].pop();mutations.append(q)
        q = copy.deepcopy(p);q['destinations'][0]['source']['reg'] = True;mutations.append(q)
        q = copy.deepcopy(p);q['destinations'][0]['source'].pop('lane');mutations.append(q)
        q = copy.deepcopy(p);q['mode'] = 'Register';mutations.append(q)
        singular = columns.copy();singular[1] = singular[0];mutations.append(owner_plan(3, singular))
        for index, q in enumerate(mutations):
            with self.subTest(index=index):
                self.assertEqual(shuffle_network_candidates(q), [])
                self.assertIsNone(plan_shuffle_network(q))
        for q in [None, [], {}, True]:self.assertIsNone(plan_shuffle_network(q))

    def test_pullback_order_is_not_reversed(self):
        # Noncommuting elementary shears expose reversed elimination/emission.
        columns = [1 << i for i in range(7)]
        columns[0] ^= 1 << 3;columns[2] ^= columns[0]
        p = owner_plan(2, columns, base=11, warp_columns=(3, 33))
        candidate = shuffle_network_candidates(p)[0]
        reversed_candidate = dict(candidate, steps=list(reversed(candidate['steps'])))
        self.assertEqual(simulate_network(candidate), expected_holders(p))
        self.assertNotEqual(simulate_network(reversed_candidate), expected_holders(p))

    def test_work_guard_can_preserve_exact_support_fallback(self):
        rng = random.Random(2026110403)
        found = False
        for _ in range(80):
            p = owner_plan(1, random_matrix(rng, 6))
            alternatives = shuffle_network_candidates(p)
            if all(n['counts']['total_work'] > n['legacy_work'] for n in alternatives):
                self.assertIsNone(plan_shuffle_network(p));found = True;break
        self.assertTrue(found)

    def test_smaller_total_work_cannot_increase_shuffle_communication(self):
        plan=owner_plan(3,[1,38,226,7,21,249,120,225])
        work,shfl=old_grammar(plan)
        self.assertEqual((work,shfl),(456,64))
        candidates=shuffle_network_candidates(plan)
        self.assertTrue(any(n['counts']['total_work']<work and n['counts']['shfl']>shfl for n in candidates))
        chosen=plan_shuffle_network(plan)
        if chosen is not None:
            self.assertLessEqual(chosen['counts']['total_work'],work)
            self.assertLessEqual(chosen['counts']['shfl'],shfl)

    def test_predicate_cache_and_bfly_identity_grammar(self):
        steps = [dict(op='reg_select', register_bit=0, thread_mask=1, constant=0),
                 dict(op='reg_select', register_bit=1, thread_mask=1, constant=0),
                 dict(op='mixed_swap', register_bit=0, lane_bit=0),
                 dict(op='reg_select', register_bit=0, thread_mask=3, constant=1),
                 dict(op='lane', columns=IDENTITY_LANE, offsets=[0]*4, register_masks=[0, 1, 0, 1])]
        counts = instruction_counts(steps, 4)
        self.assertEqual(counts['map_instructions'], 6)
        self.assertEqual(counts['shfl'], 4)
        self.assertEqual(counts['selects'], 18)

if __name__ == '__main__':
    unittest.main()
