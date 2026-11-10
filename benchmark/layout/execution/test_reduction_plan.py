"""Ordered expression differentials for the checked warp-reduction planner."""

import copy
import math
import unittest

from execution_support import SYMBOLIC_VERSION
from reduction_plan import plan_ordered_warp_reduction, verified_warp_reduction_trees


def distribution(shape, registers, logical):
    volume = math.prod(shape)
    first = [logical.index(x) for x in range(volume)]
    return dict(shape=shape, topology=[registers, 32, 4, 1, 1], logical=logical,
                first=first, replication=len(logical) // volume)


def compact(explicit):
    shape = explicit['shape']
    logical_widths = [n.bit_length() - 1 for n in shape]
    hardware_widths = [n.bit_length() - 1 for n in explicit['topology']]

    def pack(flat):
        coordinates = []
        for dim in reversed(shape):
            flat, coordinate = divmod(flat, dim)
            coordinates.append(coordinate)
        packed = shift = 0
        for x, width in zip(reversed(coordinates), logical_widths):
            packed |= x << shift
            shift += width
        return packed

    cols = [pack(explicit['logical'][1 << i]) for i in range(sum(hardware_widths))]
    rows = [sum(((col >> i) & 1) << bit for bit, col in enumerate(cols))
            for i in range(sum(logical_widths))]
    return dict(shape=shape, topology=explicit['topology'], replication=explicit['replication'],
                symbolic=dict(version=SYMBOLIC_VERSION, hardware_bits=sum(hardware_widths),
                              logical_bits=sum(logical_widths), hardware_widths=hardware_widths,
                              logical_widths=logical_widths, rows_hex=[format(r, 'x') for r in rows]))


def holder(index, registers):
    thread, register = divmod(index, registers)
    warp, lane = divmod(thread, 32)
    return [0, 0, warp, lane, register]


def certificate(source, dest, kind='sum', axis=1):
    """Construct the same canonical proof using just first holders and tree nodes."""
    extent = source['shape'][axis]
    suffix = math.prod(source['shape'][axis + 1:])
    volume = math.prod(dest['shape'])
    fibers = [dict(logicalOutput=y, nodes=[], root=None, broadcasts=[]) for y in range(volume)]
    rsrc, rdst = source['topology'][0], dest['topology'][0]
    for idx, x in enumerate(source['logical']):
        if source['first'][x] != idx:
            continue
        y = (x // (extent * suffix)) * suffix + x % suffix
        fibers[y]['nodes'].append(dict(logicalInput=x, holder=holder(idx, rsrc), left=-1, right=-1))
    for idx, y in enumerate(dest['logical']):
        fibers[y]['broadcasts'].append(holder(idx, rdst))
    scopes = set()
    for fiber in fibers:
        nodes = fiber['nodes']
        current = list(range(len(nodes)))
        while len(current) > 1:
            pairs = zip(current[::2], current[1::2])
            current = []
            for left, right in pairs:
                current.append(len(nodes))
                nodes.append(dict(logicalInput=-1, holder=nodes[left]['holder'], left=left, right=right))
        fiber['root'] = current[0]
        all_holders = [node['holder'] for node in nodes] + fiber['broadcasts']
        scope = 'cta_shared_tree' if len({h[2] for h in all_holders}) > 1 else (
            'warp' if len({h[3] for h in all_holders}) > 1 else 'register')
        scopes.add(scope)
    scope = next(s for s in ('cta_shared_tree', 'warp', 'register') if s in scopes)
    return dict(name='frisk.reduce_tensor', attrs=dict(kind=kind, dim=axis,
                **{'frisk.reduction_contract': dict(version=1, algorithm='canonical_fiber_tree_v1',
                   axis=axis, dtype='f32', kind=kind, target='sm_90a', scope=scope, threads=128,
                   input_policy='first_owner', output_policy='broadcast_complete',
                   source_shape=source['shape'], result_shape=dest['shape'])}), fibers=fibers)


def panel_fixture(size, locals_per_lane, *, skew=False, replicate=False):
    panels_per_warp = 32 // size
    rows = 4 * panels_per_warp
    columns = size * locals_per_lane // (2 if replicate else 1)
    result, logical = [], []
    for warp in range(4):
        for lane in range(32):
            if skew:
                row = warp * 8 + ((lane >> 3) << 1) + ((lane & 7).bit_count() % 2)
                position = (lane >> 1) & 3
            else:
                row = warp * panels_per_warp + lane // size
                position = lane % size
            if replicate:
                position %= size // 2
            result.append(row)
            logical.extend(row * columns + position * locals_per_lane + r for r in range(locals_per_lane))
    source = distribution([rows, columns], locals_per_lane, logical)
    dest = distribution([rows], 1, result)
    return certificate(source, dest), source, dest


class OrderedReductionTest(unittest.TestCase):
    def expression_differential(self, op, source, dest, proof, plan):
        def tree_value(tree, lookup):
            if tree[0] == 'leaf':
                return lookup(*tree[1:])
            return (tree_value(tree[1], lookup), tree_value(tree[2], lookup))

        expected = []
        for fiber in op['fibers']:
            states = []
            for node in fiber['nodes']:
                states.append(node['logicalInput'] if node['logicalInput'] >= 0
                              else (states[node['left']], states[node['right']]))
            expected.append(states[fiber['root']])
        sr, dr = source['topology'][0], dest['topology'][0]
        eager = proof.eager_registers()
        self.assertIsNotNone(eager)
        for warp in range(4):
            def logical(lane, register):
                return source['logical'][(warp * 32 + lane) * sr + register]

            for reg in range(dr):
                eager_values = [tree_value(eager[reg], lambda r, lanes: logical(lanes[lane], r))
                                for lane in range(32)]
                expected_values = [expected[dest['logical'][(warp * 32 + lane) * dr + reg]]
                                   for lane in range(32)]
                self.assertEqual(eager_values, expected_values)
                if plan is not None:
                    local = plan.registers[reg]
                    values = [tree_value(local.local_tree, lambda r: logical(lane, r)) for lane in range(32)]
                    for stage in local.stages:
                        values = [(values[lane], values[lane ^ stage.xor_mask])
                                  if (lane & stage.parity_mask).bit_count() % 2 == stage.left_parity
                                  else (values[lane ^ stage.xor_mask], values[lane]) for lane in range(32)]
                    self.assertEqual(values, expected_values)

    def check_panel(self, size, expected_masks, *, skew=False, use_compact=False, kind='sum'):
        op, source, dest = panel_fixture(size, 4, skew=skew)
        if kind != 'sum':
            op = certificate(source, dest, kind)
        proof = verified_warp_reduction_trees(op, compact(source) if use_compact else source,
                                               compact(dest) if use_compact else dest)
        plan = plan_ordered_warp_reduction(proof)
        self.assertIsNotNone(plan)
        self.assertEqual([stage.xor_mask for stage in plan.registers[0].stages], expected_masks)
        self.expression_differential(op, source, dest, proof, plan)

    def test_four_lane_sum(self):
        self.check_panel(4, [1, 2])

    def test_full_warp_max(self):
        self.check_panel(32, [1, 2, 4, 8, 16], kind='max')

    def test_noncoordinate_affine_cosets_compact(self):
        self.check_panel(4, [3, 5], skew=True, use_compact=True)

    def test_register_only(self):
        self.check_panel(1, [])

    def test_replicated_sources_use_exact_eager_fallback(self):
        op, source, dest = panel_fixture(4, 4, replicate=True)
        proof = verified_warp_reduction_trees(op, source, dest)
        self.assertIsNone(plan_ordered_warp_reduction(proof))
        self.expression_differential(op, source, dest, proof, None)

    def test_valid_cross_warp_reduction_is_outside_planner(self):
        source = distribution([1, 128], 1, list(range(128)))
        dest = distribution([1], 1, [0] * 128)
        proof = verified_warp_reduction_trees(certificate(source, dest), source, dest)
        self.assertEqual(proof.scope, 'cta_shared_tree')
        self.assertIsNone(proof.eager_registers())
        self.assertIsNone(plan_ordered_warp_reduction(proof))

    def test_noncanonical_certificate_fails_closed(self):
        op, source, dest = panel_fixture(4, 4)
        bad = copy.deepcopy(op)
        node = bad['fibers'][0]['nodes'][-1]
        node['left'], node['right'] = node['right'], node['left']
        with self.assertRaisesRegex(ValueError, 'noncanonical reduction node'):
            verified_warp_reduction_trees(bad, source, dest)
        bad = copy.deepcopy(op)
        bad['fibers'][0]['broadcasts'].pop()
        with self.assertRaisesRegex(ValueError, 'canonical reduction fiber'):
            verified_warp_reduction_trees(bad, source, dest)
        bad = copy.deepcopy(op)
        bad['attrs']['frisk.reduction_contract']['scope'] = 'register'
        with self.assertRaisesRegex(ValueError, 'scope differs'):
            verified_warp_reduction_trees(bad, source, dest)
        bad = copy.deepcopy(source)
        bad['first'][0] = False
        with self.assertRaisesRegex(ValueError, 'first-owner'):
            verified_warp_reduction_trees(op, bad, dest)

    def test_result_must_match_natural_projection(self):
        _, source, dest = panel_fixture(4, 4)
        wrong = distribution(dest['shape'], 1, [x ^ 1 for x in dest['logical']])
        with self.assertRaisesRegex(ValueError, 'natural projection'):
            verified_warp_reduction_trees(certificate(source, wrong), source, wrong)


if __name__ == '__main__':
    unittest.main()
