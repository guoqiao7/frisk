'''Checked GF(2) carrier cuts for reduction fan-out fusions.

The same bounded grammar supplies the producer and consumer choices to J and
LI. The producer's load, scale, and exponential share a carrier. Both
reduction axes use the natural projections of the selected consumer carrier.
'''
from dataclasses import dataclass


CUTS = ('AA', 'AB', 'BA', 'BB')
REDUCTION_POLICIES = ('auto', 'eager')
FUSION_COST_ID = 'sm90-fanout-issued-work-v2'


@dataclass(frozen=True)
class Carrier:
    widths: tuple
    registers: tuple
    lanes: tuple
    warps: tuple
    omitted: int | None = None

    def project(self, axis):
        return Carrier(self.widths,
                       tuple(bit for bit in self.registers if bit[0] != axis),
                       self.lanes, self.warps, axis)

    def attribute(self):
        blocks = (('register', self.registers), ('lane', self.lanes), ('warp', self.warps))
        inputs = ','.join(f'"{name}"' for name, bits in blocks if bits)
        input_bits = ','.join(str(len(bits)) for _, bits in blocks if bits)
        columns = tuple(bit for _, bits in blocks for bit in bits)
        dimensions = tuple((axis, width) for axis, width in enumerate(self.widths) if axis != self.omitted)
        outputs = ','.join(f'"dim{axis}"' for axis, _ in dimensions)
        output_bits = ','.join(str(width) for _, width in dimensions)
        logical = tuple((axis, bit) for axis, width in dimensions for bit in range(width))
        matrix = ','.join('[' + ','.join('1' if bit == column else '0' for column in columns) + ']'
                          for bit in logical)
        return (f'#frisk.distributed<map=#frisk.bit_linear<inputs=[{inputs}],input_bits=[{input_bits}],'
                f'outputs=[{outputs}],output_bits=[{output_bits}],matrix=dense<[{matrix}]> : '
                f'tensor<{len(logical)}x{len(columns)}xi1>>,'
                f'topology=[{1 << len(self.registers)},32,4,1,1],'
                f'replication={1 << (len(columns)-len(logical))}>')


def carriers(rows, cols):
    '''Adjacent coalesced layouts from one register/lane bit exchange.'''
    if (rows, cols) not in ((64, 4), (32, 8), (16, 16), (8, 32), (4, 64)):
        raise ValueError('expected a supported 256-element rectangular matrix')
    row = tuple((1, bit) for bit in range(rows.bit_length() - 1))
    col = tuple((2, bit) for bit in range(cols.bit_length() - 1))
    warps = ((0, 0), (0, 1))
    widths = (2, len(row), len(col))
    # A keeps the lowest two column bits in registers for aligned 16B loads.
    # B moves one of them to the warp lane and preserves the covered sectors.
    def carrier(column_registers):
        column_lanes = len(col) - column_registers
        row_lanes = 5 - column_lanes
        return Carrier(widths, col[:column_registers] + row[row_lanes:],
                       row[:row_lanes] + col[column_registers:], warps)
    return dict(A=carrier(2), B=carrier(1))


def source(rows, cols, producer, consumer, graph='dual-softmax'):
    '''Emit a complete checked graph with one possible cut on its shared value.'''
    if graph not in ('dual-softmax', 'softmax-column-rms', 'rms-sigmoid'):
        raise ValueError(f'unsupported fusion graph {graph}')
    choices = carriers(rows, cols)
    src, dst = choices[producer], choices[consumer]
    tile = f'4x{rows}x{cols}xf32'
    m = f'memref<{tile},1>'
    p = f'tensor<{tile},#producer>'
    c = f'tensor<{tile},#consumer>'
    row = f'tensor<4x{rows}xf32,#row>'
    col = f'tensor<4x{cols}xf32,#col>'
    storage = (f'#global = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1","dim2"],'
               f'input_extents=[4,{rows},{cols}],outputs=["byte_offset","bit_offset"],'
               f'output_extents=[{16*rows*cols},8],map=affine_map<(b,i,j)->(4*({rows*cols}*b+{cols}*i+j),0)>>,'
               'memory_space=#frisk<memory_space Global>,alignment=16,vector_granularity=1>')
    exchange = (f'      %transported = frisk.convert_layout %weights : {p} -> {c}\n'
                if producer != consumer else '')
    weights = '%transported' if exchange else '%weights'
    root_names = ('scores', 'unused0', 'unused1', 'output')
    args = ', '.join(f'%{name}:{m}' for name in root_names)
    roots = '\n'.join(f'    frisk.storage_contract %{name} {{layout=#global}} : {m}' for name in root_names)
    if graph == 'dual-softmax':
        consumers = f'''      %row_sum = frisk.reduce_tensor {weights} {{kind="sum",dim=2:i64}} : {c} -> {row}
      %row_empty = tensor.empty() : {c}
      %row_bcast = linalg.broadcast ins(%row_sum : {row}) outs(%row_empty : {c}) dimensions = [2]
      %row_p = arith.divf {weights}, %row_bcast : {c}
      %col_sum = frisk.reduce_tensor {weights} {{kind="sum",dim=1:i64}} : {c} -> {col}
      %col_empty = tensor.empty() : {c}
      %col_bcast = linalg.broadcast ins(%col_sum : {col}) outs(%col_empty : {c}) dimensions = [1]
      %col_p = arith.divf {weights}, %col_bcast : {c}
      %matches = arith.mulf %row_p, %col_p : {c}
'''
    elif graph == 'softmax-column-rms':
        consumers = f'''      %row_sum = frisk.reduce_tensor {weights} {{kind="sum",dim=2:i64}} : {c} -> {row}
      %row_empty = tensor.empty() : {c}
      %row_bcast = linalg.broadcast ins(%row_sum : {row}) outs(%row_empty : {c}) dimensions = [2]
      %row_p = arith.divf {weights}, %row_bcast : {c}
      %sq = arith.mulf {weights}, {weights} : {c}
      %col_sum = frisk.reduce_tensor %sq {{kind="sum",dim=1:i64}} : {c} -> {col}
      %col_factor = arith.constant dense<{1.0/rows:.8e}> : {col}
      %col_mean = arith.mulf %col_sum, %col_factor : {col}
      %col_eps = arith.constant dense<1.000000e-05> : {col}
      %col_regular = arith.addf %col_mean, %col_eps : {col}
      %col_inv = math.rsqrt %col_regular : {col}
      %col_empty = tensor.empty() : {c}
      %col_bcast = linalg.broadcast ins(%col_inv : {col}) outs(%col_empty : {c}) dimensions = [1]
      %matches = arith.mulf %row_p, %col_bcast : {c}
'''
    elif graph == 'rms-sigmoid':
        consumers = f'''      %sq = arith.mulf {weights}, {weights} : {c}
      %row_sum = frisk.reduce_tensor %sq {{kind="sum",dim=2:i64}} : {c} -> {row}
      %row_factor = arith.constant dense<{1.0/cols:.8e}> : {row}
      %row_mean = arith.mulf %row_sum, %row_factor : {row}
      %row_eps = arith.constant dense<1.000000e-05> : {row}
      %row_regular = arith.addf %row_mean, %row_eps : {row}
      %row_inv = math.rsqrt %row_regular : {row}
      %row_empty = tensor.empty() : {c}
      %row_bcast = linalg.broadcast ins(%row_inv : {row}) outs(%row_empty : {c}) dimensions = [2]
      %row_normalized = arith.mulf {weights}, %row_bcast : {c}
      %col_sum = frisk.reduce_tensor {weights} {{kind="sum",dim=1:i64}} : {c} -> {col}
      %col_neg_factor = arith.constant dense<{-1.0/rows:.8e}> : {col}
      %col_neg_mean = arith.mulf %col_sum, %col_neg_factor : {col}
      %col_exp = math.exp %col_neg_mean : {col}
      %col_one = arith.constant dense<1.0> : {col}
      %col_den = arith.addf %col_one, %col_exp : {col}
      %col_gate = arith.divf %col_one, %col_den : {col}
      %col_empty = tensor.empty() : {c}
      %col_bcast = linalg.broadcast ins(%col_gate : {col}) outs(%col_empty : {c}) dimensions = [1]
      %matches = arith.mulf %row_normalized, %col_bcast : {c}
'''
    exponent = f'      %weights = math.exp %scaled : {p}\n' if graph != 'rms-sigmoid' else ''
    if not exponent:
        weights = '%transported' if exchange else '%scaled'
        exchange = exchange.replace('%weights', '%scaled')
        consumers = consumers.replace('%weights', weights)
    kernel = graph.replace('-', '_') if graph != 'dual-softmax' else 'dual_softmax'
    return f'''{storage}
#producer = {src.attribute()}
#consumer = {dst.attribute()}
#row = {dst.project(2).attribute()}
#col = {dst.project(1).attribute()}
module attributes {{frisk.target = "sm_90a"}} {{
  "frisk.kernel"() <{{sym_name="{kernel}", function_type=({','.join([m]*4)})->()}}> ({{
  ^bb0({args}):
{roots}
    "frisk.parallel"() <{{ranges=array<i64:1>,threads=128:i64}}> ({{
    ^bb0(%i:index):
      %iv = frisk.layout_view %scores : {m} -> {m}
      %in = frisk.tile_load %iv : {m} -> {p}
      %temperature = arith.constant dense<0.25> : {p}
      %scaled = arith.mulf %in, %temperature : {p}
{exponent}{exchange}{consumers}      %ov = frisk.layout_view %output : {m} -> {m}
      frisk.tile_store %matches, %ov : {c}, {m}
      frisk.end
    }}) : () -> ()
    frisk.end
  }}) : () -> ()
}}
'''


def checked_pair_score(opcodes, features, trace):
    """Price the work that the checked SM90 emitter actually issued.

    Site counts already include repeated copies of a logical reduction or
    expensive scalar operation. Reduction arithmetic and cross-lane exchange
    receive an extra issue unit, while division and special-function sites
    receive separate weights. These are ordering units, not predicted cycles.
    """
    reductions = []
    for item in trace:
        if item['operation'] == 'frisk.reduce_tensor':
            axis, combines, shuffles = (item[field] for field in (
                'reduction_axis', 'reduction_combines', 'shuffles'))
            if any(type(value) is not int or value < 0 for value in (axis, combines, shuffles)):
                raise ValueError('malformed checked reduction work')
            reductions.append(dict(axis=axis, combines=combines, shuffles=shuffles))
    arithmetic = sum(row['combines'] for row in reductions)
    communication = sum(row['shuffles'] for row in reductions)
    if (not reductions or arithmetic > opcodes.get('add.rn.f32', 0) + opcodes.get('max.f32', 0)
            or communication != features['reduction_shuffles']):
        raise ValueError('reduction work does not match the checked emission')
    div = opcodes.get('div.rn.f32', 0)
    exp = opcodes.get('ex2.approx.f32', 0)
    rsqrt = opcodes.get('rsqrt.approx.f32', 0)
    site_count = sum(opcodes.values())
    weighted = site_count + arithmetic + communication + 12 * div + 3 * (exp + rsqrt)
    sectors = features['global_sector_issues32_upper']
    access_sites = features['global_load_sites'] + features['global_store_sites']
    # A tied issue proxy does not prove that the exact-division dependency
    # and its native expansion will be hidden. Prefer fewer division sites
    # before comparing Global sectors or access issue counts.
    return ((weighted, div, sectors, access_sites),
            dict(model=FUSION_COST_ID, issue_units=weighted, ptx_sites=site_count,
                 reduction_combines=arithmetic, reduction_shuffles=communication,
                 by_reduction=reductions, fp32_div=div, exp2=exp, rsqrt=rsqrt))


def select(previews):
    """Choose K4 queues from one complete, checked preview inventory.

    LI sees the whole producer stage and optimizes the remaining consumer
    completion after pinning it. Both strategies use the same static cost
    for complete candidates. J spends its four slots on layout cuts, while
    LI spends the same four on the remaining cuts and reduction policy.
    """
    if set(previews) != {(cut, policy) for cut in CUTS for policy in REDUCTION_POLICIES}:
        raise ValueError('exact common fan-out preview inventory required')
    local_scores = {}
    for carrier in 'AB':
        local_score = previews[carrier+carrier, 'auto']['local_score']
        if any(preview['local_score'] != local_score
               for (cut, _), preview in previews.items() if cut[0] == carrier):
            raise ValueError('consumer choice changed the own-producer score')
        local_scores[carrier] = local_score
    if local_scores['A'] == local_scores['B']:
        raise ValueError('this ablation requires a unique informed local choice')
    local = min(local_scores, key=local_scores.get)
    groups = {'J': [(cut, 'auto') for cut in CUTS],
              'LI': [(cut, policy) for cut in CUTS if cut[0] == local
                     for policy in REDUCTION_POLICIES]}
    fixed = {mode: min(range(len(queue)), key=lambda slot:
                      (previews[queue[slot]]['full_score'], queue[slot]))
             for mode, queue in groups.items()}
    return groups, fixed, local_scores, local
