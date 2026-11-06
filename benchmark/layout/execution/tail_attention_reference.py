"""Independent numerical contracts for the compact, strict SM90 Attention trial."""

import math

import numpy as np


D = 64
BLOCK = (128, 1, 1)
TILE = 64
SCORE_SCALE = np.float32(0.125)
LOG2_E = np.array(0x3FB8AA3B, dtype=np.uint32).view(np.float32)[()]
NAMES = ('random', 'uniform-sparse-value', 'last-key-dominant', 'key-boundary', 'neutral-extension')
EXACT = frozenset({'uniform-sparse-value', 'last-key-dominant'})
ERROR = dict(strict_atol=3e-5, strict_rtol=3e-4, fp64_atol=4e-5, fp64_rtol=4e-4)


def geometry(spec):
    if set(spec) != {'B', 'H', 'N', 'D', 'q_block', 'kv_block'} or any(type(v) is not int for v in spec.values()):
        raise ValueError('expected integer B/H/N/D and Q/KV tile sizes')
    b, h, n, d = (spec[key] for key in ('B', 'H', 'N', 'D'))
    if not (1 <= b <= 16 and 1 <= h <= 16 and n in (512, 513) and d == D
            and spec['q_block'] == spec['kv_block'] == TILE):
        raise ValueError('unregistered strict Attention shape')
    return (b, h, n, d), [b*h*math.ceil(n/TILE), 1, 1]


def _geometry_for_values(q, k, v):
    if (q.dtype != np.float16 or k.dtype != q.dtype or v.dtype != q.dtype
            or q.shape != k.shape or q.shape != v.shape or len(q.shape) != 4
            or q.shape[-1] != D or q.shape[-2] not in (512, 513)):
        raise ValueError('reference requires same-shaped FP16 compact BHSD arrays')


def ordered_sum(exponent):
    """Replay the four lane subtrees without relying on NumPy's sum order.

    Lane i owns columns 2i and 2i+1 modulo 8. The logical order within each
    lane is increasing column, also across successive KV tiles. Tree leaves
    beyond N are absent. Positive zero supplies the same numerical identity.
    """
    if exponent.dtype != np.float32 or exponent.ndim == 0 or exponent.shape[-1] == 0:
        raise ValueError('expected a nonempty FP32 exponent row')
    n = exponent.shape[-1]
    lanes = []
    for lane in range(4):
        columns = [i for base in range(0, n, 8)
                   for i in (base+2*lane, base+2*lane+1) if i < n]
        if not columns:
            lanes.append(np.zeros(exponent.shape[:-1], dtype=np.float32))
            continue
        width = 1 << (len(columns)-1).bit_length()
        subtree = np.zeros(exponent.shape[:-1]+(width,), dtype=np.float32)
        subtree[..., :len(columns)] = exponent[..., columns]
        while subtree.shape[-1] > 1:
            subtree = np.add(subtree[..., ::2], subtree[..., 1::2], dtype=np.float32)
        lanes.append(subtree[..., 0])
    left = np.add(lanes[0], lanes[1], dtype=np.float32)
    right = np.add(lanes[2], lanes[3], dtype=np.float32)
    return np.add(left, right, dtype=np.float32)[..., None]


def strict_reference(values):
    """CPU math mirror, using correctly rounded exp2 as a PTX approximation model.

    FP64 products isolate the precision of softmax from native accumulation
    order. The scalar model is used within tolerance except on registered
    fixtures whose exact QK/PV dot products and exponent bits are known.
    """
    q, k, v = values
    _geometry_for_values(q, k, v)
    scores = np.matmul(q.astype(np.float64), k.astype(np.float64).swapaxes(-1, -2)).astype(np.float32)
    scores = np.multiply(scores, SCORE_SCALE, dtype=np.float32)
    centered = np.subtract(scores, scores.max(axis=-1, keepdims=True), dtype=np.float32)
    with np.errstate(under='ignore'):
        exponent = np.exp2(np.multiply(centered, LOG2_E, dtype=np.float32), dtype=np.float32)
        probability = np.divide(exponent, ordered_sum(exponent), dtype=np.float32).astype(np.float16)
    return np.matmul(probability.astype(np.float64), v.astype(np.float64)).astype(np.float32)


def high_precision_reference(values):
    """Independent stable float64 softmax and dot products, with the FP16 P boundary."""
    q, k, v = values
    _geometry_for_values(q, k, v)
    scores = np.matmul(q.astype(np.float64), k.astype(np.float64).swapaxes(-1, -2)) * 0.125
    centered = scores - scores.max(axis=-1, keepdims=True)
    with np.errstate(under='ignore'):
        exponent = np.exp(centered)
        probability = (exponent / exponent.sum(axis=-1, keepdims=True)).astype(np.float16)
    return np.matmul(probability.astype(np.float64), v.astype(np.float64))


def _sparse_value(v):
    b, h, n, d = v.shape
    heads = np.arange(b*h, dtype=np.float16).reshape(b, h, 1) / np.float16(16)
    features = (np.arange(d, dtype=np.float16) % np.float16(8)) / np.float16(32)
    for value, column in enumerate((0, 63, 64, n-2, n-1), 1):
        v[..., column, :] = np.float16(value/8) + heads + features


def inputs(spec, name, seed=14261107):
    shape, _ = geometry(spec)
    b, h, n, d = shape
    rng = np.random.default_rng(seed)
    if name == 'random':
        return tuple(rng.uniform(-1, 1, shape).astype(np.float16) for _ in range(3))
    if name == 'neutral-extension':
        prefix = (b,h,512,d)
        q, k, v = [np.zeros(shape, dtype=np.float16) for _ in range(3)]
        q[..., :512, :] = rng.integers(-8, 9, prefix, dtype=np.int16) / 16
        k[..., :512, :] = rng.integers(-8, 9, prefix, dtype=np.int16) / 16
        v[..., :512, :] = rng.integers(1, 17, prefix, dtype=np.int16) / 16
        q[..., :, 0] = 1
        k[..., :512, 0] = 0
        if n == 513:
            q[..., -1, :] = q[..., 0, :]
            k[..., -1, 0] = -8192
            v[..., -1, :] = -32768
        return q, k, v
    q, k, v = [np.zeros(shape, dtype=np.float16) for _ in range(3)]
    _sparse_value(v)
    if name == 'uniform-sparse-value':
        return q, k, v
    if name == 'last-key-dominant':
        q[..., :, 0] = 1
        q[..., -1, 0] = -1
        k[..., :-1, 0] = -8192
        k[..., -1, 0] = 8192
    elif name == 'key-boundary':
        q[..., ::2, 0] = 1
        q[..., 1::2, 0] = -1
        k[..., 63, 0] = 12
        k[..., 64, 0] = -16
        k[..., -2, 0] = -20
        k[..., -1, 0] = 32
    else:
        raise ValueError('unknown strict Attention fixture')
    return q, k, v


def exact_reference(values, name):
    """An analytic oracle that deliberately bypasses the balanced-tree reference."""
    if name not in EXACT:
        raise ValueError('no analytical bitwise reference for this fixture')
    q, k, v = values
    _geometry_for_values(q, k, v)
    n = v.shape[-2]
    output = np.empty(v.shape, dtype=np.float32)
    if name == 'uniform-sparse-value':
        weight = np.float16(np.float32(1) / np.float32(n))
        output[...] = weight.astype(np.float64) * v.astype(np.float64).sum(axis=-2, keepdims=True)
    else:
        output[..., :-1, :] = v[..., -1:, :]
        weight = np.float16(np.float32(1) / np.float32(n-1))
        output[..., -1, :] = weight.astype(np.float64) * v[..., :-1, :].astype(np.float64).sum(axis=-2)
    return output


def check_output(observed, references, name):
    if observed.dtype != np.float32 or observed.shape != references['strict'].shape or not np.isfinite(observed).all():
        raise ValueError('output is nonfinite, incomplete, or outside compact BHSD geometry')
    out = observed.astype(np.float64)
    facts = {}
    for mode in ('strict', 'fp64'):
        ref = references[mode]
        error = np.abs(out - ref)
        bound = ERROR[mode+'_atol'] + ERROR[mode+'_rtol'] * np.abs(ref)
        facts[mode+'_max_abs'] = float(error.max())
        facts[mode+'_max_bound_ratio'] = float((error/bound).max())
        if not (error <= bound).all():
            index = np.unravel_index((error/bound).argmax(), out.shape)
            raise ValueError(f'{name}/{mode} reference mismatch at {index}: {out[index]} vs {ref[index]}')
    if name in EXACT:
        if not np.array_equal(observed.view(np.uint32), references['exact'].view(np.uint32)):
            raise ValueError(f'{name} analytical output differs bitwise')
        facts['analytical_bitwise'] = True
    return facts


def references(values, name):
    result = dict(strict=strict_reference(values), fp64=high_precision_reference(values))
    if name in EXACT:
        result['exact'] = exact_reference(values, name)
    return result
