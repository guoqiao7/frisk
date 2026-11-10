"""Independent references for strict compact D64 tiled Attention.

A query uses the complete row maximum and the canonical ordered FP32 sum.
Normalized probabilities are rounded to FP16 before the FP32 value product.
The reference checks this boundary independently of the GPU layout.
"""

import numpy as np


D = 64
BLOCK = (128, 1, 1)
TILE = 64
SCORE_SCALE = np.float32(0.125)
LOG2_E = np.array(0x3FB8AA3B, dtype=np.uint32).view(np.float32)[()]
NAMES = ('random', 'uniform-sparse-value', 'last-key-dominant',
         'tail-uniform', 'key-boundary', 'neutral-extension')
EXACT = frozenset(('uniform-sparse-value', 'last-key-dominant', 'tail-uniform'))
ERROR = dict(strict_atol=3e-5, strict_rtol=3e-4, fp64_atol=4e-5, fp64_rtol=4e-4)
_QUERY_CHUNK = 128
_INDEX_LIMIT = (1 << 31) - 1


def geometry(spec):
    keys = {'B', 'H', 'N', 'D', 'q_block', 'kv_block'}
    if type(spec) is not dict or set(spec) != keys or any(type(v) is not int for v in spec.values()):
        raise ValueError('expected integer B/H/N/D and Q/KV tile sizes')
    b, h, n, d = (spec[key] for key in ('B', 'H', 'N', 'D'))
    if (min(b, h) <= 0 or not TILE <= n <= 1025 or d != D or b * h * n * d > _INDEX_LIMIT
            or spec['q_block'] != TILE or spec['kv_block'] != TILE):
        raise ValueError('shape outside compact D64 Attention launch contract')
    return (b, h, n, d), [b * h * ((n + TILE - 1) // TILE), 1, 1]


def _values(values):
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise ValueError('expected Q, K, V')
    q, k, v = values
    if any(not isinstance(x, np.ndarray) or x.dtype != np.float16 or x.ndim != 4
           or x.shape != q.shape or not x.flags.c_contiguous for x in (q, k, v)):
        raise ValueError('reference requires same-shaped compact FP16 BHSD arrays')
    b, h, n, d = q.shape
    geometry(dict(B=b, H=h, N=n, D=d, q_block=TILE, kv_block=TILE))
    if not all(np.isfinite(x).all() for x in (q, k, v)):
        raise ValueError('registered strict Attention inputs are finite')
    return q, k, v


def ordered_sum(exponent):
    """Exact FP32 balanced tree, first within each modulo-8 lane subgroup."""
    if (not isinstance(exponent, np.ndarray) or exponent.dtype != np.float32
            or not exponent.ndim or exponent.shape[-1] == 0):
        raise ValueError('expected a nonempty FP32 exponent row')
    n = exponent.shape[-1]
    lanes = []
    for lane in range(4):
        columns = np.arange(2 * lane, n, 8)
        pairs = np.stack((columns, columns + 1), axis=-1).reshape(-1)
        pairs = pairs[pairs < n]
        if not pairs.size:
            lanes.append(np.zeros(exponent.shape[:-1], dtype=np.float32))
            continue
        width = 1 << (int(pairs.size) - 1).bit_length()
        subtree = np.zeros(exponent.shape[:-1] + (width,), dtype=np.float32)
        subtree[..., :pairs.size] = exponent[..., pairs]
        while subtree.shape[-1] > 1:
            subtree = np.add(subtree[..., ::2], subtree[..., 1::2], dtype=np.float32)
        lanes.append(subtree[..., 0])
    left = np.add(lanes[0], lanes[1], dtype=np.float32)
    right = np.add(lanes[2], lanes[3], dtype=np.float32)
    return np.add(left, right, dtype=np.float32)[..., None]


def _strict_probability(dot):
    score = np.multiply(dot.astype(np.float32), SCORE_SCALE, dtype=np.float32)
    centered = np.subtract(score, score.max(axis=-1, keepdims=True), dtype=np.float32)
    exponent = np.exp2(np.multiply(centered, LOG2_E, dtype=np.float32), dtype=np.float32)
    return np.divide(exponent, ordered_sum(exponent), dtype=np.float32).astype(np.float16)


def _fp64_probability(dot):
    score = dot * np.float64(SCORE_SCALE)
    centered = score - score.max(axis=-1, keepdims=True)
    exponent = np.exp(centered)
    return (exponent / exponent.sum(axis=-1, keepdims=True)).astype(np.float16)


def _oracle(values, modes, *, repeated_queries=False):
    q, k, v = _values(values)
    n = q.shape[-2]
    types = {'strict': np.float32, 'fp64': np.float64}
    outputs = {mode: np.empty(q.shape, dtype=types[mode]) for mode in modes}
    flat = {mode: result.reshape(-1, n, D) for mode, result in outputs.items()}
    kernels = {'strict': _strict_probability, 'fp64': _fp64_probability}
    for head, (queries, keys, vals) in enumerate(zip(q.reshape(-1, n, D), k.reshape(-1, n, D),
                                                    v.reshape(-1, n, D), strict=True)):
        if repeated_queries:
            queries, inverse = np.unique(queries, axis=0, return_inverse=True)
            targets = {mode: np.empty((len(queries), D), dtype=types[mode]) for mode in modes}
        else:
            targets = {mode: result[head] for mode, result in flat.items()}
        keys = keys.astype(np.float64).T
        vals = vals.astype(np.float64)
        with np.errstate(under='ignore'):
            for begin in range(0, len(queries), _QUERY_CHUNK):
                chunk = slice(begin, begin + _QUERY_CHUNK)
                dot = queries[chunk].astype(np.float64) @ keys
                for mode in modes:
                    probability = kernels[mode](dot)
                    targets[mode][chunk] = probability.astype(np.float64) @ vals
        if repeated_queries:
            for mode in modes:
                flat[mode][head] = targets[mode][inverse]
    return outputs


def strict_reference(values):
    """FP64 dots, strict FP32 softmax tree, and FP16 probabilities before PV."""
    return _oracle(values, ('strict',))['strict']


def high_precision_reference(values):
    """Independent FP64 dots and stable softmax, retaining the FP16 P boundary."""
    return _oracle(values, ('fp64',))['fp64']


def _tail_start(n):
    return ((n - 1) // TILE) * TILE


def _boundary_keys(n):
    return tuple(dict.fromkeys((0, min(TILE - 1, n - 1), min(TILE, n - 1),
                                max(0, _tail_start(n) - 1), _tail_start(n),
                                max(0, n - 2), n - 1)))


def _one_hot_values(v, keys):
    """One exact halfword product per feature, with head and feature identity."""
    b, h, _, _ = v.shape
    heads = (np.arange(b * h).reshape(b, h) % 512).astype(np.float32) / 512
    for feature in range(D):
        value = 0.125 + heads + (feature % 8) / 32
        v[..., keys[feature % len(keys)], feature] = value.astype(np.float16)


def neutral_prefix(n):
    """Length of the common non-neutral prefix for an aligned/tail comparison."""
    if type(n) is not int or n <= 0:
        raise ValueError('positive length required for neutral extension')
    return max(1, n // TILE * TILE)


def inputs(spec, name, seed=14261107):
    shape, _ = geometry(spec)
    b, h, n, d = shape
    rng = np.random.default_rng(seed)
    if name == 'random':
        # A float32 temporary per input avoids one full-size FP64 allocation.
        return tuple(np.subtract(rng.random(shape, dtype=np.float32) * 2, 1).astype(np.float16)
                     for _ in range(3))
    if name not in NAMES:
        raise ValueError('unknown strict Attention fixture')
    q, k, v = [np.zeros(shape, dtype=np.float16) for _ in range(3)]
    if name == 'neutral-extension':
        prefix = neutral_prefix(n)
        # For the sweep's N <= 1025, these logits keep live FP16 P on a
        # grid no finer than 2^-21. Products with V are multiples of 2^-25,
        # and their positive sum stays below 0.5, so PV is exact in FP32.
        templates = rng.integers(-4, 5, (b, h, 4, d), dtype=np.int16).astype(np.float32) / 32
        q[..., :prefix, :] = templates[..., np.arange(prefix) % 4, :]
        base_shape = (b, h, prefix, d)
        k[..., :prefix, :] = rng.integers(-4, 5, base_shape, dtype=np.int16).astype(np.float32) / 32
        v[..., :prefix, :] = rng.integers(1, 8, base_shape, dtype=np.int16).astype(np.float32) / 16
        q[..., :prefix, 0] = 1
        k[..., :prefix, 0] = 0
        q[..., prefix:, :] = q[..., :1, :]
        k[..., prefix:, 0] = -8192
        v[..., prefix:, :] = -32768
        return q, k, v
    if name == 'tail-uniform':
        start = _tail_start(n)
        _one_hot_values(v, tuple(range(start, n)))
        q[..., :, 0] = 1
        k[..., :start, 0] = -8192
        k[..., start:, 0] = 8192
        return q, k, v
    keys = _boundary_keys(n)
    _one_hot_values(v, keys)
    if name == 'last-key-dominant':
        q[..., :, 0] = 1
        if n > 1:
            q[..., -1, 0] = -1
            k[..., :-1, 0] = -8192
        k[..., -1, 0] = 8192
    elif name == 'key-boundary':
        q[..., ::2, 0] = 1
        q[..., 1::2, 0] = -1
        for key, logit in zip(keys, (12, -16, 8, -20, 24, -28, 32)):
            k[..., key, 0] = logit
    return q, k, v


def _validate_exact(q, k, v, name):
    n = v.shape[-2]
    if np.any(np.count_nonzero(v, axis=-2) > 1) or np.any((v == 0) & np.signbit(v)):
        raise ValueError('exact fixture requires canonical zeros and at most one nonzero value per output feature')
    if name == 'uniform-sparse-value':
        valid = not np.any(q) and not np.any(k)
    else:
        valid = not np.any(q[..., 1:]) and not np.any(k[..., 1:])
        if name == 'last-key-dominant':
            valid &= (np.all(q[..., :-1, 0] == 1) and np.all(q[..., -1, 0] == (-1 if n > 1 else 1))
                      and np.all(k[..., :-1, 0] == -8192) and np.all(k[..., -1, 0] == 8192))
        elif name == 'tail-uniform':
            start = _tail_start(n)
            valid &= (np.all(q[..., :, 0] == 1) and np.all(k[..., :start, 0] == -8192)
                      and np.all(k[..., start:, 0] == 8192) and not np.any(v[..., :start, :]))
        else:
            raise ValueError('no analytical bitwise reference for this fixture')
    if not valid:
        raise ValueError('inputs do not satisfy the registered exact fixture')


def _analytical(v, name, *, fp64):
    # Each feature has at most one nonzero V, so finite FP16 products are
    # exact in FP32 and summing zero products cannot change a nonzero value.
    n = v.shape[-2]
    dtype = np.float64 if fp64 else np.float32

    def coefficient(count):
        return np.float64(np.float16(dtype(1) / dtype(count)))

    result = np.empty(v.shape, dtype=dtype)
    if name == 'uniform-sparse-value':
        result[...] = coefficient(n) * v.sum(axis=-2, dtype=np.float64, keepdims=True)
    elif name == 'tail-uniform':
        result[...] = coefficient(n - _tail_start(n)) * v.sum(axis=-2, dtype=np.float64, keepdims=True)
    elif n == 1:
        result[...] = v
    else:
        result[..., :-1, :] = v[..., -1:, :]
        result[..., -1, :] = coefficient(n - 1) * v[..., :-1, :].sum(axis=-2, dtype=np.float64)
    return result


def exact_reference(values, name):
    if name not in EXACT:
        raise ValueError('no analytical bitwise reference for this fixture')
    q, k, v = _values(values)
    _validate_exact(q, k, v, name)
    return _analytical(v, name, fp64=False)


def references(values, name):
    if name not in NAMES:
        raise ValueError('unknown strict Attention fixture')
    if name in EXACT:
        q, k, v = _values(values)
        _validate_exact(q, k, v, name)
        strict = _analytical(v, name, fp64=False)
        return dict(strict=strict, fp64=_analytical(v, name, fp64=True), exact=strict)
    return _oracle(values, ('strict', 'fp64'), repeated_queries=name != 'random')


def check_output(observed, expected, name):
    if (not isinstance(observed, np.ndarray) or observed.dtype != np.float32
            or observed.shape != expected['strict'].shape or not np.isfinite(observed).all()):
        raise ValueError('output is nonfinite, incomplete, or outside compact BHSD geometry')
    facts = {}
    out = observed.astype(np.float64)
    for mode in ('strict', 'fp64'):
        ref = expected[mode]
        error = np.abs(out - ref)
        bound = ERROR[mode + '_atol'] + ERROR[mode + '_rtol'] * np.abs(ref)
        ratio = error / bound
        facts[mode + '_max_abs'] = float(error.max())
        facts[mode + '_max_bound_ratio'] = float(ratio.max())
        if facts[mode + '_max_bound_ratio'] > 1:
            index = np.unravel_index(ratio.argmax(), out.shape)
            raise ValueError(f'{name}/{mode} reference mismatch at {index}: {out[index]} vs {ref[index]}')
    if name in EXACT:
        if not np.array_equal(observed.view(np.uint32), expected['exact'].view(np.uint32)):
            raise ValueError(f'{name} analytical output differs bitwise')
        facts['analytical_bitwise'] = True
    return facts


def check_neutral_extension(base, tail, *, require_bitwise=True):
    """Check the shared prefix and each duplicated query of the neutral tail."""
    if (type(require_bitwise) is not bool or not isinstance(base, np.ndarray)
            or not isinstance(tail, np.ndarray) or base.dtype != np.float32
            or tail.dtype != np.float32 or base.ndim != 4 or tail.ndim != 4
            or base.shape[:2] != tail.shape[:2] or base.shape[-1] != D or tail.shape[-1] != D):
        raise ValueError('expected paired compact FP32 BHSD outputs')
    prefix, length = base.shape[-2], tail.shape[-2]
    if not 0 < prefix < length or prefix != neutral_prefix(length):
        raise ValueError('paired neutral prefix differs from the fixture')
    observed = tail[..., :prefix, :]
    duplicate = tail[..., prefix:, :]
    first = np.broadcast_to(tail[..., :1, :], duplicate.shape)
    changed = int(np.count_nonzero(observed.view(np.uint32) != base.view(np.uint32)))
    duplicate_changed = int(np.count_nonzero(duplicate.view(np.uint32) != first.view(np.uint32)))
    if require_bitwise:
        if changed:
            raise ValueError('zero-weight KV extension changed old output bits')
        if duplicate_changed:
            raise ValueError('a tail query differs from its duplicated query 0')
    if not require_bitwise:
        limits = dict(rtol=ERROR['strict_rtol'], atol=ERROR['strict_atol'])
        if not (np.allclose(observed, base, **limits) and np.allclose(duplicate, first, **limits)):
            raise ValueError('neutral extension or duplicated query exceeds tolerance')
    return dict(prefix=prefix, tail_length=length, changed_elements=changed,
                duplicated_query_changed_elements=duplicate_changed,
                bitwise=not (changed or duplicate_changed))
