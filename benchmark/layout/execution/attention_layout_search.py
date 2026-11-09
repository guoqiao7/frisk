"""Checked D64 Attention layout and schedule search, with one paid K4 queue per mode."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from time import perf_counter_ns

from attention_copy_layout import attention_copy_layout_evidence, derive_attention_copy_layouts
from attention_tile_mma import TILE, score_width_for_prefix
from tiled_attention import checked_geometry, emit_tiled_attention


K4 = 4
MODES = ('J', 'LI', 'legacy')
CONTRACT_FIELDS = (
    'contract', 'grid', 'block', 'entry', 'alignment_bytes', 'head_bytes', 'root_spans',
    'argument_roles', 'argument_dtypes', 'dynamic_shared_bytes', 'scratch_bytes',
    'passes', 'reduction_order', 'p_rounding', 'ftz', 'fp_fusion', 'mask',
    'fragment_gf2_columns',
)
LOOP = re.compile(r'^(row_(?:max|sum|value)_loop):$')
WGMMA = re.compile(r'^wgmma\.mma_async\.sync\.aligned\.m64n(\d+)k16\.')
LOOP_COUNT = re.compile(r'^setp\.lt\.u32 [^,]+, [^,]+, (\d+);$')


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return sha256(encoded).hexdigest()


@dataclass
class Preview:
    layout: str
    schedule: str
    tuning: dict
    ttgir: str
    proof: dict
    features: dict


def schedules(n):
    """The frozen legacy queue and the common J/LI policy archive."""
    anchor = dict(kv_width=64, vector_store=True)
    extra = {'kv64-factored': dict(anchor, factored_copy=True)}
    if n % 64:
        wide = dict(anchor, kv_width=128, retain_tail_scores=True)
        thin = dict(anchor, retain_tail_scores=True, narrow_tail=True, simplify_tail_copy=True,
                    factored_copy=True)
        tail = (dict(wide, retain_residual_scores=True, simplify_tail_copy=True, factored_copy=True)
                if n % 128 > 64 else dict(wide, narrow_tail=True, simplify_tail_copy=True,
                                         factored_copy=True))
        options = {'kv64': anchor, 'kv128-tail': tail, 'kv64-tail': thin,
                   'kv128-streaming-bulk': dict(tail, streaming_bulk_values=True)}
        if n >= 128:
            extra['kv128-no-retention'] = dict(anchor, kv_width=128, factored_copy=True)
    else:
        wide = dict(anchor, kv_width=128, factored_copy=True)
        compact = dict(anchor, kv_width=32, factored_copy=True)
        if n in (128, 256):
            cached = dict(anchor, kv_width=n, retain_all_scores=True, factored_copy=True)
            options = {'kv64': anchor, 'kv128-factored': wide, 'kv32-factored': compact,
                       'score-cache-bulk': dict(cached, bulk_value_publish=True)}
            extra['score-cache-staged'] = cached
            extra['kv128-streaming-bulk'] = dict(wide, retain_tail_scores=True,
                                                 streaming_bulk_values=True)
        else:
            options = {'kv64': anchor, 'kv128-factored': wide, 'kv32-factored': compact,
                       'kv128-streaming-bulk': dict(wide, retain_tail_scores=True,
                                                    streaming_bulk_values=True)}
        if n == 64:
            # Retain a real K4 even where KV128 is outside the shape domain.
            cached = dict(anchor, retain_all_scores=True, factored_copy=True)
            extra.update({'score-cache-staged': cached,
                          'score-cache-bulk': dict(cached, bulk_value_publish=True)})
    return options, dict(options, **extra)


def _emitted_work(text):
    """Weight loop bodies by their checked trip counts.

    Predicated instructions and the branches inside a loop are counted as an
    upper bound on issue work. The score is a label-free proxy, in work units.
    """
    counts = Counter()
    body = None
    for line in text.split(r'\0A'):
        if match := LOOP.fullmatch(line):
            if body is not None:
                raise ValueError('nested Attention cost loop')
            loop, body = match[1], Counter()
        if body is not None and line.endswith(f' bra.uni {loop};'):
            if (trip := LOOP_COUNT.fullmatch(previous)) is None or not int(trip[1]):
                raise ValueError('unknown checked Attention loop trip count')
            counts.update({op: int(trip[1]) * count for op, count in body.items()})
            body = None
            continue
        previous = line
        dest = counts if body is None else body
        op = line.split(' ', 1)[-1] if line.startswith('@') else line
        if match := WGMMA.match(op):
            dest['mma_n8_units'] += int(match[1]) // 8
            dest['mma_issue'] += 1
        elif op.startswith(('bar.sync ', 'wgmma.wait_group.', 'fence.proxy.')):
            dest['sync'] += 1
        elif op.startswith('ld.global.v4.b32 '):
            dest['copy_loads'] += 1
        elif op.startswith('st.shared.v4.b32 '):
            dest['copy_stores'] += 1
        elif op.startswith(('div.rn.f32 ', 'ex2.', 'shfl.')):
            dest['scalar_special'] += 1
        elif op.startswith(('st.global.', 'add.', 'xor.', 'setp.', 'and.', 'cvt.', 'mov.')):
            dest['scalar_issue'] += 1
    if body is not None or not counts['mma_issue'] or not counts['copy_loads']:
        raise ValueError('incomplete checked Attention emission for costing')
    return dict(counts)


def _features(text, proof, copies):
    work = Counter(_emitted_work(text))
    full = copies[TILE]
    tail = copies.get(proof['final_rows'])
    # Static sectors are averaged across the two valid 16B alignment phases.
    # The body counts already include guarded issue sites, so only excess
    # active warps for a narrow tail relative to an ideal packed copy are
    # charged here. The scope does not model dynamic memory overlap.
    read = sum(full['global_sectors32']) / 8 * work['copy_loads']
    publication = (16 + full['shared_excess_wavefronts'] / 4) * work['copy_stores']
    tail_issue = 0
    if tail and proof.get('narrow_tail'):
        load_warps = sum(group['load_warps'] for group in tail['groups'])
        ideal_warps = (proof['final_rows'] * 8 + 31) // 32
        tail_issue = max(0, load_warps - ideal_warps) + tail['shared_excess_wavefronts']
    cache = max(proof.get('score_cache_slots', 0), proof.get('residual_score_cache_slots', 0),
                len(proof.get('retain_tail_score_slots', ())))
    width = proof['execution_kv_width']
    registers = 72 + max(cache, min(width, 128) // 2)
    registers = (registers + 7) // 8 * 8
    shared = proof['static_shared_bytes']
    ctas_per_sm = min(16, 65536 // (128 * registers), 228 * 1024 // (shared + 1024))
    if not ctas_per_sm:
        raise ValueError('static Attention plan exceeds an SM90 resource limit')
    resident = proof['grid'][0] / 114 / ctas_per_sm
    pressure = 1 + max(0, resident - 0.75) * 0.10
    cost = pressure * (work['mma_n8_units'] / 2 + 2 * work['mma_issue'] + 4 * work['sync']
                       + read / 8 + publication / 8 + tail_issue + work['scalar_special'] / 2
                       + work['scalar_issue'] / 16)
    return dict(tracked_dynamic_issue_upper_bound=dict(work), copy_full=full, copy_tail=tail,
                estimated_registers=registers, estimated_ctas_per_sm=ctas_per_sm, tail_issue_penalty=tail_issue,
                static_shared_bytes=shared, cost=cost)


def select_attention(spec, tile_packet, *, problem_sha256):
    """Score one checked prepared graph. LI locks a producer-local layout.

    J and LI use the same emitted pair costs, the same physical contract and
    the same K4 projection. Legacy uses the previous fixed-layout schedule
    order. All three receive the common bounded preview archive.
    """
    if not re.fullmatch('[0-9a-f]{64}', problem_sha256):
        raise ValueError('a checked prepared Attention source digest is required')
    geometry = checked_geometry(spec)
    start = perf_counter_ns()
    legacy_options, common_options = schedules(spec['N'])
    layouts = derive_attention_copy_layouts(tile_packet)
    owners = {name: 'execution' for name in layouts}
    if 'coalesced' not in layouts or len(layouts) < 2 or 'legacy' in layouts:
        raise ValueError('checked Attention archive needs named GF(2) layout alternatives')
    layouts = dict(layouts, legacy=tile_packet)
    owners['legacy'] = 'gf2'
    tail_rows = geometry['final_rows']
    copy_features = {}
    for name, packet in layouts.items():
        # The legacy GF(2) optimizer's exact ownership is materialized by
        # the coalesced checked replay.
        explicit = layouts['coalesced'] if name == 'legacy' else packet
        evidence = {TILE: attention_copy_layout_evidence(explicit)}
        if tail_rows:
            evidence[tail_rows] = attention_copy_layout_evidence(explicit, valid_rows=tail_rows,
                                      storage_rows=score_width_for_prefix(tail_rows))
        copy_features[name] = evidence

    def local_copy_work(copy):
        groups = copy['groups']
        return (sum(copy['global_sectors32']) / 16 + sum(g['load_warps'] for g in groups)
                + sum(bool(g['initialized_vectors']) for g in groups)
                + copy['shared_excess_wavefronts'] + 2 * (not copy['carry_free_translates']))

    def local_key(name):
        evidence = copy_features[name]
        full_work = local_copy_work(evidence[TILE])
        tail_work = local_copy_work(evidence[tail_rows]) if tail_rows else 0
        return (geometry['full_kv_tiles'] * full_work + tail_work, name)

    local_keys = {name: local_key(name) for name in layouts.keys() - {'legacy'}}
    local_layout = min(local_keys, key=local_keys.__getitem__)
    candidates = {}
    attempts = []
    contract = None
    for name, packet in layouts.items():
        plans = legacy_options if name == 'legacy' else common_options
        for schedule, flags in plans.items():
            start_preview = perf_counter_ns()
            tuning = dict(flags, copy_owner_policy=owners[name])
            row = dict(layout=name, schedule=schedule, tuning=tuning)
            try:
                checked_geometry(spec, kv_width=tuning['kv_width'])
                text, proof = emit_tiled_attention(spec, tile_packet=packet, **tuning)
                current = {key: proof[key] for key in CONTRACT_FIELDS}
                if contract is not None and current != contract:
                    raise RuntimeError('Attention layout or schedule changed the physical or arithmetic contract')
                contract = current
                features = _features(text, proof, copy_features[name])
            except ValueError as error:
                row.update(status='rejected', reason=str(error))
            else:
                key = (name, schedule)
                candidates[key] = Preview(name, schedule, tuning, text, proof, features)
                row.update(status='checked', ttgir_sha256=proof['ttgir_sha256'], features=features)
            row['wall_ns'] = perf_counter_ns() - start_preview
            attempts.append(row)
    anchor = candidates.get(('legacy', 'kv64'))
    if anchor is None:
        raise ValueError('checked legacy KV64 anchor is missing')

    def project(layout_names):
        order = sorted((item for item in candidates.values() if item.layout in layout_names),
                       key=lambda item: (item.features['cost'], item.schedule, item.layout))
        if not order:
            raise ValueError('checked layout projection has no legal schedule')
        unique = {}
        for item in order:
            unique.setdefault(item.proof['ttgir_sha256'], item)
        ranked = list(unique.values())
        result = [ranked[0]]
        # Reserve at most one paid slot for a matched layout intervention
        # on the fixed winner's schedule. LI uses the identical projection
        # and spends that slot on another schedule if its layout is pinned.
        fixed = ranked[0]
        alternate = next((p for p in ranked if p.layout != fixed.layout
                          and p.schedule == fixed.schedule), None)
        limit = K4 - (alternate is not None)
        used_schedules = {fixed.schedule}
        for item in ranked:
            if len(result) == limit:
                break
            if item.schedule not in used_schedules:
                result.append(item)
                used_schedules.add(item.schedule)
        for item in ranked:
            if len(result) == limit:
                break
            if item is not alternate and all(item is not chosen for chosen in result):
                result.append(item)
        if alternate is not None:
            result.append(alternate)
        return (result + [fixed] * K4)[:K4]

    eligible = layouts.keys() - {'legacy'}
    choices = {'J': project(eligible), 'LI': project({local_layout}),
               'legacy': [candidates.get(('legacy', name)) for name in legacy_options]}
    # Unsupported legacy slots are kept as paid failures, even if another
    # checked candidate would fill the slot.
    audit = dict(problem_sha256=problem_sha256, prepared_instance_sha256=digest([problem_sha256, spec]),
                 input_packet_sha256=digest(tile_packet), layout_packet_sha256={
                     name: digest(packet) for name, packet in layouts.items()},
                 contract_sha256=digest(contract), local_layout=local_layout, local_layout_keys=local_keys,
                 legacy_schedule_order=list(legacy_options), attempts=attempts,
                 common_preview_attempts=len(attempts), paid_slots_per_mode=K4,
                 copy_layouts=copy_features, wall_ns=perf_counter_ns() - start)
    return choices, audit
