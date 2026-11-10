"""Checked D64 Attention layout and schedule search, with one paid K4 queue per mode."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from time import perf_counter_ns

from attention_copy_layout import attention_copy_layout_evidence, derive_attention_copy_layouts
from attention_fragment_layout import NATIVE_FRAGMENT, ROW_LANE_REGISTER, checked_fragment_layout
from attention_residency_cost import DRIVER_SHARED_PER_CTA, score
from attention_tile_mma import TILE, score_width_for_prefix, shared_offsets_64x64
from tiled_attention import checked_geometry, emit_tiled_attention


K4 = 4
PREVIEW_CAP = 32
LAYOUT_CONTROL = 'kv64-factored'
K_STAGE_CONTROL = 'kv64-k-stage-control'
K_STAGE_REUSE = 'kv64-k-stage-reuse'
SHORT_SCORE_STAGED = 'kv128-score-staged'
SHORT_SCORE_BULK = 'kv128-score-bulk'
SHORT_SCORE_QUEUE = (K_STAGE_CONTROL, K_STAGE_REUSE, SHORT_SCORE_STAGED, SHORT_SCORE_BULK)
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
    if TILE < n < 2 * TILE:
        # Keep the legal pre-reuse legacy queue frozen for true tails.
        retained = dict(anchor, retain_tail_scores=True)
        options = {'kv64': anchor, LAYOUT_CONTROL: extra.pop(LAYOUT_CONTROL),
                   'kv64-tail-retained': retained,
                   'kv64-tail': dict(retained, narrow_tail=True, simplify_tail_copy=True,
                                     factored_copy=True)}
        cached = dict(anchor, kv_width=2 * TILE, retain_all_scores=True,
                      factored_copy=True, tail_score_carrier=True)
        extra[SHORT_SCORE_STAGED] = cached
        extra[SHORT_SCORE_BULK] = dict(cached, bulk_value_publish=True)
    elif n % TILE:
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
    if TILE < n < 2 * TILE or (n >= 2 * TILE and not n % TILE):
        short = n < 2 * TILE
        control, reuse = ('late-v-control', 'resident-k') if short else ('boundary-control', 'boundary-reuse')
        matched = dict(anchor, factored_copy=True, retain_tail_scores=short)
        extra[K_STAGE_CONTROL] = dict(matched, k_stage_policy=control)
        extra[K_STAGE_REUSE] = dict(matched, k_stage_policy=reuse)
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
            if op.startswith('wgmma.wait_group.'):
                dest['wait_groups'] += 1
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


def _output_work(proof):
    output = proof.get('output_store', {})
    full = proof['q_tiles'] - bool(proof['final_rows'])
    weighted = ((full, output.get(TILE)), (1, output.get(proof['final_rows'])))
    return sum(count * (sum(row['global_sectors32']) / 16 + row['warp_issues'])
               for count, row in weighted if row) / proof['q_tiles']


def _k_stage_work(proof, copies, query_copies, emitted):
    stage = proof['k_stage_reuse']
    for key in ('barriers', 'proxy_fences', 'wgmma_wait0'):
        if type(stage.get(key)) is not int or stage[key] < 0:
            raise ValueError('invalid checked K-stage synchronization ledger')
    if (emitted['sync'] != stage['barriers'] + stage['proxy_fences'] + stage['wgmma_wait0']
            or emitted['mma_issue'] != stage['wgmma_wait0'] * 4):
        raise ValueError('emitted work differs from the checked K-stage ledger')

    def sectors(copy):
        return sum(copy['global_sectors32']) / 2

    if 'k0_publications' in stage:
        publications = [event for event in stage['events'] if event['kind'] == 'publish']
        kv_read = sum(sectors(copies[event['valid_rows']]) for event in publications)
    else:
        kv_read = (stage['k64_publications'] + stage['v_publications']) * sectors(copies[TILE])
    q_tiles, final = proof['q_tiles'], proof['final_rows']
    query_read = ((q_tiles - bool(final)) * sectors(query_copies[TILE])
                  + (sectors(query_copies[final]) if final else 0)) / q_tiles
    return dict(policy=stage['policy'], read_sector_issues=kv_read + query_read,
                kv_global_bytes=stage['kv_global_bytes'], kv_shared_store_bytes=stage['kv_shared_store_bytes'],
                barriers=stage['barriers'], proxy_fences=stage['proxy_fences'], wait0=stage['wgmma_wait0'])


def _features(text, proof, copies, query_copies):
    work = Counter(_emitted_work(text))
    full = copies[TILE]
    tail = copies.get(proof['final_rows'])
    # Static sectors are averaged across the two valid 16B alignment phases.
    # The body counts already include guarded issue sites, so only excess
    # active warps for a narrow tail relative to an ideal packed copy are
    # charged here. The scope does not model dynamic memory overlap.
    read = sum(full['global_sectors32']) / 8 * work['copy_loads']
    publication = (16 + full['shared_excess_wavefronts'] / 4) * work['copy_stores']
    # There is exactly one Q publication per CTA. Charge its actual
    # Shared destination wavefronts when a logical row layout remaps Q.
    query_full = query_copies[TILE]
    publication += query_full['shared_excess_wavefronts'] - full['shared_excess_wavefronts']
    stage = None
    if 'k_stage_reuse' in proof:
        stage = _k_stage_work(proof, copies, query_copies, work)
        read = stage['read_sector_issues']
    short_score = proof.get('tail_score_carrier', False)
    if short_score:
        schedule = proof['score_tail_schedule']
        if (work['mma_issue'] != schedule['qk_atoms'] + proof['pv_atoms']
                or work['mma_n8_units'] != 128
                or work['wait_groups'] != schedule['wgmma_wait0']
                or work['sync'] != schedule['barriers'] + schedule['proxy_fences'] + schedule['wgmma_wait0']
                or tail['storage_rows'] != TILE):
            raise ValueError('emitted work differs from the checked short score carrier')
        # K and V each publish one full block and one zero-filled tail.
        # Q publishes either a full block or a guarded tail, by CTA.
        q_tail = query_copies[proof['final_rows']]
        q_full_fraction = (proof['q_tiles'] - 1) / proof['q_tiles']
        q_tail_fraction = 1 / proof['q_tiles']
        read = (sum(full['global_sectors32']) + sum(tail['global_sectors32'])
                + (q_full_fraction * sum(query_full['global_sectors32'])
                   + q_tail_fraction * sum(q_tail['global_sectors32'])) / 2)
        publication = (16 * work['copy_stores']
                       + (2 * (full['shared_excess_wavefronts'] + tail['shared_excess_wavefronts'])
                          + q_full_fraction * query_full['shared_excess_wavefronts']
                          + q_tail_fraction * q_tail['shared_excess_wavefronts']) / 4)
    tail_issue = 0
    if tail and proof.get('narrow_tail'):
        load_warps = sum(group['load_warps'] for group in tail['groups'])
        ideal_warps = (proof['final_rows'] * 8 + 31) // 32
        tail_issue = max(0, load_warps - ideal_warps) + tail['shared_excess_wavefronts']
    width = proof['execution_kv_width']
    if short_score:
        # QK writes the full wide accumulator. After max, only valid
        # exponent holders survive, with packing and PV state overlapping.
        active = proof['score_active_slots']
        packed_live = active // 2 if proof['value_publications'] == 1 else TILE // 4
        registers = max(56 + proof['score_cache_slots'], 72 + active // 2 + packed_live)
    elif proof.get('retain_all_scores'):
        registers = 48 + proof['score_cache_slots'] + 12 * (not proof.get('value_publications') == 1)
    else:
        registers = {32: 60, 64: 72, 128: 80}[width] + width // 2
        if proof.get('retain_residual_scores'):
            # The residual score carrier survives statistics and overlaps the
            # native fragment of subsequent complete groups.
            registers += max(0, proof['residual_score_cache_slots'] - 12)
        else:
            retained = len(proof.get('retain_tail_score_slots', ()))
            registers += (3 * retained // 4) if width == 64 else (retained // 2)
    if stage and stage['policy'].startswith('boundary-'):
        registers += 2  # The two per-thread row partials survive the shortened sum loop.
    shared = proof['static_shared_bytes']
    work_score = (work['mma_n8_units'] / 2 + 2 * work['mma_issue'] + 4 * work['sync']
                  + read / 8 + publication / 8 + tail_issue + work['scalar_special'] / 2
                  + work['scalar_issue'] / 16 + _output_work(proof) / 4)
    resources = score(work_score, registers=registers, shared_bytes=shared + DRIVER_SHARED_PER_CTA,
                      grid=proof['grid'][0], threads=proof['block'][0])
    return dict(tracked_dynamic_issue_upper_bound=dict(work), copy_full=full, copy_tail=tail, query_copy_full=query_full,
                estimated_residency=resources, estimated_registers=registers,
                estimated_ctas_per_sm=resources['ctas_per_sm'], tail_issue_penalty=tail_issue,
                static_shared_bytes=shared, work=work_score, cost=resources['cost'], k_stage_work=stage)


def select_attention(spec, tile_packet, *, problem_sha256):
    """Project a common checked graph archive into three paid K4 queues.

    LI picks its locally preferred layout for each legal schedule, then uses
    the same whole-kernel score as J to pick a schedule. This lets LI retain
    the latest wide schedules even if their Shared layout domain is narrower.
    """
    if not re.fullmatch('[0-9a-f]{64}', problem_sha256):
        raise ValueError('a checked prepared Attention source digest is required')
    geometry = checked_geometry(spec)
    start = perf_counter_ns()
    legacy_options, common_options = schedules(spec['N'])
    owner_packets = derive_attention_copy_layouts(tile_packet)
    coalesced = owner_packets['coalesced']
    layouts = {
        'coalesced': (coalesced, 128, NATIVE_FRAGMENT),
        'warp-interleaved': (owner_packets['warp-interleaved'], 128, NATIVE_FRAGMENT),
        'shared-sw64': (coalesced, 64, NATIVE_FRAGMENT),
        'mma-row': (coalesced, 128, ROW_LANE_REGISTER),
        'shared-sw64-mma-row': (coalesced, 64, ROW_LANE_REGISTER),
        'legacy': (tile_packet, 128, NATIVE_FRAGMENT),
    }
    if K_STAGE_REUSE in common_options:
        # The matched policies need native SW128. Drop the optional combined
        # layout probe here to keep the existing shared preview cap.
        del layouts['shared-sw64-mma-row']
    tail_rows = geometry['final_rows']
    copy_features, query_features, wide_tail_features = {}, {}, {}
    copy_cache = {}
    for name, (packet, swizzle, fragment_name) in layouts.items():
        explicit = coalesced if name == 'legacy' else packet
        key = (id(explicit), swizzle)
        if key not in copy_cache:
            evidence = {TILE: attention_copy_layout_evidence(explicit, shared_swizzle=swizzle)}
            if tail_rows:
                evidence[tail_rows] = attention_copy_layout_evidence(explicit, valid_rows=tail_rows,
                    storage_rows=score_width_for_prefix(tail_rows), shared_swizzle=swizzle)
            copy_cache[key] = evidence
        copy_features[name] = copy_cache[key]
        if (SHORT_SCORE_STAGED in common_options and name != 'legacy' and swizzle == 128
                and fragment_name == NATIVE_FRAGMENT):
            evidence = copy_cache[key]
            wide_tail = (evidence[tail_rows] if evidence[tail_rows]['storage_rows'] == TILE else
                         attention_copy_layout_evidence(explicit, valid_rows=tail_rows, shared_swizzle=swizzle))
            wide_tail_features[name] = evidence | {tail_rows: wide_tail}
        if fragment_name == NATIVE_FRAGMENT:
            query_features[name] = copy_cache[key]
        else:
            query_offsets = checked_fragment_layout(fragment_name).query_offsets(
                shared_offsets_64x64(swizzle))
            query_features[name] = {rows: attention_copy_layout_evidence(explicit, valid_rows=rows,
                shared_swizzle=swizzle, destination_offsets=query_offsets)
                for rows in copy_cache[key]}

    def local_copy_work(copy):
        groups = copy['groups']
        return (sum(copy['global_sectors32']) / 16 + sum(g['load_warps'] for g in groups)
                + sum(bool(g['initialized_vectors']) for g in groups)
                + copy['shared_excess_wavefronts'] + 2 * (not copy['carry_free_translates']))

    def local_key(name, proof):
        evidence, query = ((wide_tail_features[name], wide_tail_features[name])
                           if proof.get('tail_score_carrier') else (copy_features[name], query_features[name]))
        full_work = local_copy_work(evidence[TILE])
        tail_work = local_copy_work(evidence[tail_rows]) if tail_rows else 0
        qwork = local_copy_work(query[TILE])
        return (geometry['full_kv_tiles'] * full_work + tail_work + qwork + _output_work(proof), name)

    candidates = {}
    attempts = []
    contract = None
    for name, (packet, swizzle, fragment_name) in layouts.items():
        plans = legacy_options if name == 'legacy' else common_options
        for schedule, flags in plans.items():
            if ((flags.get('k_stage_policy') or flags.get('tail_score_carrier'))
                    and (swizzle != 128 or fragment_name != NATIVE_FRAGMENT)):
                continue
            if swizzle == 64 and (flags['kv_width'] != TILE or flags.get('retain_all_scores')):
                continue
            start_preview = perf_counter_ns()
            tuning = dict(flags, copy_owner_policy='gf2' if name == 'legacy' else 'execution',
                          shared_swizzle=swizzle, fragment_layout=fragment_name)
            row = dict(layout=name, schedule=schedule, tuning=tuning)
            try:
                checked_geometry(spec, kv_width=tuning['kv_width'],
                                 tail_score_carrier=tuning.get('tail_score_carrier', False))
                text, proof = emit_tiled_attention(spec, tile_packet=packet, **tuning)
                current = {key: proof[key] for key in CONTRACT_FIELDS}
                if contract is not None and current != contract:
                    raise RuntimeError('Attention layout or schedule changed the physical or arithmetic contract')
                contract = current
                if (proof.get('shared_swizzle', 128) != swizzle
                        or proof.get('fragment_layout', NATIVE_FRAGMENT) != fragment_name):
                    raise RuntimeError('emitted Attention layout differs from its paid axes')
                if proof.get('k_stage_reuse', {}).get('policy') != tuning.get('k_stage_policy'):
                    raise RuntimeError('emitted Attention K-stage strategy differs from its candidate')
                if proof.get('tail_score_carrier', False) != tuning.get('tail_score_carrier', False):
                    raise RuntimeError('emitted score carrier differs from its candidate')
                evidence = wide_tail_features[name] if proof.get('tail_score_carrier') else copy_features[name]
                query = evidence if proof.get('tail_score_carrier') else query_features[name]
                features = _features(text, proof, evidence, query)
                features['local_layout_key'] = local_key(name, proof)
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

    if len(attempts) > PREVIEW_CAP:
        raise ValueError('Attention exhausted the common static preview budget')

    def project(pool):
        ranked = sorted(pool, key=lambda item: (item.features['cost'], item.schedule, item.layout))
        if not ranked:
            raise ValueError('checked layout projection has no legal schedule')
        result = []
        identities = set()

        def add(item):
            if item is not None and len(result) < K4 and item.proof['ttgir_sha256'] not in identities:
                result.append(item)
                identities.add(item.proof['ttgir_sha256'])

        add(ranked[0])
        if K_STAGE_REUSE in common_options:
            control_by_layout = {p.layout: p for p in ranked if p.schedule == K_STAGE_CONTROL}
            reuse = next((p for p in ranked if p.schedule == K_STAGE_REUSE
                          and p.layout in control_by_layout), None)
            if reuse is None:
                raise ValueError('checked K-stage ablation lacks a compatible control and reuse pair')
            add(control_by_layout[reuse.layout])
            add(reuse)
            add(next((p for p in ranked if p.schedule not in (K_STAGE_CONTROL, K_STAGE_REUSE)), None))
        else:
            control = next((p for p in ranked if p.layout == 'coalesced'
                            and p.schedule == LAYOUT_CONTROL), None)
            if control is None:
                control = next((p for p in ranked if p.schedule == LAYOUT_CONTROL), None)
            add(control)
            if control is not None and control.layout == 'coalesced':
                for alternate in ('shared-sw64', 'mma-row'):
                    add(next((p for p in ranked if p.layout == alternate and p.schedule == control.schedule),
                             None))
        for item in ranked:
            if all(chosen.schedule != item.schedule for chosen in result):
                add(item)
        for item in ranked:
            add(item)
        return (result + [ranked[0]] * K4)[:K4]

    joint = [item for item in candidates.values() if item.layout != 'legacy']
    local = {}
    for item in joint:
        current = local.get(item.schedule)
        if current is None or item.features['local_layout_key'] < current.features['local_layout_key']:
            local[item.schedule] = item
    if SHORT_SCORE_STAGED in common_options:
        # Keep the four schedules on one Copy16 layout within each mode so
        # the paired comparisons isolate K lifetime, QK reuse, and PV grouping.
        matched = {
            name: [candidates[name, schedule] for schedule in SHORT_SCORE_QUEUE]
            for name in wide_tail_features
            if all((name, schedule) in candidates for schedule in SHORT_SCORE_QUEUE)
        }
        if not matched:
            raise ValueError('checked short score carrier lacks a complete matched K4 queue')

        def short_queue(items):
            result = sorted(items, key=lambda item: (item.features['cost'], item.schedule))
            if len({item.proof['ttgir_sha256'] for item in result}) != K4:
                raise ValueError('short score carrier K4 queue requires distinct images')
            return result

        joint_queue = min(matched.values(), key=lambda items: (min(item.features['cost'] for item in items),
                                                               items[0].layout))
        local_queue = min(matched.values(), key=lambda items: (sum(item.features['local_layout_key'][0]
                                                                  for item in items), items[0].layout))
        choices = {'J': short_queue(joint_queue), 'LI': short_queue(local_queue)}
    else:
        choices = {'J': project(joint), 'LI': project(local.values())}
    choices['legacy'] = [candidates.get(('legacy', name)) for name in legacy_options]
    # Unsupported legacy slots are kept as paid failures, even if another
    # checked candidate would fill the slot.
    audit = dict(problem_sha256=problem_sha256, prepared_instance_sha256=digest([problem_sha256, spec]),
                 input_packet_sha256=digest(tile_packet), layout_packet_sha256={
                     name: digest(packet) for name, (packet, _, _) in layouts.items()},
                 contract_sha256=digest(contract), local_layout=local[LAYOUT_CONTROL].layout,
                 local_layout_by_schedule={name: item.layout for name, item in local.items()},
                 layout_axes={name: {'shared_swizzle': swizzle, 'fragment_layout': fragment}
                              for name, (_, swizzle, fragment) in layouts.items()},
                 legacy_schedule_order=list(legacy_options), attempts=attempts,
                 k_stage_pairs={mode: {item.schedule: item.layout for item in choices[mode]
                                       if item.schedule in (K_STAGE_CONTROL, K_STAGE_REUSE)}
                                for mode in ('J', 'LI')},
                 common_preview_attempts=len(attempts), common_preview_cap=PREVIEW_CAP, paid_slots_per_mode=K4,
                 copy_layouts=copy_features, wall_ns=perf_counter_ns() - start)
    return choices, audit
