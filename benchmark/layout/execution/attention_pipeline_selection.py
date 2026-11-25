"""Bounded plan selection for packed SM90 FlashAttention.

Each candidate binds its execution contract before the multi-anchor layout pass.
The caller supplies independent screening measurements for the executable plans.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import math
import shutil
import statistics
import subprocess

from backend import compile_execution
from attention_pipeline_candidates import AttentionPipeline, attention_pipelines, pipeline_candidates
from native_online_fixture import online_fixture
from online_execution_contract import packed_copy_supports_lookahead


def fringe_pipelines(plan, rows, keys, heads, dim, causal, sm_count, allow_reassociated):
    """Generate only tail policies that can change the work on this grid."""
    if sm_count is not None and (type(sm_count) is not int or sm_count<1):
        raise ValueError("SM count must be a positive integer")
    if dim!=128 or plan.query_tile!=128 or plan.contract['kind']=='single' or plan.writeback!='register':
        return []
    full, tail = divmod(rows,128)
    blocks=(full+bool(tail))*heads
    # The imported FA3 plan needs more than half the available SM90 shared
    # memory. Its 384-thread block therefore has single-block residency.
    def waves(n):
        return (n+sm_count-1)//sm_count
    single_resident = plan.contract.get('arena',0)>228*1024//2
    capacity_known = sm_count is not None and single_resident
    crosses = not capacity_known or waves(blocks)>waves(full*heads)
    multiple = not capacity_known or blocks>sm_count
    short = allow_reassociated and 0<keys%128<=8 and (not causal or rows>keys-keys%128)
    kinds=[]
    if short:
        kinds.append("short")
    if allow_reassociated and not causal and 0<tail<=8 and crosses and (keys+63)//64<=65535:
        kinds.append("split")
    if causal and multiple:
        kinds.append("reverse")
        if short:
            kinds.append("reverse-short")
    return [replace(plan,name=plan.name+"-"+kind,fringe_policy=kind) for kind in kinds]


@dataclass(frozen=True, eq=False)
class PreparedAttentionK:
    image: object
    key: tuple
    k: object
    descriptor: object


@dataclass(frozen=True, eq=False)
class PreparedAttentionV:
    image: object
    key: tuple
    v: object
    descriptor: object


class PackedAttentionImage:
    """Four contiguous BHSD tensors, with the alignment promised at compile time."""
    def __init__(self, image, info, rows, heads, dim, half_output, *, kv_rows=None, kv_heads=None, value_dim=None):
        self._image = image
        self._info = info
        self._geometry = (rows, heads, dim)
        self._kv_geometry = (kv_rows if kv_rows is not None else rows, kv_heads if kv_heads is not None else heads, value_dim if value_dim is not None else dim)
        self._half_output = half_output
        self._map_cache = {}

    def prepare_k(self, k):
        """Retain this handle with the inputs when capturing a CUDA graph."""
        return self._prepare_map("K", k)

    def prepare_v(self, v):
        """Prepare a by-value V map for reuse, including CUDA graph capture."""
        return self._prepare_map("V", v)

    def _prepare_map(self, operand, tensor):
        import torch
        from async_host import prepare_attention_k_map, prepare_attention_v_map
        plan = self._info.get(operand.lower() + '_tensor_map')
        if plan is None:
            raise ValueError(f"this kernel does not take a {operand} tensor map")
        key = (tensor.device, tensor.data_ptr())
        cached = self._map_cache.get(operand)
        if cached is None or cached.key != key:
            with torch.cuda.device(tensor.device):
                if torch.cuda.is_current_stream_capturing():
                    raise ValueError(f"prepare the {operand} tensor map before capture")
                encode = prepare_attention_k_map if operand == 'K' else prepare_attention_v_map
                descriptor = encode(plan, tensor)
            handle_type = PreparedAttentionK if operand == "K" else PreparedAttentionV
            cached = handle_type(self._image, key, tensor, descriptor)
            self._map_cache[operand] = cached
        return cached

    def __getattr__(self, name):
        return getattr(self._image, name)

    def __getitem__(self, grid):
        if tuple(grid) != tuple(self._info['grid']):
            raise ValueError("packed attention launch grid does not match the compiled geometry")
        run = self._image[tuple(grid)]
        def launch(q, k, v, o, **kwargs):
            import torch
            rows, heads, dim = self._geometry
            if q.ndim != 4 or q.shape[0]*q.shape[1] != heads or tuple(q.shape[2:]) != (rows,dim):
                raise ValueError("expected packed BHSD tensors matching the compiled geometry")
            batch=q.shape[0]
            length, kv_heads, value_dim = self._kv_geometry
            if kv_heads % batch or q.shape[1] % (kv_heads//batch):
                raise ValueError("batch and grouped KV heads disagree with the compiled geometry")
            expected=((batch,kv_heads//batch,length,dim),(batch,kv_heads//batch,length,value_dim),
                      (batch,q.shape[1],rows,value_dim))
            if any(tuple(t.shape)!=shape for t,shape in zip((k,v,o),expected)):
                raise ValueError("packed K, V or O shape disagrees with the compiled geometry")
            for tensor in (q,k,v,o):
                if not tensor.is_cuda or tensor.device != q.device or not tensor.is_contiguous():
                    raise ValueError("expected contiguous CUDA tensors on one device")
            if any(x.dtype != torch.float16 or x.data_ptr() % self._info['required_input_alignment'] for x in (q,k,v)):
                raise ValueError("input dtype or input alignment disagrees with the compiled contract")
            if o.dtype != (torch.float16 if self._half_output else torch.float32) or o.data_ptr() % self._info['required_output_alignment']:
                raise ValueError("output dtype or output alignment disagrees with the compiled contract")
            if any(x.data_ptr() < o.data_ptr()+o.nbytes and o.data_ptr() < x.data_ptr()+x.nbytes for x in (q,k,v)):
                raise ValueError("output aliases an input")
            with torch.cuda.device(q.device):
                if 'k_tensor_map' in self._info:
                    stream = torch.cuda.current_stream(q.device)
                    stream_id = kwargs.get("stream")
                    scope = nullcontext()
                    if stream_id is not None and stream_id != stream.cuda_stream:
                        stream = (torch.cuda.ExternalStream(stream_id, device=q.device)
                                  if stream_id else torch.cuda.default_stream(q.device))
                        scope = torch.cuda.stream(stream)
                    with scope:
                        handle = kwargs.pop("prepared_k", None)
                        if handle is None:
                            if torch.cuda.is_current_stream_capturing():
                                raise ValueError("capture requires a retained prepare_k(k) handle")
                            handle = self.prepare_k(k)
                        if (not isinstance(handle, PreparedAttentionK) or handle.image is not self._image
                                or handle.key != (k.device, k.data_ptr())):
                            raise ValueError("prepared K tensor map disagrees with this launch")
                        if self._info['k_tensor_map'].get('abi','global') == 'global':
                            handle.descriptor.record_stream(stream)
                        k.record_stream(stream)
                        if 'v_tensor_map' in self._info:
                            value = kwargs.pop('prepared_v', None)
                            if value is None:
                                value = self.prepare_v(v)
                            if (not isinstance(value, PreparedAttentionV) or value.image is not self._image
                                    or value.key != (v.device, v.data_ptr())):
                                raise ValueError("prepared V tensor map disagrees with this launch")
                            v.record_stream(stream)
                            return run(q,k,v,o,handle.descriptor,value.descriptor,**kwargs)
                        return run(q,k,v,o,handle.descriptor,**kwargs)
                return run(q,k,v,o,**kwargs)
        return launch


def _online_expectation(plan, k_transport):
    return dict(q_tile=plan.query_tile, kv_tile=plan.kv_tile or 64, batch=plan.contract["batch"],
                paths=dict(q="cp.async",k=k_transport,v=plan.v_transport),
                writer=plan.writeback if plan.writeback=="producer" else "solver",
                post_qk=plan.prefetch_after_qk, plan=plan.name)


def compile_attention_pipeline(plan, opt, directory, *, rows, heads, dim, causal, aligned_inputs=False, row_reciprocal=False, softmax_math="default", kv_rows=None, kv_heads=None, value_dim=None, k_transport="cp.async", k_map_abi="value", overlap_pv_normalizer=True, cross_iteration_pv=False, _layout_cache=None):
    """Compile one plan. A joint K/V TMA plan uses two by-value tensor maps."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if k_map_abi not in ("value", "global"):
        raise ValueError("K tensor map ABI must be value or global")
    if k_transport not in ("cp.async", "tma") or (k_transport=="tma" and
            (plan.contract["kind"]!="single" or not aligned_inputs or not plan.defer_v_wait)):
        raise ValueError("K TMA requires an aligned single-group two-slot plan")
    kv_heads = heads if kv_heads is None else kv_heads
    value_dim = dim if value_dim is None else value_dim
    if type(kv_heads) is not int or kv_heads<=0 or heads%kv_heads:
        raise ValueError("packed query heads must be divisible by packed KV heads")
    if plan.contract['kind']!='single' and value_dim!=dim:
        raise ValueError("imported FA3 requires equal head dimensions")
    if plan.fringe_policy!="base" and plan.contract["kind"]=="single":
        raise ValueError("FA3 fringe policies require an imported FA3 execution contract")
    if softmax_math not in ("default","ftz","base2-ftz"):
        raise ValueError("unknown softmax numerical contract")
    if plan.contract['kind'] != 'single' and rows % plan.query_tile and plan.writeback != 'register':
        raise ValueError('partial FA3 output tiles require the guarded register writer')
    if plan.defer_v_wait and (plan.contract['kind'] != 'single' or not aligned_inputs):
        raise ValueError('deferred V requires aligned single-group copies')
    source = directory / 'input.mlir'
    packet = directory / 'execution.json'
    is_fa3 = plan.contract["kind"] != "single"
    if plan.event_schedule is not None and not is_fa3:
        raise ValueError("FA3 events require a native FA3 execution contract")
    layout_contract = plan.contract
    event_schedule = plan.event_schedule
    if is_fa3:
        from fa3_event_schedule import resolve_events
        event_schedule = resolve_events(plan.contract["kind"],event_schedule)
        if _layout_cache is not None:
            # The event frame may change after layout; every physical field is retained.
            layout_contract = dict(plan.contract,kind="fa3_qk_next")
    if plan.v_transport not in ('cp.async','tma') or (plan.v_transport == 'tma' and
            (is_fa3 or cross_iteration_pv or plan.kv_prefetch_edges is not None or k_transport != 'tma' or k_map_abi != 'value')):
        raise ValueError('V TMA requires joint K/V issue and by-value native maps')
    if plan.kv_prefetch_edges is not None and (is_fa3 or plan.prefetch_after_qk or cross_iteration_pv
            or k_transport != 'tma' or k_map_abi != 'value'):
        raise ValueError('independent K/V issue requires the original waits and by-value K TMA')
    if plan.prefetch_after_qk and (is_fa3 or cross_iteration_pv or k_transport != 'tma' or k_map_abi != 'value'):
        raise ValueError('post-QK prefetch requires a single-group by-value K-TMA plan with same-iteration PV')
    if cross_iteration_pv and is_fa3:
        raise ValueError("cross-iteration PV requires a single native warpgroup")
    kv_tile = plan.kv_tile
    grouped_qk_ring = (not is_fa3 and dim == value_dim == 128 and kv_heads < heads and
                       kv_tile in (64,128) and plan.query_tile == 64 and plan.contract["cta"] == 128
                       and plan.defer_v_wait and aligned_inputs and not plan.half_output
                       and (k_transport == "cp.async" or k_map_abi == "value"))
    paired_rows = (not is_fa3 and plan.query_tile == 128 and dim == value_dim == 64
                   and not causal and not plan.half_output and kv_heads < heads and plan.contract['cta'] == 256
                   and k_transport == 'tma' and k_map_abi == 'value' and aligned_inputs
                   and plan.defer_v_wait and plan.kv_prefetch_edges is None and kv_tile in (64,128))
    if not is_fa3 and plan.query_tile == 128 and k_transport == "tma" and not paired_rows:
        raise ValueError("Q128 K TMA requires the ordinary grouped D64 two-warpgroup plan")
    if not is_fa3 and kv_tile==128 and not (grouped_qk_ring or paired_rows) and not (
            plan.query_tile==dim==value_dim==64 and plan.contract["cta"]==128 and aligned_inputs
            and plan.defer_v_wait and k_transport=="tma"
            and k_map_abi=="value"):
        raise ValueError("native KV128 requires the D64 by-value K-TMA two-slot plan")
    if plan.v_row_major_tma and (is_fa3 or plan.v_transport != 'tma' or dim != 64 or value_dim != 64):
        raise ValueError('row-major SW128 V requires native D64 V TMA')
    ir = online_fixture(plan.query_tile, dim, rows, heads, kv_heads,
                                     causal, execution_contract=layout_contract, kv_rows=kv_rows, kv_tile=kv_tile, value_dim=value_dim,
                                     v_row_major=plan.v_row_major_tma, packed_copy=plan.contract["kind"]=="single",
                                     row_reciprocal=row_reciprocal and plan.contract['kind']=='single',
                                     half_output=plan.half_output,
                                     base2_logits=(softmax_math=="base2-ftz" and plan.contract["kind"]=="single"))
    source.write_text(ir)
    conversion_pass = '--frisk-optimize-layout-conversions'
    if plan.contract['kind'] == 'single' and plan.writeback == 'producer':
        conversion_pass += '=producer-store=true'
    # This cache is private to one tuning invocation. Key the entire input and
    # pass configuration, and publish a packet only after a successful solve.
    key = (ir,str(opt),conversion_pass)
    share_layout = _layout_cache is not None
    cached = _layout_cache.get(key) if share_layout else None
    solved = directory/'solved.mlir'
    if cached is None:
        subprocess.run([str(opt), str(source), '--frisk-multi-anchor-layout-pipeline',
                        conversion_pass, '-verify-each',
                        '--frisk-export-execution=output='+str(packet)+' minify=true',
                        '-o', str(solved)], check=True)
        if share_layout:
            _layout_cache[key] = directory
    else:
        shutil.copyfile(cached/'execution.json',packet)
        shutil.copyfile(cached/'solved.mlir',solved)
    data = json.loads(packet.read_text())
    if plan.contract['kind'] == 'single':
        image, info = compile_execution(data, online_loop_schedule="grouped-qk-ring" if grouped_qk_ring else "auto", v_transport=plan.v_transport, kv_prefetch_edges=plan.kv_prefetch_edges, prefetch_after_qk=plan.prefetch_after_qk, cross_iteration_pv=cross_iteration_pv, overlap_pv_normalizer=overlap_pv_normalizer, k_transport=k_transport, k_map_abi=k_map_abi, aligned_online_inputs=aligned_inputs, defer_v_wait=plan.defer_v_wait,
                                        flush_exp_subnormals=softmax_math!="default")
        if (grouped_qk_ring or paired_rows) and not info.get("loop_optimizations",{}).get("ring"):
            raise ValueError("the native allocator could not realize the grouped QK ring")
        (directory/'kernel.cubin').write_bytes(image.asm['cubin'])
        (directory/'kernel.ptx').write_text(image.asm['ptx'])
    else:
        image, info = compile_execution(data, output_directory=directory, event_schedule=event_schedule, fa3_writeback=plan.writeback,fringe_policy=plan.fringe_policy,
                                        cta_dispatch=plan.cta_dispatch,
                                        row_reciprocal=row_reciprocal,flush_exp_subnormals=softmax_math!="default")
        info['source_packet']=str(((cached or directory)/'execution.json').resolve())
    info["softmax_math"] = softmax_math
    info["required_output_alignment"] = 16 if plan.half_output else (128 if plan.contract["kind"]=="single" else 16)
    info.update(row_reciprocal=row_reciprocal, required_input_alignment=16 if aligned_inputs and plan.contract['kind']=='single' else (1 if plan.contract['kind']=='single' else 16))
    expected_contract = dict(plan.contract,kind=event_schedule.contract_kind) if is_fa3 else plan.contract
    if info.get('execution_contract') != expected_contract:
        raise ValueError("native lowering used an unexpected execution contract")
    if plan.contract['kind']=='single':
        info['grid'] = info['launch_grid']
        # Preserve the raw kernel ABI for existing single-group plans.
        if aligned_inputs or row_reciprocal or plan.half_output or plan.writeback == 'producer' or kv_heads!=heads or kv_rows not in (None,rows) or value_dim!=dim:
            image = PackedAttentionImage(image, info, rows, heads, dim, plan.half_output,
                                         kv_rows=kv_rows,kv_heads=kv_heads,value_dim=value_dim)
        store, = [op for op in data['operations'] if op['name']=='frisk.tile_store']
        info['terminal_delivery'] = dict(requested=plan.writeback if plan.writeback=='producer' else 'solver',
            consumes_conversion=any(op['name']=='frisk.convert_layout' and
                                    store['operands'][0] in op['results'] for op in data['operations']),
            solver_cost_stage='before physical store selection')
    if not is_fa3:
        from attention_execution_audit import digest
        physical = asdict(plan)
        physical.pop("name")
        physical.update(rows=rows, keys=rows if kv_rows is None else kv_rows, heads=heads,
                        kv_heads=kv_heads, dim=dim, value_dim=value_dim, k_transport=k_transport,
                        k_map_abi=k_map_abi if k_transport == "tma" else None,
                        aligned_inputs=aligned_inputs, row_reciprocal=row_reciprocal,
                        softmax_math=softmax_math, overlap_pv_normalizer=overlap_pv_normalizer,
                        cross_iteration_pv=cross_iteration_pv, causal=causal)
        info.update(physical_candidate=plan.name, physical_contract=physical,
                    physical_contract_sha256=digest(physical))
    audit_grouped = (dim == value_dim == 128 and kv_heads < heads and not plan.half_output
                     and plan.query_tile == 64 and plan.defer_v_wait)
    if not is_fa3 and aligned_inputs and (dim == value_dim == 64 or audit_grouped):
        from attention_execution_audit import audit_attention
        expected = _online_expectation(plan, k_transport)
        if plan.v_transport == "tma":
            expected["v_swizzle"] = 128 if plan.v_row_major_tma else 32
        if k_transport == "tma" or grouped_qk_ring:
            expected["kv_versions"] = 2
        elif kv_rows in (None, rows) and not cross_iteration_pv:
            lookahead = (dim == 64 and not plan.half_output and plan.defer_v_wait and
                         plan.contract["cta"] == 128 and packed_copy_supports_lookahead((rows+63)//64))
            expected["kv_versions"] = 2 if lookahead else 1
        info["execution_audit"] = audit_attention(data, info, image.asm, expected)
        info["layout_solve_reused"] = cached is not None
    (directory/'metadata.json').write_text(json.dumps(info,indent=2))
    return image, info


def compile_static_d64_attention(opt, directory, *, rows, heads, causal, sm_count=114,
                                 aligned_inputs, row_reciprocal, softmax_math, output_bits=32):
    """Compile one H100 PCIe D64 self-attention plan without device timing.

    The calibrated contract uses packed BHSD FP16 Q/K/V, base-2 logits, FP32
    row reciprocals, and 16-byte input alignment. Other numerical or launch
    contracts can continue to use compile_attention_pipeline or measured tuning.
    """
    from d64_static_pipeline import select_d64_pipeline
    if not aligned_inputs or not row_reciprocal or softmax_math != "base2-ftz":
        raise ValueError("D64 static policy requires aligned packed inputs, row reciprocals and base2-ftz")
    candidate, receipt = select_d64_pipeline(rows, heads, causal, sm_count, output_bits=output_bits)
    return _compile_static_online(opt,directory,rows,heads,heads,64,causal,candidate,receipt)


def compile_static_gqa_attention(opt, directory, *, rows, heads, kv_heads, dim, causal,
                                 aligned_inputs, row_reciprocal, softmax_math, sm_count=114, output_bits=32):
    """Select and compile one single-group GQA image on H100 PCIe."""
    from d64_static_pipeline import select_grouped_pipeline
    if not aligned_inputs or not row_reciprocal or softmax_math!="base2-ftz":
        raise ValueError("GQA static policy requires aligned packed inputs, row reciprocals and base2-ftz")
    candidate, receipt = select_grouped_pipeline(rows,heads,kv_heads,dim,causal,sm_count,output_bits=output_bits)
    return _compile_static_online(opt,directory,rows,heads,kv_heads,dim,causal,candidate,receipt)


def _compile_static_online(opt, directory, rows, heads, kv_heads, dim, causal, candidate, receipt):
    image, metadata = compile_attention_pipeline(
        candidate.plan, opt, directory, rows=rows, heads=heads, kv_heads=kv_heads, dim=dim,
        causal=causal, aligned_inputs=True, row_reciprocal=True, softmax_math="base2-ftz",
        k_transport=candidate.k_transport, k_map_abi="value")
    receipt["execution_audit"] = metadata["execution_audit"]
    return image, metadata, receipt


def select_attention_pipeline(plans, paired_samples):
    names = [p.name for p in plans]
    if not names or len(set(names)) != len(names) or not paired_samples:
        raise ValueError("expected unique executable plans and paired measurements")
    for row in paired_samples:
        if set(row)!=set(names) or any(not math.isfinite(x) or x<=0 for x in row.values()):
            raise ValueError("invalid per-plan latency")
    scores = {name:statistics.median(row[name] for row in paired_samples) for name in names}
    return min(plans,key=lambda plan:scores[plan.name]), scores


def tune_attention_pipelines(opt, directory, *, rows, heads, dim, causal,
                             output_bits, measure, overlap_schedules=(), aligned_inputs=False, row_reciprocal=False, softmax_math="default", kv_rows=None, kv_heads=None, value_dim=None, sm_count=None,
                             allow_reassociated_fringe=False, include_late_k=False, fa3_query_tile=128,
                             k_transport=None, k_map_abi="value", include_post_qk_prefetch=False, include_independent_kv=False, include_v_tma=True):
    """Return the winning compiled image, its metadata, and a selection receipt.

    aligned_inputs promises 16-byte-aligned Q/K/V bases for single-group copies.
    row_reciprocal permits FP32 reciprocal rounding followed by FP32 multiplication.
    softmax_math permits exponential FTZ, and optionally scaling logits into base-2 units.
    sm_count is the SM count available to the launch, including MIG partitions.
    allow_reassociated_fringe permits FP32 reduction reassociation and locally rounded FP16 probabilities.
    measure(images) validates and measures identical inputs. Its argument maps
    plan names to (image, metadata) pairs, and its result is a sequence of
    paired latency dictionaries. Measurement is supplied by the runtime so the
    compiler does not assume an allocator, CUDA context, or tensor library.
    """
    candidates = pipeline_candidates(
        rows=rows, heads=heads, dim=dim, causal=causal, output_bits=output_bits,
        aligned_inputs=aligned_inputs, kv_rows=kv_rows, kv_heads=kv_heads, value_dim=value_dim,
        k_transport=k_transport, k_map_abi=k_map_abi, include_post_qk_prefetch=include_post_qk_prefetch,
        include_independent_kv=include_independent_kv, include_v_tma=include_v_tma,
        overlap_schedules=overlap_schedules, sm_count=sm_count, include_late_k=include_late_k,
        fa3_query_tile=fa3_query_tile)
    admitted, images, rejected = [], {}, {}
    layout_cache = {}
    for candidate in candidates:
        plan = candidate.plan
        try:
            images[plan.name] = compile_attention_pipeline(
                plan, opt, Path(directory)/plan.name,
                rows=rows, heads=heads, dim=dim, causal=causal,
                aligned_inputs=aligned_inputs, row_reciprocal=row_reciprocal, softmax_math=softmax_math, kv_rows=kv_rows,
                kv_heads=kv_heads, value_dim=value_dim,
                k_transport=candidate.k_transport,
                k_map_abi=k_map_abi, _layout_cache=layout_cache)
        except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            rejected[plan.name] = str(exc)
            continue
        admitted.append(plan)
        for variant in fringe_pipelines(plan,rows,rows if kv_rows is None else kv_rows,heads,dim,
                                       causal,sm_count,allow_reassociated_fringe):
            location=Path(directory)/variant.name
            location.mkdir(parents=True,exist_ok=True)
            base=Path(directory)/plan.name/'execution.json'
            try:
                packet=json.loads(base.read_text())
                from fa3_event_schedule import resolve_events
                events=resolve_events(plan.contract["kind"],plan.event_schedule)
                image, meta=compile_execution(packet,output_directory=location,fa3_writeback=plan.writeback,event_schedule=events,
                    row_reciprocal=row_reciprocal,flush_exp_subnormals=softmax_math!="default",
                    fringe_policy=variant.fringe_policy)
                # Both variants consume the identical solved SSA and anchor proof.
                info=dict(images[plan.name][1],**meta)
                (location/'metadata.json').write_text(json.dumps(info,indent=2))
            except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
                rejected[variant.name]=str(exc)
                continue
            admitted.append(variant)
            images[variant.name]=(image,info)
    if not images:
        raise ValueError(f"no executable attention pipeline: {rejected}")
    winner, scores = select_attention_pipeline(admitted, list(measure(images)))
    image, info = images[winner.name]
    receipt = dict(selected=winner.name, scores=scores, rejected=rejected)
    return image, info, receipt
