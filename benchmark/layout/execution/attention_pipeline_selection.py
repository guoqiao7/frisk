"""Bounded plan selection for packed SM90 FlashAttention.

Each candidate binds its execution contract before the multi-anchor layout pass.
The caller supplies independent screening measurements for the executable plans.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import math
import statistics
import subprocess

from backend import compile_execution
from native_online_fixture import online_fixture
from online_execution_contract import from_overlap, single_group


@dataclass(frozen=True)
class AttentionPipeline:
    name: str
    query_tile: int
    contract: dict
    writeback: str = "tma"
    half_output: bool = False
    defer_v_wait: bool = False
    fringe_policy: str = "base"


def attention_pipelines(output_bits, overlap_schedules=(), *, native_delivery=False, aligned_inputs=False):
    extra = []
    if native_delivery and output_bits in (16,32):
        extra.append(AttentionPipeline("q64-producer-wait4",64,single_group(4,128),
                                       "producer",output_bits==16))
        if aligned_inputs:
            extra.append(AttentionPipeline("q64-producer-wait4-late-v",64,single_group(4,128),
                                           "producer",output_bits==16,True))
    if output_bits == 16:
        plans = []
        for source in overlap_schedules:
            with Path(source).open() as stream:
                contract = from_overlap(json.load(stream))
            plans.append(AttentionPipeline(Path(source).stem, 128, contract))
        if not plans and not extra:
            raise ValueError("supply imported FA3 schedules for the FP16 output path")
        return plans + [AttentionPipeline(p.name+"-register",p.query_tile,p.contract,"register") for p in plans] + extra
    if output_bits != 32:
        raise ValueError("unsupported attention output type")
    return [AttentionPipeline(f"q{q}-wait{w}", q, single_group(w,128 if q==64 else 256))
            for q, w in ((128,1),(128,4),(256,1),(256,4),(64,1),(64,4))] + extra


def fringe_pipelines(plan, rows, keys, heads, dim, causal, sm_count, allow_reassociated):
    """Generate only tail policies that can change the work on this grid."""
    if sm_count is not None and (type(sm_count) is not int or sm_count<1):
        raise ValueError("SM count must be a positive integer")
    if dim!=128 or plan.contract['kind']=='single' or plan.writeback!='register':
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


class PackedAttentionImage:
    """Four contiguous BHSD tensors, with the alignment promised at compile time."""
    def __init__(self, image, info, rows, heads, dim, half_output):
        self._image = image
        self._info = info
        self._geometry = (rows, heads, dim)
        self._half_output = half_output

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
            for tensor in (q,k,v,o):
                if tensor.shape != q.shape or not tensor.is_cuda or tensor.device != q.device or not tensor.is_contiguous():
                    raise ValueError("expected contiguous CUDA tensors on one device")
            if any(x.dtype != torch.float16 or x.data_ptr() % self._info['required_input_alignment'] for x in (q,k,v)):
                raise ValueError("input dtype or input alignment disagrees with the compiled contract")
            if o.dtype != (torch.float16 if self._half_output else torch.float32) or o.data_ptr() % self._info['required_output_alignment']:
                raise ValueError("output dtype or output alignment disagrees with the compiled contract")
            if any(x.data_ptr() < o.data_ptr()+o.nbytes and o.data_ptr() < x.data_ptr()+x.nbytes for x in (q,k,v)):
                raise ValueError("output aliases an input")
            with torch.cuda.device(q.device):
                return run(q,k,v,o,**kwargs)
        return launch


def compile_attention_pipeline(plan, opt, directory, *, rows, heads, dim, causal, aligned_inputs=False, row_reciprocal=False, softmax_math="default", kv_rows=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if kv_rows not in (None, rows) and plan.contract['kind']=='single':
        raise ValueError('the single-group packed launcher expects equal Q and KV lengths')
    if plan.fringe_policy!="base" and plan.contract["kind"]=="single":
        raise ValueError("FA3 fringe policies require an imported FA3 execution contract")
    if softmax_math not in ("default","ftz","base2-ftz"):
        raise ValueError("unknown softmax numerical contract")
    if plan.contract['kind'] != 'single' and rows % 128 and plan.writeback != 'register':
        raise ValueError('partial FA3 output tiles require the guarded register writer')
    if plan.defer_v_wait and (plan.contract['kind'] != 'single' or not aligned_inputs):
        raise ValueError('deferred V requires aligned single-group copies')
    source = directory / 'input.mlir'
    packet = directory / 'execution.json'
    source.write_text(online_fixture(plan.query_tile, dim, rows, heads, heads,
                                     causal, execution_contract=plan.contract, kv_rows=kv_rows,
                                     packed_copy=plan.contract["kind"]=="single",
                                     row_reciprocal=row_reciprocal and plan.contract['kind']=='single',
                                     half_output=plan.half_output,
                                     base2_logits=(softmax_math=="base2-ftz" and plan.contract["kind"]=="single")))
    conversion_pass = '--frisk-optimize-layout-conversions'
    if plan.contract['kind'] == 'single' and plan.writeback == 'producer':
        conversion_pass += '=producer-store=true'
    subprocess.run([str(opt), str(source), '--frisk-multi-anchor-layout-pipeline',
                    conversion_pass, '-verify-each',
                    '--frisk-export-execution=output='+str(packet)+' minify=true',
                    '-o', str(directory/'solved.mlir')], check=True)
    data = json.loads(packet.read_text())
    if plan.contract['kind'] == 'single':
        image, info = compile_execution(data, aligned_online_inputs=aligned_inputs, defer_v_wait=plan.defer_v_wait,
                                        flush_exp_subnormals=softmax_math!="default")
        (directory/'kernel.cubin').write_bytes(image.asm['cubin'])
        (directory/'kernel.ptx').write_text(image.asm['ptx'])
    else:
        image, info = compile_execution(data, output_directory=directory, fa3_writeback=plan.writeback,fringe_policy=plan.fringe_policy,
                                        row_reciprocal=row_reciprocal,flush_exp_subnormals=softmax_math!="default")
    info["softmax_math"] = softmax_math
    info["required_output_alignment"] = 16 if plan.half_output else (128 if plan.contract["kind"]=="single" else 16)
    info.update(row_reciprocal=row_reciprocal, required_input_alignment=16 if aligned_inputs and plan.contract['kind']=='single' else (1 if plan.contract['kind']=='single' else 16))
    if info.get('execution_contract') != plan.contract:
        raise ValueError("native lowering used an unexpected execution contract")
    if plan.contract['kind']=='single':
        info['grid'] = info['launch_grid']
        # Preserve the raw kernel ABI for existing single-group plans.
        if aligned_inputs or row_reciprocal or plan.half_output or plan.writeback == 'producer':
            image = PackedAttentionImage(image, info, rows, heads, dim, plan.half_output)
        store, = [op for op in data['operations'] if op['name']=='frisk.tile_store']
        info['terminal_delivery'] = dict(requested=plan.writeback if plan.writeback=='producer' else 'solver',
            consumes_conversion=any(op['name']=='frisk.convert_layout' and
                                    store['operands'][0] in op['results'] for op in data['operations']),
            solver_cost_stage='before physical store selection')
    (directory/'metadata.json').write_text(json.dumps(info,indent=2))
    return image, info


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
                             output_bits, measure, overlap_schedules=(), aligned_inputs=False, row_reciprocal=False, softmax_math="default", kv_rows=None, sm_count=None,
                             allow_reassociated_fringe=False):
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
    plans = attention_pipelines(output_bits, overlap_schedules, native_delivery=True,
                                aligned_inputs=aligned_inputs)
    if len({plan.name for plan in plans}) != len(plans):
        raise ValueError("each proposed schedule must have a distinct name")
    admitted, images, rejected = [], {}, {}
    for plan in plans:
        try:
            images[plan.name] = compile_attention_pipeline(
                plan, opt, Path(directory)/plan.name,
                rows=rows, heads=heads, dim=dim, causal=causal,
                aligned_inputs=aligned_inputs, row_reciprocal=row_reciprocal, softmax_math=softmax_math, kv_rows=kv_rows)
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
                image, meta=compile_execution(packet,output_directory=location,fa3_writeback=plan.writeback,
                    row_reciprocal=row_reciprocal,flush_exp_subnormals=softmax_math!="default",
                    fringe_policy=variant.fringe_policy)
                # Both variants consume the identical solved SSA and anchor proof.
                info=dict(images[plan.name][1],**meta,source_packet=str(base.resolve()))
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
