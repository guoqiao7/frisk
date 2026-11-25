"""Physical candidate catalog shared by packed attention ranking and tuning."""
from dataclasses import dataclass, replace
import json
from pathlib import Path

from online_execution_contract import (fa3_query_plan, from_overlap, short_causal_row_major_v,
                                       short_kv64_sw128_tma, single_group)


@dataclass(frozen=True)
class AttentionPipeline:
    name: str
    query_tile: int
    contract: dict
    writeback: str = "tma"
    half_output: bool = False
    defer_v_wait: bool = False
    fringe_policy: str = "base"
    event_schedule: str = None
    kv_tile: int = None
    prefetch_after_qk: bool = False
    kv_prefetch_edges: tuple[str, str] = None
    v_transport: str = "cp.async"
    cta_dispatch: str = None
    v_row_major_tma: bool = False


@dataclass(frozen=True)
class _PipelineCandidate:
    plan: AttentionPipeline
    k_transport: str
    # Static model key. Additional tuning controls have no calibration.
    ranked_family: str | None


def attention_pipelines(output_bits, overlap_schedules=(), *, native_delivery=False, aligned_inputs=False, include_late_k=False, fa3_query_tile=128):
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
                if fa3_query_tile != 128:
                    contract = fa3_query_plan(contract, fa3_query_tile)
            plans.append(AttentionPipeline(Path(source).stem, fa3_query_tile, contract))
            if include_late_k and contract['kind']=='fa3_qk_next':
                plans.append(replace(plans[-1],name=plans[-1].name+"-late-k",event_schedule="late_k"))
        if not plans and not extra:
            raise ValueError("supply imported FA3 schedules for the FP16 output path")
        return plans + [replace(p,name=p.name+"-register",writeback="register") for p in plans] + extra
    if output_bits != 32:
        raise ValueError("unsupported attention output type")
    return [AttentionPipeline(f"q{q}-wait{w}", q, single_group(w,128 if q==64 else 256))
            for q, w in ((128,1),(128,4),(256,1),(256,4),(64,1),(64,4))] + extra


def pipeline_candidates(*, rows, heads, dim, causal, output_bits, aligned_inputs,
                         kv_rows=None, kv_heads=None, value_dim=None, k_transport=None,
                         k_map_abi="value", include_post_qk_prefetch=False, include_independent_kv=False,
                         include_v_tma=True, overlap_schedules=(), sm_count=None, include_late_k=False,
                         fa3_query_tile=128):
    """Bind physical plans and their calibrated families before either caller compiles."""
    d64 = dim == 64 and value_dim in (None, 64)
    grouped_d128 = dim == 128 and value_dim in (None, 128) and kv_heads not in (None, heads)
    mixed_profile = (d64 and output_bits in (16, 32)) or (grouped_d128 and output_bits == 32)
    combined = k_transport is None and aligned_inputs and k_map_abi == "value" and mixed_profile
    k_transport = k_transport or ("tma" if combined else "cp.async")
    key_rows = rows if kv_rows is None else kv_rows
    if (include_post_qk_prefetch or include_independent_kv) and (dim!=64 or k_transport != 'tma' or k_map_abi != 'value'):
        raise ValueError('post-QK prefetch tuning requires by-value K TMA')
    geometry = None
    if overlap_schedules and dim == 128 and output_bits == 16 and fa3_query_tile == "auto":
        from fa3_query_tile import select_query_geometry
        if sm_count not in (None, 114):
            raise ValueError("automatic FA3 query tiling uses the full H100 PCIe profile")
        geometry=select_query_geometry(rows,kv_rows or rows,heads,causal,guarded_query=kv_heads not in (None,heads))
        fa3_query_tile=geometry["query_tile"]
    if k_transport == "tma":
        if not aligned_inputs or output_bits not in ((16,32) if dim==64 else (32,)) or not ((dim==64 and value_dim in (None,64)) or (grouped_d128 and k_map_abi=="value")):
            raise ValueError("TMA tuning requires aligned D64 FP16/FP32 output or grouped D128 FP32 output")
        base = AttentionPipeline("q64-kv64-k-tma",64,single_group(4,128),"producer",output_bits==16,True,kv_tile=64)
        plans = [base] if (key_rows+63)//64>=4 else []
        if k_map_abi=="value" and (key_rows+127)//128 >= 4:
            plans.append(replace(base,name="q64-kv128-k-tma",kv_tile=128))
        if (include_post_qk_prefetch or (combined and d64)) and not causal:
            plans += [replace(p, name=p.name+"-post-qk", prefetch_after_qk=True)
                      for p in plans if (key_rows+p.kv_tile-1)//p.kv_tile >= 4]
        if include_v_tma and k_map_abi == 'value' and key_rows >= 8:
            plans += [replace(p, name=p.name+'-v-tma', v_transport='tma')
                      for p in plans if (key_rows+p.kv_tile-1)//p.kv_tile >= 4]
        if (include_v_tma and k_map_abi == "value" and d64 and causal and output_bits == 32
                and kv_heads in (None, heads) and key_rows == rows and key_rows >= 8
                and (key_rows + 127) // 128 <= 2):
            plans.append(replace(base, name="q64-kv128-k-tma-v-tma", kv_tile=128, v_transport="tma"))
        if short_causal_row_major_v(rows,key_rows,heads,kv_heads or heads,dim,value_dim or dim,causal,output_bits):
            plans += [replace(p, name=p.name+"-sw128", v_row_major_tma=True)
                      for p in plans if p.query_tile == 64 and p.v_transport == "tma"]
        if (include_v_tma and k_map_abi == "value" and aligned_inputs
                and short_kv64_sw128_tma(rows, key_rows, heads, kv_heads or heads, dim,
                                          value_dim or dim, causal, output_bits)):
            plans.append(replace(base, name="q64-kv64-k-tma-v-tma-sw128",
                                 v_transport="tma", v_row_major_tma=True))
        if (d64 and not causal and output_bits == 32 and kv_heads not in (None, heads)
                and key_rows == rows and k_map_abi == "value"):
            plans += [replace(p, name=p.name.replace("q64-", "q128-", 1),
                              query_tile=128, contract=single_group(4,256))
                      for p in plans if p.v_transport == "tma"]
            # The ordinary GQA post-QK profile uses a row-major PV descriptor.
            # Preserve SW32 as a separate measurable physical alternative.
            plans += [replace(p, name=p.name+"-sw128", v_row_major_tma=True)
                      for p in plans if p.prefetch_after_qk and p.v_transport == "tma"]
        if include_independent_kv and not causal and (key_rows+63)//64 >= 4:
            # The unmodified base plan already issues both transfers before QK.
            for k_edge, v_edge in [('before','after'), ('after','before'), ('after','after')]:
                plans.append(replace(base, name=f"{base.name}-k-{k_edge}-v-{v_edge}",
                                     kv_prefetch_edges=(k_edge,v_edge)))
    else:
        plans = attention_pipelines(output_bits, overlap_schedules, native_delivery=True,
                                    aligned_inputs=aligned_inputs,include_late_k=include_late_k, fa3_query_tile=fa3_query_tile)
    if geometry and geometry["cta_dispatch"]=="reverse":
        # Q128/register already has an equivalent legacy fringe candidate.
        plans += [replace(p,name=p.name+"-dispatch-reverse",cta_dispatch="reverse") for p in plans
                  if p.contract["kind"]!="single" and (p.query_tile!=128 or p.writeback!="register")]
    if combined:
        plans += attention_pipelines(output_bits,native_delivery=True,aligned_inputs=aligned_inputs)
    cp_rings = {}
    if grouped_d128 and combined:
        for width in (64,128):
            if (key_rows+width-1)//width>=4:
                candidate=replace(base,name=f"ring{width}",kv_tile=width)
                cp_rings[candidate.name]=candidate
        plans.extend(cp_rings.values())
    if len({plan.name for plan in plans}) != len(plans):
        raise ValueError("each proposed schedule must have a distinct name")
    row_major_families = {p.name.removesuffix("-sw128") for p in plans if p.v_row_major_tma}
    candidates = []
    for plan in plans:
        path = "cp.async" if (combined and plan.kv_tile is None) or plan.name in cp_rings else k_transport
        if plan.v_row_major_tma:
            family = plan.name.removesuffix("-sw128")
        elif plan.name in row_major_families:
            family = None
        elif plan.contract["kind"] == "single" and plan.kv_tile is None:
            family = "cp-" + plan.name
        else:
            family = plan.name
        candidates.append(_PipelineCandidate(plan, path, family))
    return candidates


def bind_ranked_pipeline(receipt, rows, heads, kv_heads, dim, causal, output_bits):
    """Bind all model families to physical candidates and return the selected object."""
    candidates = pipeline_candidates(rows=rows, heads=heads, kv_heads=kv_heads, dim=dim,
                                      causal=causal, output_bits=output_bits, aligned_inputs=True)
    calibrated = [c for c in candidates if c.ranked_family is not None]
    by_family = {c.ranked_family: c for c in calibrated}
    if len(by_family) != len(calibrated):
        raise ValueError("a ranked attention family has multiple physical bindings")
    cost_key = "estimated_us" if "estimated_us" in receipt else "estimated_ratio"
    costs = receipt[cost_key]
    missing = costs.keys() - by_family.keys()
    if missing:
        raise ValueError(f"ranked families have no physical candidate: {sorted(missing)}")
    family = receipt["selected"]
    candidate = by_family[family]
    # Top-level costs, the winner, and measure(images) use physical names.
    # Nested model diagnostics keep their original calibration keys.
    names = {f: by_family[f].plan.name for f in costs}
    receipt[cost_key] = {names[f]: cost for f, cost in costs.items()}
    receipt.update(ranked_family=family, selected=candidate.plan.name, physical_by_family=names)
    return candidate, receipt
