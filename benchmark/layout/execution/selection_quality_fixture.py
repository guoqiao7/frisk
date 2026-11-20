"""Preregistered whole-program holdouts for bounded model/selection qualification.
This module has no CUDA or subprocess side effects.  Frozen prior coefficients
are evaluated without fitting or threshold changes from these holdout results.
"""
import hashlib
import json
import re
import numpy as np
from calibration_fixture import fixture as calibration_program, inputs
from support_closure_fixture import (PROFILE_RELATIVE, PROFILE_SHA256, GPU_UUID,
    numerical_cases, reference, offsets)
from support_closure_fixture import check_numerics as previous_check_numerics

SEED = 20261102


def program_identity(spec):
    return (spec["case_id"], tuple(spec["shape"]), spec.get("tail", "none"),
            spec.get("abi_stratum", "weak-vector"), spec.get("dtype", "float32"),
            spec.get("threads", 128))


def historical_specifications():
    """The registered fixture inventory; all previous holdouts are now observed."""
    from calibration_fixture import specifications as calibration
    from resource_repair_fixture import specifications as resource
    from support_closure_fixture import specifications as closure
    from joint_fixture import cases as joint
    rows = calibration() + resource() + closure()
    rows += [dict(case_id=x["id"], shape=x["shape"], tail="none",
                  abi_stratum="weak-vector", dtype="float32", threads=128)
             for x in joint()]
    unique = {program_identity(x): x for x in rows}
    return [dict(case_id=k[0], shape=list(k[1]), tail=k[2], abi_stratum=k[3],
                 dtype=k[4], threads=k[5]) for k in sorted(unique)]


def specifications():
    def item(name, case, shape, tail="none"):
        return dict(key="holdout-" + name + "-" + str(shape[0]) + "x" + str(shape[1]),
            split="holdout", role="new-whole-program-holdout", case_id=case,
            shape=list(shape), tail=tail, abi_stratum="weak-vector", dtype="float32",
            threads=128, graph_name=name)
    rows = [item("diamond", "diamond", (64,32)),
            item("remat", "remat", (64,32)),
            item("conflict", "conflict", (128,16)),
            item("mixed", "no-conflict", (16,64), "mixed"),
            item("div", "no-conflict", (32,64), "div6"),
            item("exp", "no-conflict", (64,32), "exp1")]
    for abi in ("weak-vector", "strong-cpasync", "strong-tma"):
        rows.append(dict(key="anchor-attention-"+abi, split="observed-anchor",
            role="previously-observed-functional-and-timing-anchor", case_id="attention",
            shape=[64,16], tail="none", abi_stratum=abi, dtype="float16", threads=128,
            graph_name="attention"))
    validate_specifications(rows)
    return rows


def validate_specifications(rows):
    keys=[x["key"] for x in rows]
    identities=[program_identity(x) for x in rows]
    if len(keys)!=len(set(keys)) or len(identities)!=len(set(identities)):
        raise ValueError("duplicate whole-program specification")
    historical={program_identity(x) for x in historical_specifications()}
    for row in rows:
        if row["split"]=="holdout" and program_identity(row) in historical:
            raise ValueError("previously observed program cannot become a new holdout")
        if row["split"]=="observed-anchor" and program_identity(row) not in historical:
            raise ValueError("observed anchor must be in the historical inventory")
    return True


def source(spec, root, controlled=False):
    if spec["case_id"]=="attention":
        if controlled: raise ValueError("Attention anchors use natural domains only")
        suffix={"weak-vector":"fixed", "strong-cpasync":"cp-async",
                "strong-tma":"tma"}[spec["abi_stratum"]]
        return (root/("test/Transforms/attention-layout-"+suffix+"-abi.mlir")).read_text()
    text=calibration_program(spec)
    if controlled: return text
    text=text.replace(",joint.A=#A,joint.B=#B", "")
    return "\n".join(line for line in text.splitlines()
        if not re.match(r"#(?:A|B)=", line))+"\n"


def check_numerics(spec, got, expected):
    # exp1 and mixed keep the same previously registered transcendental tolerance.
    # The closure stage had no exp1 rows, so its bit-exact default needs this
    # explicit extension before this new stage's exp1 program is executed.
    if spec["case_id"]!="attention" and spec["tail"] in ("exp1", "mixed"):
        proxy=dict(spec, tail="mixed")
        return previous_check_numerics(proxy,got,expected)
    return previous_check_numerics(spec,got,expected)


def safety_input_identity(spec, info, abi):
    cases=[dict(name=name, inputs=[hashlib.sha256(x.tobytes()).hexdigest()
                                  for x in (q,k,v)])
           for name,q,k,v in numerical_cases(spec)[:1]]
    return dict(spec=spec, cases=cases, offsets=offsets(spec,info["copy_path"]),
        physical_abi=abi, repeats=1, guard_byte=165, threads=128, grid=[1,1,1],
        default_stream=True)


def protocol():
    return dict(schema_version=1, stage="model-selection-quality-qualification", seed=SEED,
        specifications=specifications(), historical_inventory=historical_specifications(),
        split_rule="entire case/shape/tail/ABI/dtype/thread program; all candidate and model variants share a split",
        freshness="six new complete-program combinations against the registered historical fixture inventory; dimensions may have occurred in other programs; Attention is observed anchor only",
        models=["legacy", "calibrated"],
        model_ids=dict(legacy="sm90-layout-lex9-legacy-v2", calibrated="sm90-layout-calibrated-v3-<profile-hash32>-leg"),
        profile=dict(path=PROFILE_RELATIVE,sha256=PROFILE_SHA256,
                     format="sm90-layout-calibrated-v2",semantics="calibrated-v3",fit=False),
        experiment=dict(conversion="C", selections=["LI","J"],
            controlled="two producer candidates A/B; all other variables singleton; exact finite-domain oracle and force A/B replay",
            natural="unchanged initialized candidate domain; LI/J compare the identical frozen domain and budget; empirical timing reference is the union of four chosen legacy/calibrated LI/J images, not the full raw-candidate optimum",
            oracle_scope="exact exhaustive model oracle only in controlled two-candidate domains; no natural full-domain optimum or global search-quality claim",
            domain_equality="legacy/calibrated domain and coverage digests must match; model/package identity remains distinct",
            informed_local="LI-C includes own producer arithmetic/access and required input repair; no future-consumer information",
            excluded="LI-O is not exercised because the known entire-domain own-repair restriction is not qualified"),
        budget=dict(exact_combination_limit=256,beam_width=32,max_expanded_states=65536,
                    refine_beam_solution=True,max_refinement_probes=256),
        measurement=dict(batches=10,warmup=100,pilot_nodes=100,target_graph_ms=5,
            max_nodes=4096,min_nodes=1,replays=5,bootstrap_resamples=10000,
            seed=SEED,confidence="paired percentile nominal 95% descriptive intervals; ranking uses Bonferroni over unique-image pairs within each program; promotion per-case guard uses familywise alpha=.05 over the nine natural programs (Bonferroni .05/9)",
            foreign_pid_rule="invalidate entire paired batch; retain every invalid record and never replace",
            aliases="one measured observation per exact cubin/config/ABI identity within a program and batch; retain all selected-arm aliases; identical images are not independent samples"),
        acceptance=dict(
            functional="all predeclared executable candidates and chosen arms must pass; unsupported or failed candidates remain in coverage denominators",
            model_oracle="controlled J must equal the exhausted full-graph model optimum; LI stage1 and fixed-producer stage2 checked against their respective exhausted oracles",
            bounded_selection_quality=dict(geomean_regret_max=1.03,max_case_regret_max=1.10,
                domain="controlled: exhaustive two producer choices; natural: union of four chosen legacy/calibrated LI/J images, not full raw-candidate or unrestricted GPU optimum",
                promotion_mandatory="calibrated J must satisfy the bounded regret gate in both controlled and natural suites",
                confidence="report paired-bootstrap intervals and distinguish threshold pass, fail, and unresolved uncertainty"),
            calibrated_promotion=dict(geomean_calibrated_over_legacy_ci95_upper_lt=1.0/1.03,
                primary="natural J calibrated/legacy timing on the six fresh whole-program holdouts only; Attention anchors excluded from primary aggregate",
                aggregation="fixed-suite geometric mean with independent paired-block bootstrap within each program; no new-program generalization interval",
                minimum_practical_speedup=1.03,
                guard="all six fresh natural programs plus all three observed Attention ABI anchors; familywise per-case upper bounds use Bonferroni alpha=.05/9",
                guard_familywise_alpha=0.05,guard_comparison_count=9,
                per_case_calibrated_over_legacy_ci95_upper_max=1.05,
                requirements="complete coverage; same algorithm/ABI/precision/domain/budget; bounded regret gate; no hard-resource violation; otherwise retain legacy"),
            joint_superiority="report J/LI paired timing with uncertainty; no mandatory speedup and no superiority claim from a model-cost gap alone"),
        resources=dict(shared="structural pre-native plan must equal exact compiled/loaded shared bytes",
            physical_registers="compiler/driver feedback only; zero-error identity confirmation; static predictor remains not-qualified",
            hard_limits="preserve compiled resource rejection, compile cost, measurement cost and failed candidates; no silent infeasible-candidate deletion",
            local="record driver local bytes; spill count is unknown without explicit native metadata"),
        numerics=dict(ordinary="finite uint32 bit equality including signed zero",
            exp1_and_mixed="finite abs(error)<=1e-5+2e-6*abs(reference)",
            attention="finite abs(error)<=.002+.002*abs(reference)",
            inputs="unchanged support-closure independent inputs/reference/offsets; seven f32 cases and fifteen Attention cases; timing f32 batch i uses calibration_fixture.inputs(spec,i), seed 20261031+i for i=0..9"),
        safety=dict(tools=["memcheck","racecheck","synccheck","initcheck"],
            reuse="exact cubin bytes + complete emission config + physical ABI + identical spec and safety input protocol"),
        execution=dict(gpu_uuid=GPU_UUID,device_ordinal=1,grid=[1,1,1],threads=128,
            entry="frisk_execution",num_warps=4,enable_fp_fusion=False,
            output_aliasing="O disjoint from Q/K/V; read-only Q/K/V may overlap"),
        holdout_rule="freeze code/profile/protocol before final holdout measurements; no fitting, threshold changes or holdout-based model repair; prospective failures remain results",
        scope="bounded single-CTA SM90 checked backend; no universal latency prediction or cross-framework performance claim")


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
