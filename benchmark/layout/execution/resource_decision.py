"""Compiled resource Pareto evidence for bounded layout/network decisions.

Driver allocation and resource residency ceilings are independent observed axes,
not converted to dynamic traffic, measured occupancy, or hardware latency.
The extended affine certificate quantifies over unknown shared nonnegative
resource penalties. Incomparable tradeoffs keep the preregistered fallback.
"""
import copy
import hashlib
import json
from pathlib import Path
import decision_priority as priority

CONTRACT='frisk-compiled-resource-frontier-v1'
IMPLEMENTATION_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DIMENSIONS=('registers_per_thread','local_frame_bytes_per_thread',
            'shared_bytes_per_block','resource_residency_warp_deficit')


def candidate_evidence(directory, row):
    base=priority.candidate_evidence(directory,row)
    correctness=json.loads((Path(directory)/'correctness.json').read_text())
    physical=correctness['physical_resources']; confirmed=correctness['compiler_resource_confirmation']
    residency=physical['residency']; limits=physical['device_limits']
    if (confirmed.get('status')!='confirmed' or residency.get('kind')!='theoretical-resource-ceiling'
        or residency.get('api')!='cuOccupancyMaxActiveBlocksPerMultiprocessor'
        or residency.get('launchable_by_resources') is not True
        or confirmed['registers_per_thread']!=physical['registers_per_thread']
        or confirmed['local_bytes']!=physical['local_bytes']
        or confirmed['shared_bytes']!=physical['shared_bytes']):
        raise ValueError('confirmed loaded-image resource evidence required')
    maxwarps=residency['max_warps_per_sm']; active=residency['active_warps_per_sm']
    if (not priority.integer(maxwarps) or maxwarps==0 or not priority.integer(active)
        or active==0 or active>maxwarps or physical['threads_per_block']!=128
        or residency['active_blocks_per_sm']*residency['warps_per_block']!=active):
        raise ValueError('invalid resource residency ceiling')
    counts=dict(registers_per_thread=physical['registers_per_thread'],
        local_frame_bytes_per_thread=physical['local_bytes'],
        shared_bytes_per_block=physical['total_shared_bytes_per_block'],
        resource_residency_warp_deficit=maxwarps-active)
    if set(counts)!=set(DIMENSIONS) or any(not priority.integer(x) for x in counts.values()):
        raise ValueError('invalid exact resource counts')
    result=copy.deepcopy(base)
    result['payload']['compiled_resource_frontier']=dict(contract=CONTRACT,
        implementation_sha256=IMPLEMENTATION_SHA256, counts=counts,
        limits_sha256=priority.digest(limits), max_warps_per_sm=maxwarps,
        cubin_sha256=base['payload']['cubin_sha256'],
        correctness_sha256=base['payload']['bindings']['correctness.json'],
        uses_gpu_timing=False, semantics='driver allocation and theoretical ceiling; not measured occupancy or dynamic traffic')
    result['payload_sha256']=priority.digest(result['payload'])
    return result


def verify_candidate_evidence(directory,row,receipt):
    expected=candidate_evidence(directory,row)
    if receipt!=expected: raise ValueError('compiled resource evidence changed')
    return expected


def resource_rows(evidence):
    ev=priority._evidence(evidence); result={}; limits=set()
    for arm,p in ev.items():
        r=p.get('compiled_resource_frontier',{})
        if (r.get('contract')!=CONTRACT or r.get('implementation_sha256')!=IMPLEMENTATION_SHA256
            or r.get('uses_gpu_timing') is not False or r.get('cubin_sha256')!=p['cubin_sha256']
            or r.get('correctness_sha256')!=p['bindings']['correctness.json']
            or set(r.get('counts',{}))!=set(DIMENSIONS)
            or any(not priority.integer(x) for x in r['counts'].values())):
            raise ValueError('whole pool requires bound exact resource evidence')
        limits.add((r['limits_sha256'],r['max_warps_per_sm']))
        result[arm]=r['counts']
    if len(limits)!=1: raise ValueError('mixed device resource domain')
    return result


def pareto_fallback(evidence,static_order):
    ev=priority._evidence(evidence); resources=resource_rows(evidence)
    old=priority.static_fallback(evidence,static_order,enabled=True)
    members=old['payload']['primary_members']; witnesses={}
    for a in members:
        for b in members:
            if a==b: continue
            if (all(resources[b][d]<=resources[a][d] for d in DIMENSIONS)
                and any(resources[b][d]<resources[a][d] for d in DIMENSIONS)):
                witnesses[a]=b;break
    frontier=[a for a in old['payload']['ordered_members'] if a not in witnesses]
    if not frontier: raise ValueError('nonempty finite resource frontier required')
    return priority._seal(dict(schema_version=1,contract=CONTRACT,
        implementation_sha256=IMPLEMENTATION_SHA256,kind='primary-resource-frontier-fallback',
        selected_source=frontier[0],frontier=frontier,domination_witnesses=witnesses,
        counts=resources,dimensions=list(DIMENSIONS),prior_fallback=old,
        evidence={a:priority.digest(evidence[a]) for a in sorted(evidence)},
        uses_gpu_timing=False,claim='Pareto resource preference inside static primary minimum; no latency guarantee'))


def extend_affine_matrix(matrix,evidence):
    # Reuse the necessary-and-sufficient nonnegative coordinate-dominance proof.
    p=copy.deepcopy(priority._open(matrix,priority.AFFINE_CONTRACT))
    resources=resource_rows(evidence)
    if set(resources)!=set(p['rows']): raise ValueError('resource/model pool mismatch')
    dimensions=['resource:'+d for d in DIMENSIONS]
    if set(dimensions)&set(p['dimensions']): raise ValueError('duplicate affine resource dimension')
    for a,row in p['rows'].items():
        if row['status']=='available': row['counts'].update({'resource:'+d:resources[a][d] for d in DIMENSIONS})
    p['dimensions']+=dimensions
    p['coefficient_domain'].update({d:dict(lower=0,upper=None,shared_across_all_candidates=True) for d in dimensions})
    p['resource_extension']=dict(contract=CONTRACT,implementation_sha256=IMPLEMENTATION_SHA256,
        evidence={a:priority.digest(evidence[a]) for a in sorted(evidence)},
        semantics='independent nonnegative unknown penalties on allocation/ceiling; conditional family only')
    return priority._seal(p)


def select(intervals,evidence,profile,static_order,original_priority):
    fallback=pareto_fallback(evidence,static_order)
    base=priority.affine_matrix(intervals,evidence,profile)
    matrix=extend_affine_matrix(base,evidence)
    decision=priority.select_affine(matrix,fallback_arm=fallback['payload']['selected_source'],priority=original_priority)
    return dict(selected_source=decision['payload']['selected_source'],status=decision['payload']['status'],
        uses_gpu_timing=False,intervals=intervals,affine_matrix=matrix,affine_selection=decision,
        resource_fallback=fallback,rule='resource-frontier',fixed_original_priority=original_priority)
