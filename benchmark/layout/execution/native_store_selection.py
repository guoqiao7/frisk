"""Compile and select the terminal O layout of a native online kernel.

compile_native_store accepts Frisk IR with a module-level online launch.
Each output directory records the materialized IR, compiled candidates, SASS
and selection receipt. The returned image is the already-compiled winner.
"""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import re
import subprocess
import time

from backend import compile_execution
from shared_allocation import SharedBudgetExceeded
from h100_paired_mainloop import RAW_HORIZONS, paired_native_mainloop
from h100_segment_cost import NativeWavePair, attach_segments, attach_writer, attach_drain, guarded_drain, segment_work
from h100_store_schedule import H100StoreSchedule


def _store_line(ir):
    lines = [line for line in ir.splitlines() if re.match(r'\s*frisk\.tile_store\s',line)]
    if len(lines)!=1:
        raise ValueError('paired native selection requires one terminal store')
    return lines[0]


def _annotate(ir, ctx):
    line = _store_line(ir)
    operands, types = line.split(' : tensor<',1)
    if '{' in operands:
        raise ValueError('expected a clean materialized native store')
    return ir.replace(line,operands+' {'+ctx.mlir_attribute()+'} : tensor<'+types,1)


def _semantic_ir(ir):
    line = _store_line(ir)
    lhs,rhs = line.rsplit('} : tensor<',1)
    ir=ir.replace(line,lhs.split(' {',1)[0]+' : tensor<'+rhs,1)
    return re.sub(r'frisk\.fixed_store_cost = array<i64: [^>]+>, ', '', ir, count=1)


def _native_stages(ir):
    stages = []
    for line in ir.splitlines():
        if ' = frisk.mma ' not in line:
            continue
        def array(name):
            return tuple(map(int,re.search(name+r' = array<i64: ([0-9, ]+)>',line)[1].split(',')))
        m,n,k = array('atom')
        r,c,d = array('repeats')
        form = re.search(r'form = "(rs|ss)"',line)[1]
        if array('grid')!=(2,1) or (m,k)!=(64,16) or c!=1 or min(r,d)<=0:
            raise ValueError("unsupported native fixed anchor")
        stages.append((form,n,r*d))
    if [form for form,_,_ in stages] != ['ss','rs']:
        raise ValueError("expected one QK and one PV stage")
    return [(n,issues) for _,n,issues in stages]


def _scheduled_score(line):
    return list(map(int,re.search(r'frisk.scheduled_store_score = array<i64: ([^>]+)>',line)[1].split(',')))


def _device_shared_limit():
    # Offline compilation and other targets keep the packet planner's limit.
    from triton.runtime import driver
    try:
        active=driver.active
        target=active.get_current_target()
        if target.backend=='cuda' and target.arch==90:
            props=active.utils.get_device_properties(active.get_current_device())
            return props['max_shared_mem']
    except RuntimeError:
        pass
    return None


def _compile_online_image(data, group):
    try:
        return compile_execution(data, mma_group_size=group)
    except SharedBudgetExceeded:
        # A caller-declared budget is binding. Only the implicit conservative
        # default can be relaxed to the active SM90 device's opt-in capacity.
        if 'shared_limit_bytes' in data:
            raise
        limit=_device_shared_limit()
        if limit is None:
            raise
        return compile_execution(dict(data,shared_limit_bytes=limit), mma_group_size=group)


def _within_gap(direct, striped, percent):
    return 100 * abs(direct-striped) <= percent * max(direct,striped)


def _memory_gate(line, stages, rows, costs, percent):
    # Uncoalesced upper estimate at PV: old and new FP32 accumulator, plus
    # four packed A registers per RS issue. Allocation can alias both outputs.
    slots=int(re.search(r'frisk.spill_pressure = array<i64: (\d+),',line)[1])
    live=2*slots+4*stages[1][1]
    kv_tile=stages[0][0]
    warp_iterations=8*((rows+kv_tile-1)//kv_tile)
    risk_q10=1024*max(0,live-255)*warp_iterations
    near=_within_gap(*costs,percent)
    exceeds=risk_q10>0 and risk_q10>=abs(costs[0]-costs[1])
    return dict(open=near or exceeds, near=near, pressure_covers_gap=exceeds,
                transient_words=live, warp_iterations=warp_iterations, risk_q10=risk_q10)


def compile_native_store(source, opt, directory, *, sm_count, paired=True, mma_group_size=1, cuobjdump='cuobjdump', paired_gap_percent=None, pair_policy="segments"):
    """Compile the frozen native store with independent or paired scoring.

    When a percentage is supplied the memory policy also admits pairs whose
    projected PV pressure covers the cost gap. The legacy policy gates only
    on the percentage. Use memory64 for the previous single-horizon score
    under the same pressure gate. The segments policy also accounts for cache carveout
    and scalar work in the writeback suffix. All modes honor the caller's Triton cache configuration."""
    if type(mma_group_size) is not int or mma_group_size not in (1, 4):
        raise ValueError("native WGMMA group size must be 1 or 4")
    if paired_gap_percent is not None and (not paired or type(paired_gap_percent) is not int or not 0 <= paired_gap_percent <= 100):
        raise ValueError('paired_gap_percent requires paired=True and an integer in [0,100]')
    if pair_policy not in ('wave','drain','writer','segments','memory','memory64','legacy'):
        raise ValueError("unknown native pair_policy")
    start=time.perf_counter()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    base=directory/'base.mlir'
    def run(src,dst,*args):
        subprocess.run([str(opt),str(src),*args,'-verify-each','-o',str(dst)],check=True)
    from online_execution_contract import single_group, bind_module
    input_ir=Path(source).read_text()
    if 'frisk.execution_contract' not in input_ir:
        source=directory/'contract-input.mlir'
        source.write_text(bind_module(input_ir,single_group(mma_group_size)))
    run(source,base,'--frisk-layout-pipeline')
    ir=base.read_text()
    launch=re.search(r'frisk\.online_launch = \{([^}]+)\}',ir)
    if not launch:
        raise ValueError('expected an online launch')
    def integer(key):
        return int(re.search(r'\b'+key+r' = (\d+) : i64',launch[1])[1])
    rows,heads=integer('rows'),integer('heads')
    kv_match=re.search(r'\bkv_rows = (\d+) : i64',launch[1])
    kv_rows=int(kv_match[1]) if kv_match else rows
    stages=_native_stages(ir)
    issues=sum(count for _,count in stages)
    ctx=H100StoreSchedule(rows,heads,sm_count,issues,issues//mma_group_size,False)
    line=_store_line(ir)
    source_map=re.search(r'tensor<\d+x\d+xf32, (#frisk\.distributed<.*>)>, memref<',line)
    if not source_map:
        raise ValueError('expected a materialized f32 store source')
    scheduled=directory/'scheduled.mlir';scheduled.write_text(_annotate(ir,ctx))
    flags='--frisk-protected-store-layout=fixed-anchors=true'
    gate_score=gate_details=static_choice=None
    if paired_gap_percent is not None:
        static=directory/'static.mlir'
        run(scheduled,static,flags)
        static_line=_store_line(static.read_text())
        gate_score=_scheduled_score(static_line)[-2:]
        static_choice='striped' if gate_score[1]<gate_score[0] else 'direct'
        if pair_policy!='legacy':
            gate_details=_memory_gate(static_line,stages,kv_rows,gate_score,paired_gap_percent)
            paired=gate_details['open']
        else:
            paired=_within_gap(*gate_score,paired_gap_percent)
    images={};metadata={};times={};listing={}
    def compile_arm(arm):
        before=time.perf_counter();folder=directory/arm;folder.mkdir(exist_ok=True)
        packet=folder/'execution.json';module=folder/'candidate.mlir'
        export='--frisk-export-execution=output='+str(packet)+' minify=true'
        if gate_score is not None and (arm=='auto' or (pair_policy!='legacy' and arm==static_choice)):
            run(static,module,export)
        else:
            run(scheduled,module,flags+' materialize-store='+arm,export)
        mid=time.perf_counter()
        data=json.loads(packet.read_text())
        image,info=_compile_online_image(data,mma_group_size)
        finished=time.perf_counter()
        cubin=folder/'kernel.cubin';cubin.write_bytes(image.asm['cubin'])
        images[arm]=image;metadata[arm]=info
        times[arm]={'materialize_export_s':mid-before,'read_emit_compile_s':finished-mid}
        (folder/'meta.json').write_text(json.dumps(info))
    budget_limited=False
    for arm in (('direct','striped') if paired else ('auto',)):
        try:
            compile_arm(arm)
        except SharedBudgetExceeded:
            failed_line=_store_line((directory/arm/'candidate.mlir').read_text())
            if not re.search(r'frisk.store_score = array<i64: [^>]+, 1>',failed_line):
                raise
            budget_limited=True
            if 'direct' not in images:
                compile_arm('direct')
            paired=False
            break
    if budget_limited:
        selected=directory/'direct/candidate.mlir'
    elif paired:
        for arm in images:
            folder=directory/arm
            listing[arm]=subprocess.check_output([cuobjdump,'-sass',str(folder/'kernel.cubin')],text=True)
            (folder/'kernel.sass').write_text(listing[arm])
        if metadata['direct']['native_online_launch']!=metadata['striped']['native_online_launch']:
            raise ValueError('candidate launch mismatch')
        info=metadata['direct']
        pair=paired_native_mainloop(listing['direct'],listing['striped'],source=source_map[1],
                                    kv_tiles=info['kv_tiles'],warps=info['threads']//32,
                                    mma_group_size=mma_group_size, stages=stages, memory=pair_policy!='legacy',
                                    long_loads=pair_policy in ('wave','drain','writer','segments','memory'))
        if pair_policy in ('wave','drain','writer','segments'):
            segment_features = {}
            for arm in ('direct','striped'):
                resource = subprocess.check_output([cuobjdump,'-res-usage',str(directory/arm/'kernel.cubin')],text=True)
                segment_features[arm] = segment_work(listing[arm],resource,metadata[arm]['dynamic_shared_bytes'],info['threads'],writer=pair_policy in ('wave','drain','writer'))
            pair = attach_segments(pair,segment_features['direct'],segment_features['striped'],info['threads']//32)
            if pair_policy in ('wave','drain','writer'):
                pair = attach_writer(pair,segment_features['direct'],segment_features['striped'],info['threads']//32)
        if pair_policy in ('wave','drain'):
            # Packed full tiles carry no output row guard in this native emitter.
            # Use zero feedback when either SASS cannot certify the masked suffix.
            drains = {arm:guarded_drain(listing[arm]) if rows%info['q_tile'] else None
                      for arm in ('direct','striped')}
            pair = attach_drain(pair,drains['direct'],drains['striped'],info['threads']//32)
            if pair_policy=='wave' and all(d is not None for d in drains.values()):
                pair = NativeWavePair(**vars(pair))
        feedback=directory/'feedback.mlir'
        feedback.write_text(_annotate(ir,replace(ctx,paired_mainloop=pair)))
        selected=directory/'selected.mlir';run(feedback,selected,flags)
    else:
        selected=directory/'auto/candidate.mlir'
    final_ir=selected.read_text();line=_store_line(final_ir)
    change=int(re.search(r'frisk.store_score = array<i64: [^>]+, ([01])>',line)[1])
    choice='striped' if change else 'direct'
    key=choice if paired or budget_limited else 'auto'
    if paired and _semantic_ir(final_ir)!=_semantic_ir((directory/key/'candidate.mlir').read_text()):
        raise ValueError('selected materialized graph differs from the compiled candidate')
    receipt={'mma_group_size':mma_group_size,'choice':choice,'phases':times,'compile_wall_s':time.perf_counter()-start,
             'winner_sha256':hashlib.sha256(images[key].asm['cubin']).hexdigest(),
             'score':_scheduled_score(line),'stages':stages,'paired':paired,'pair_policy':pair_policy,'striped_over_budget':budget_limited}
    if gate_score is not None:
        receipt.update(static_costs=gate_score,paired_gap_percent=paired_gap_percent)
    if paired:
        receipt.update(warp_issues_delta=pair.warp_issues_delta,warp_imads_delta=pair.warp_imads_delta,
                       mainloop_delta_q10=int(re.search(r'frisk.mainloop_delta_q10 = (-?\d+)',line)[1]))
    if gate_details is not None:
        receipt['gate']=gate_details
    if paired and pair_policy!='legacy':
        receipt['loop_memory']={'direct':asdict(pair.direct),'striped':asdict(pair.striped),
                                'warp_iterations':pair.warp_iterations,
                                'warp_load_wait_delta':pair.warp_load_wait_delta,
                                'warp_local_word_delta':pair.warp_local_word_delta}
        receipt['loop_memory']['raw_horizons']=list(RAW_HORIZONS) if pair_policy in ('wave','drain','writer','segments','memory') else [64]
    if paired and pair_policy in ('wave','drain','writer','segments'):
        receipt['segments']={arm:asdict(v) for arm,v in segment_features.items()}
        receipt['segments'].update(cache_miss_sectors_delta=pair.cache_miss_sectors_delta,
                                   tail_address_issues_delta=pair.tail_address_issues_delta,
                                   delta_q10=int(re.search(r'frisk.segment_delta_q10 = (-?\d+)',line)[1]))
    if paired and pair_policy in ('wave','drain','writer'):
        banks = re.search(r'frisk.writer_bank_extra = array<i64: (\d+), (\d+)>',line)
        receipt['writer'] = dict(tail_miss_sectors=pair.tail_cache_miss_sectors_delta,
                                 publish_extra=int(banks[1]), gather_extra=int(banks[2]),
                                 cost_q10=int(re.search(r'frisk.writer_delta_q10 = (-?\d+)',line)[1]))
    if paired and pair_policy in ('wave','drain'):
        receipt['drain']={arm:asdict(v) if v is not None else None for arm,v in drains.items()}
        receipt['drain'].update(address_delta=pair.drain_address_issues_delta,
                                interstore_delta=pair.drain_interstore_issues_delta,
                                cost_q10=int(re.search(r'frisk.drain_delta_q10 = (-?\d+)',line)[1]))
    if paired and pair_policy=='wave':
        receipt['wave'] = None
        if isinstance(pair,NativeWavePair):
            traffic = re.search(r'frisk.wave_sectors = array<i64: ([^>]+)>',line)
            receipt['wave'] = dict(sectors=[int(x) for x in traffic[1].split(', ')],
                                   cost_q10=int(re.search(r'frisk.wave_delta_q10 = (-?\d+)',line)[1]))
    (directory/'receipt.json').write_text(json.dumps(receipt,indent=2))
    return images[key],metadata[key],receipt
