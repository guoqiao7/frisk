"""Preregistered crossed seed/input-offset/output-offset probe of two sealed images.

Only preparation and integrity checks run on CPU. `measure` reuses the accepted
selection_quality CUDA/graph/numerical/guard loop and never compiles a kernel.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import network_selection_fixture as numerical
import selection_quality as quality
import selection_refinement_decision as evidence

CONTRACT='known-div-independent-abi-offset-cross-v1'
BATCH_CONTRACT='explicit-preregistered-batches-v1'
KEY='holdout-network-div-512x128'
GROUP='known-div-512x128-offset-cross'
ARMS=('joint-c0','joint-c2')
CUBINS={'joint-c0':'3d2a809da3bff774b7740f08ac936bed6c1c6c496a2e41ad0fba22a80c842ebd',
        'joint-c2':'87c77155f355c6b40c6375e98bbceb8a009a990280935a7e62b0f08da3731532'}
OLD_RUN=Path('/data0/baopeihua-frisk-network-20261105/run-network-selection-r3')
SEEDS=(0,1,2)
OFFSETS=(1,2,3)
REPEATS=4
RANDOM_SEED=20261105


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def input_contents(spec):
    result={}
    for seed in SEEDS:
        arrays=numerical.inputs(spec,seed);reference=numerical.reference(spec,*arrays)
        result[str(seed)]=dict(input_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
                              reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest())
    return result


def schedule(images,contents,random_seed=RANDOM_SEED):
    if type(random_seed) is not int or not 0<=random_seed<(1<<31):raise ValueError('bounded integer randomization seed required')
    if set(images)!=set(ARMS) or set(contents)!={str(n) for n in SEEDS}:raise ValueError('complete fixed image/input seed identities required')
    cells=[(seed,io,oo) for seed in SEEDS for io in OFFSETS for oo in OFFSETS]
    rng=random.Random(random_seed);start_order={cell:rng.randrange(2) for cell in cells};rows=[]
    for repeat in range(REPEATS):
        order=list(cells);rng.shuffle(order)
        for seed,io,oo in order:
            arm_order=list(ARMS if (repeat+start_order[seed,io,oo])%2==0 else reversed(ARMS))
            metadata=dict(contract=CONTRACT,cell_id=f'seed{seed}-io{io}-oo{oo}',input_seed=seed,
                replicate_block=repeat,grid=[1,1,1],threads=128,gpu_uuid=numerical.GPU_UUID,
                images=images,input_content=contents[str(seed)])
            rows.append(dict(input_batch=seed,input_offset=io,output_offset=oo,
                             metadata=metadata,arm_order=arm_order))
    return rows


def validate_plan(plan):
    fields={'schema_version','contract','role','spec','images','input_contents','random_seed','schedule',
            'provenance','measurement','statistical_unit'}
    if (not isinstance(plan,dict) or set(plan)!=fields or type(plan['schema_version']) is not int
        or plan['schema_version']!=1 or plan['contract']!=CONTRACT
        or plan['role']!='known-image-confounding-probe-not-fresh-qualification'
        or plan['statistical_unit']!='whole paired block; event samples are dependent repetitions'
        or plan['spec'].get('key')!=KEY or plan['spec'].get('shape')!=[512,128]
        or plan['spec'].get('abi_stratum')!='weak-vector' or plan['spec'].get('threads')!=128
        or plan['spec'].get('dtype')!='float32' or plan['spec'].get('tail')!='div6'):
        raise ValueError('fixed known div probe contract differs')
    if plan['schedule']!=schedule(plan['images'],plan['input_contents'],plan['random_seed']):
        raise ValueError('crossed seed/offset/balanced-order schedule differs')
    expected=dict(batches=108,warmup=100,pilot_nodes=100,target_graph_ms=5,max_nodes=4096,
        min_nodes=1,replays=5,bootstrap_resamples=10000,seed=plan['random_seed'],
        foreign_pid_rule='invalidate whole paired block; retain and never replace',
        confidence='diagnostic paired block intervals; no promotion or holdout claim',
        batch_provider_contract=BATCH_CONTRACT,batch_plan={GROUP:plan['schedule']})
    if plan['measurement']!=expected:raise ValueError('registered measurement/repetition budget differs')
    return plan


def sealed_rows():
    seal=evidence.SealedInputs();manifest=seal.read(OLD_RUN/'manifest.json');registration=seal.read(OLD_RUN/'registration.json')
    rows=[];images={}
    for arm in ARMS:
        matches=[row for row in manifest['rows'] if row['spec']['key']==KEY and row['arm']==arm]
        if len(matches)!=1:raise ValueError('exact known source row required')
        row=copy.deepcopy(matches[0]);directory=Path(row['path'])
        for name,claimed in row['artifact_hashes'].items():
            if seal.check(directory/name)!=claimed:raise ValueError('sealed image manifest differs')
        correctness=quality.closure.verify_accepted_row(directory,row)
        if row['cubin_sha256']!=CUBINS[arm] or correctness['cubin_sha256']!=CUBINS[arm]:raise ValueError('fixed known CUBIN differs')
        if len(row['safety'])!=4 or {item['tool'] for item in row['safety']}!={'memcheck','racecheck','synccheck','initcheck'} or any(item['exit_code']!=0 for item in row['safety']):
            raise ValueError('original exact image lacks four accepted sanitizer records')
        row['resource_comparison']=correctness['resource_comparison'];rows.append(row)
        images[arm]=dict(cubin_sha256=row['cubin_sha256'],execution_config_sha256=digest(read(directory/'execution_config.json')),
            image_identity_sha256=row['image_identity_sha256'],execution_identity=row['execution_identity'],
            physical_abi_sha256=digest(correctness['physical_params']))
    if rows[0]['spec']!=rows[1]['spec']:raise ValueError('common known algorithm/shape required')
    provenance=dict(run=str(OLD_RUN),manifest_sha256=seal.check(OLD_RUN/'manifest.json'),
        registration_sha256=seal.check(OLD_RUN/'registration.json'),source=registration['source'],
        old_tools_digest=digest(registration['tools']),rows=[dict(arm=r['arm'],path=r['path'],correctness_sha256=r['correctness_sha256'],
        image_identity_sha256=r['image_identity_sha256'],cubin_sha256=r['cubin_sha256']) for r in rows])
    return seal,rows,images,provenance


def prepare(output,random_seed=RANDOM_SEED):
    seal,rows,images,provenance=sealed_rows();spec=rows[0]['spec'];contents=input_contents(spec);batches=schedule(images,contents,random_seed)
    measurement=dict(batches=len(batches),warmup=100,pilot_nodes=100,target_graph_ms=5,max_nodes=4096,min_nodes=1,replays=5,
        bootstrap_resamples=10000,seed=random_seed,foreign_pid_rule='invalidate whole paired block; retain and never replace',
        confidence='diagnostic paired block intervals; no promotion or holdout claim',
        batch_provider_contract=BATCH_CONTRACT,batch_plan={GROUP:batches})
    plan=dict(schema_version=1,contract=CONTRACT,role='known-image-confounding-probe-not-fresh-qualification',
        spec=spec,images=images,input_contents=contents,random_seed=random_seed,schedule=batches,
        provenance=provenance,measurement=measurement,
        statistical_unit='whole paired block; event samples are dependent repetitions')
    validate_plan(plan);out=evidence._fresh_output(output);write(out/'plan.json',plan)
    write(out/'cpu-preparation.json',dict(status='prepared',plan_sha256=evidence.sha(out/'plan.json'),
        old_inputs_checked=seal.checked,gpu_initialized=False,native_compiler_called=False,
        cells=27,paired_blocks=108,candidate_observations=216,orders_per_cell={'AB':2,'BA':2}))
    return out/'plan.json'


def init(run,plan_path):
    from cost_calibration import source_identity
    plan_path=Path(plan_path).resolve();plan=validate_plan(read(plan_path));seal,rows,images,provenance=sealed_rows()
    if plan['images']!=images or plan['provenance']!=provenance or plan['spec']!=rows[0]['spec'] or plan['input_contents']!=input_contents(rows[0]['spec']):
        raise ValueError('registered image/input provenance changed')
    run=evidence._fresh_output(run)
    protocol=dict(schema_version=1,stage=CONTRACT,role=plan['role'],specifications=[plan['spec']],
        measurement=plan['measurement'],image_provenance=provenance,plan_sha256=evidence.sha(plan_path),
        claim='crossed known-image ABI/input diagnostic; no new layout or performance qualification')
    write(run/'protocol.json',protocol);write(run/'measurement-aliases.json',{GROUP:{arm:dict(arm=arm,execution_identity=images[arm]['execution_identity']) for arm in ARMS}})
    immutable=dict(seal.checked);immutable[str(seal.path)]=evidence.KNOWN_INDEX_SHA256;immutable[str(plan_path)]=evidence.sha(plan_path)
    immutable[str(run/'measurement-aliases.json')]=evidence.sha(run/'measurement-aliases.json')
    # This freezes the current measurement harness. Old source/tools above are
    # image provenance only; old-source compatibility is never granted to caches.
    tools=quality.closure.frozen_tools(quality.OPT)
    registration=dict(created_ns=time.time_ns(),source=source_identity(),tools=tools,
        optimizer=str(quality.OPT),protocol_sha256=evidence.sha(run/'protocol.json'),
        protocol_digest=digest(protocol),immutable_inputs=immutable,plan=str(plan_path))
    write(run/'registration.json',registration)
    write(run/'manifest.json',dict(schema_version=1,status='validated',registration_sha256=evidence.sha(run/'registration.json'),rows=rows))
    return run


def verify(run):
    run=Path(run);protocol=quality.verify_protocol(run);registration=read(run/'registration.json')
    plan=validate_plan(read(registration['plan']));seal,rows,images,provenance=sealed_rows()
    if (protocol['stage']!=CONTRACT or protocol['measurement']!=plan['measurement']
        or protocol['image_provenance']!=provenance or plan['images']!=images
        or plan['input_contents']!=input_contents(rows[0]['spec'])
        or read(run/'manifest.json')['rows']!=rows):raise ValueError('probe inventory or registered schedule changed')
    return plan,{GROUP:rows}


def provider(plan):
    validate_plan(plan)
    def batch(spec,index):
        if spec!=plan['spec']:raise ValueError('probe cannot change specification')
        if type(index) is not int or not 0<=index<len(plan['schedule']):raise ValueError('probe batch index differs')
        return copy.deepcopy(plan['schedule'][index])
    return batch


def measure(run,output):
    plan,groups=verify(run)
    return quality.measure(Path(run),Path(output),suite=numerical,groups=groups,batch_provider=provider(plan))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','init','verify','measure'))
    parser.add_argument('directory',type=Path);parser.add_argument('--plan',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--random-seed',type=int,default=RANDOM_SEED);args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.directory,args.random_seed)
    elif args.command=='init':
        if args.plan is None:parser.error('init requires --plan')
        result=init(args.directory,args.plan)
    elif args.command=='measure':
        if args.output is None:parser.error('measure requires --output')
        result=measure(args.directory,args.output)
    else:result=verify(args.directory)[0]['contract']
    print(str(result))


if __name__=='__main__':main()
