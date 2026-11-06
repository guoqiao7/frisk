"""Re-score all previously measured pure-f32 images with the v6 static ledger.
No holdout timing, no fresh assignment search, no changes to sealed old images.
"""
import argparse,json,math,re,shutil,sys,time
from pathlib import Path
import numpy as np
from support_closure import sha,read,write,fresh,invoke
from native_cost_calibration import ROOT,OLD_RUNS,communication_floors
from demand_repair import BENCH,bridge_assignment
from calibration_analysis import FEATURES as OLD_FEATURES,coordinate_fit
FEATURES=tuple(OLD_FEATURES)+('shared_read_bank_excess','shared_write_bank_excess',
    'shared_warp_issues','shuffle_warp_issues','communication_alu_warp_issues')
# Communication instructions rather than byte+instruction duplicate channels.
FIXED_ZERO=('shared_bank_excess','conversion_register_bytes','conversion_shuffle_bytes',
            'conversion_shared_bytes','conversion_conservative_bytes')

def mark_execution_domain(text,spec):
    if spec['case_id']=='attention' or re.search(r'\bfrisk\.execution_domain\s*=',text):return text
    marker='module attributes {'
    if text.count(marker)!=1:raise ValueError('module attribute block missing')
    return text.replace(marker,marker+'frisk.execution_domain="sm90-scalar-f32-dag-v1",',1)


def collect(out):
    out=fresh(out);unit={name:(0 if name in FIXED_ZERO else 1) for name in FEATURES}
    write(out/'unit-profile.json',dict(schema_version=1,model='sm90-layout-surviving-calibrated-v1',coefficients=unit,provenance='all-observed exact-image v6 feature extraction; not fitted'))
    rows=[];prepared={}
    for run in OLD_RUNS:
        freeze=read(run/'measurement/freeze.json');completion=read(run/'measurement/completion.json')
        if completion['status']!='complete' or completion['freeze_sha256']!=sha(run/'measurement/freeze.json'):raise ValueError('old freeze changed')
        for group,images in freeze['groups'].items():
            rawpath=run/'measurement'/(group+'-raw.json');raw=read(rawpath)
            if sha(rawpath)!=completion['evidence_hashes'][rawpath.name]:raise ValueError('old timing changed')
            for image in images:
                spec=image['spec']
                if spec['case_id']=='attention':continue
                source=Path(image['path']);oldprep=Path(image['prepared']);key=(str(run),group)
                if key not in prepared:
                    base=fresh(out/'preparations'/(run.name+'-'+group));inputfile=base/'input.mlir';inputfile.write_text(mark_execution_domain((oldprep/'input.mlir').read_text(),spec));prep=base/'prepared'
                    args=[BENCH,'--prepare',inputfile,spec['case_id'],prep,'--cost-profile='+str(out/'unit-profile.json')]
                    if image['scope']=='natural':args+=['--natural']
                    result=invoke(args,base/'prepare.log')
                    if result['exit_code']:raise RuntimeError('v6 prepare failed')
                    meta=read(prep/'prepared.json');oldmeta=read(oldprep/'prepared.json')
                    if any(meta[n]!=oldmeta[n] for n in ('common_digest','domain_digest','coverage_digest')) or (prep/'domains.json').read_bytes()!=(oldprep/'domains.json').read_bytes():raise ValueError('known domain changed')
                    prepared[key]=(base,prep,meta)
                base,prep,meta=prepared[key];target=out/'images'/(run.name+'-'+group+'-'+image['arm']);oldrecord=read(source/'record.json');assignment=base/(image['arm']+'-assignment.json');write(assignment,bridge_assignment(oldrecord,meta))
                result=invoke([BENCH,'--prepared='+str(prep),'--case='+spec['case_id'],'--selection=J','--conversion=C','--assignment='+str(assignment),'--cost-profile='+str(out/'unit-profile.json'),'--seed-beam-solution=false','--demand-ledger','--output-dir='+str(target)],base/(image['arm']+'.log'))
                if result['exit_code']:raise RuntimeError('v6 assignment replay failed')
                record=read(target/'record.json')
                if record['complete_assignment_sha256']!=oldrecord['complete_assignment_sha256'] or record.get('selection_bypassed') is not True:raise ValueError('known replay assignment drift')
                from backend import emit
                tt,config=emit(read(target/'execution.json'))
                if tt.encode()!=(source/'kernel.ttgir').read_bytes() or json.dumps(config,indent=2).encode()!=(source/'execution_config.json').read_bytes():raise ValueError('known native emission bytes drift')
                if sha(source/'kernel.cubin')!=image['cubin_sha256']:raise ValueError('known cubin drift')
                f=record['cost']['features'];features={n:f.get(n,0) for n in FEATURES}
                if set(f)-set(FEATURES) or any(type(v) is not int or v<0 for v in features.values()):raise ValueError('v6 feature schema mismatch')
                selected=sorted((r for r in raw if r['arm']==image['arm']),key=lambda r:r['batch'])
                if len(selected)!=10 or [r['batch'] for r in selected]!=list(range(10)) or any(not r['valid'] or r['cubin_sha256']!=image['cubin_sha256'] for r in selected):raise ValueError('known labels incomplete')
                med=[float(np.median(r['samples_ms']))*1000 for r in selected]
                rows.append(dict(source_run=str(run),group=group,spec=spec,path=str(target),old_path=str(source),assignment=record['complete_assignment_sha256'],record_sha256=sha(target/'record.json'),execution_sha256=sha(target/'execution.json'),features=features,cubin_sha256=image['cubin_sha256'],median_us=float(np.median(med)),batch_medians_us=med,timing_path=str(rawpath),timing_sha256=sha(rawpath),exact_native_bytes_reused=True))
                write(out/'manifest.json',dict(status='collecting',rows=rows))
    write(out/'manifest.json',dict(status='complete',rows=rows,no_holdout_use=True,unit_profile_sha256=sha(out/'unit-profile.json')))
    return out


def fit(training,micro,out):
    training=Path(training);micro=Path(micro);manifest=read(training/'manifest.json');rows=manifest['rows']
    if manifest['status']!='complete' or manifest.get('no_holdout_use') is not True:raise ValueError('complete known collection required')
    completion=read(micro/'completion.json')
    if completion['status']!='complete' or sha(micro/'labels.json')!=completion['labels_sha256'] or sha(micro/'micro-raw.json')!=completion['raw_sha256']:raise ValueError('micro evidence changed')
    floors=communication_floors(read(micro/'labels.json'))
    # native static instruction counts are per instruction occurrence; the
    # demand ledger counts one warp issue for each of four warps. Scalar shared
    # micro operations are 4B/thread, so 1 byte of native static transfer maps
    # to 1 warp issue across a four-warp CTA. SHFL maps 1 native inst to4issues.
    priors={'shared_warp_issues':max(1,math.ceil(floors['shared_transfer_bytes']['floor_us_per_unit']*1e6)),
        'shuffle_warp_issues':max(1,math.ceil(floors['shuffle_instructions']['floor_us_per_unit']/4*1e6)),
        'synchronization_barriers':max(1,math.ceil(floors['barrier_instructions']['floor_us_per_unit']*1e6)),
        'communication_alu_warp_issues':1,'alu_warp_issues':1}
    for row in rows:
        if sha(row['timing_path'])!=row['timing_sha256'] or sha(Path(row['path'])/'record.json')!=row['record_sha256'] or sha(Path(row['path'])/'execution.json')!=row['execution_sha256']:raise ValueError('known static training changed')
    x=np.array([[r['features'][n] for n in FEATURES] for r in rows],dtype=float);y=np.array([r['median_us'] for r in rows]);fitx=x.copy()
    for name in FIXED_ZERO:fitx[:,FEATURES.index(name)]=0
    lower=np.array([priors.get(n,0)/1e6 for n in FEATURES]);adjusted=y-fitx@lower
    if np.any(adjusted<0):raise ValueError('communication priors exceed timing')
    solution=coordinate_fit(fitx,adjusted,ridge=1e-6,max_iterations=20000,tolerance=1e-10)
    if not solution['converged']:raise ValueError('v6 NNLS not converged')
    weights={n:(0 if n in FIXED_ZERO else max(priors.get(n,0),int(round((v+lo)*1e6)))) for n,v,lo in zip(FEATURES,solution['coefficients'],lower)}
    out=fresh(out);profile=dict(schema_version=1,model='sm90-layout-surviving-calibrated-v1',coefficients=weights,provenance='all previously observed pure-f32 exact image labels; positive communication issue calibration priors; no holdout labels')
    write(out/'profile.json',profile);write(out/'fit.json',dict(solution,features=list(FEATURES),communication_priors_ticks=priors,communication_calibration=floors,fixed_zero=list(FIXED_ZERO),training_images=len(rows),intercept_usage='diagnostics only; excluded from same-program objective',engineering_priors=['ALU and communication-address ALU >=1tick are registered nonzero engineering priors, not measured hardware lower bounds'],no_holdout_use=True));write(out/'training-ledger.json',rows)
    write(out/'inputs.json',dict(training_manifest=str(training/'manifest.json'),training_manifest_sha256=sha(training/'manifest.json'),micro_completion=str(micro/'completion.json'),micro_completion_sha256=sha(micro/'completion.json'),profile_sha256=sha(out/'profile.json'),fit_sha256=sha(out/'fit.json'),training_ledger_sha256=sha(out/'training-ledger.json'),no_holdout_use=True))
    return out


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True);q=s.add_parser('collect');q.add_argument('out',type=Path);q=s.add_parser('fit');q.add_argument('training',type=Path);q.add_argument('micro',type=Path);q.add_argument('out',type=Path);a=p.parse_args()
    if a.action=='collect':collect(a.out)
    else:fit(a.training,a.micro,a.out)
if __name__=='__main__':main()
