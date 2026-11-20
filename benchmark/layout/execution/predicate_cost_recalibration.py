"""Known-only recalibration after aggregate-PR native dependency repair.

Every native feature is re-extracted from the original exact CUBIN by analyzer
v2. Old timing labels remain observed training data. Shared burden is migrated
only through an independently checked Shared-semantics bridge; this is not full
old/new generated-image equality and never changes old sealed evidence.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil

import numpy as np
import native_cost_calibration as base
import native_local_calibration as local
import native_cost_analysis as native
from calibration_analysis import coordinate_fit

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'guoqiao/optimize_plan/network_selection_execution_artifacts'
CONTRACT = 'frisk-predicate-native-recalibration-v1'
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SOURCES = {
    'native-known': ROOT/'guoqiao/optimize_plan/native_cost_repair_artifacts/native-known-training-r3',
    'local-known': ROOT/'guoqiao/optimize_plan/partial_cost_efficiency_artifacts/local-aware-known-r1',
    'communication': ROOT/'guoqiao/optimize_plan/native_cost_repair_artifacts/communication-training-r3-native-refresh',
    'local': ROOT/'guoqiao/optimize_plan/partial_cost_efficiency_artifacts/local-training-r2',
}
EXPECTED_IMAGES = {'native-known':42,'local-known':56,'communication':16,'local':22}
INDICES = {
    ROOT/'guoqiao/optimize_plan/native_cost_repair_artifacts/evidence-index.json':'66b28196c5065d5cc4504d3a069991b766718ec7d5618163533c120ba33ded9b',
    ROOT/'guoqiao/optimize_plan/partial_cost_efficiency_artifacts/evidence-index.json':'eff4c3ca39681c91115e258638a34f5242ca1f4eb8d984a67835928657337fef',
}
FEATURES = base.MODEL_FEATURES


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def fresh(path):
    path=Path(path).resolve()
    if not path.is_relative_to(BASE.resolve()):raise ValueError('fresh current-stage output required')
    path.mkdir(parents=True,exist_ok=False)
    return path


def _dependencies(paths):
    result={}
    for p in paths:
        p=Path(p)
        if p.is_symlink() or not p.is_file():raise ValueError('regular immutable dependency required')
        result[str(p.resolve())]=sha(p)
    return result


def _verify_dependencies(dependencies):
    if type(dependencies) is not dict or not dependencies:raise ValueError('nonempty dependency map required')
    for name,value in dependencies.items():
        p=Path(name)
        if not p.is_absolute() or p.is_symlink() or not p.is_file() or sha(p)!=value:
            raise ValueError('recalibration dependency changed: '+name)


def _trusted_entries():
    """Bind old training files to the recorded seal, not self-consistent edits."""
    entries={}
    for path,expected in INDICES.items():
        if sha(path)!=expected:raise ValueError("old sealed index changed")
        index=read(path)
        if 'repository_root' in index:
            if Path(index['repository_root']).resolve()!=ROOT.resolve():raise ValueError("sealed training repository differs")
        elif index.get('contained_root')!=str(Path(path).parent.relative_to(ROOT)):
            raise ValueError("unknown sealed contained-root format")
        for row in index['entries']:
            target=(ROOT/row['path']).resolve()
            if not target.is_relative_to(ROOT.resolve()):raise ValueError("sealed path escapes repository")
            name=str(target);value=row['sha256']
            if name in entries and entries[name]!=value:raise ValueError("conflicting sealed entry")
            entries[name]=value
    return entries


def _check_sealed_source(path,entries):
    path=Path(path)
    if path.is_symlink():raise ValueError("regular sealed source required")
    expected=entries.get(str(path.resolve()))
    if expected is None or sha(path)!=expected:
        raise ValueError("training source differs from its recorded sealed entry: "+str(path))
    return expected


def _bridge(path):
    import compatible_native_profiles
    if not hasattr(compatible_native_profiles,'validate_shared_semantics_bridge'):
        raise ValueError('strict Shared semantics bridge verifier is unavailable')
    bridge=compatible_native_profiles.validate_shared_semantics_bridge(path)
    required={'dependencies','old_bank_implementation_sha256','new_bank_implementation_sha256','contract'}
    if not required <= set(bridge):raise ValueError('incomplete Shared semantics bridge')
    _verify_dependencies(bridge['dependencies'])
    if bridge['new_bank_implementation_sha256']!=base.bank_implementation_identity():
        raise ValueError('Shared bridge does not describe the current bank implementation')
    return bridge


def reattach_known_shared(new,old,directory,bridge,bridge_path):
    """Keep bank numbers, prove their source, update only certified Shared identity."""
    if new['status']!='available':raise ValueError('known image no longer in native analysis domain')
    directory=Path(directory);banks=copy.deepcopy(old['symbolic_banks'])
    if banks.get('contract')!='frisk-emitted-affine-shared-bank-v1' or banks['implementation_sha256']!=bridge['old_bank_implementation_sha256']:
        raise ValueError('old bank proof outside certified bridge domain')
    for name,key in [('execution.json','execution_sha256'),('kernel.ttgir','emitted_ttgir_sha256'),('kernel.cubin','native_cubin_sha256')]:
        if sha(directory/name)!=banks.get(key):raise ValueError('old exact image/bank evidence mismatch: '+name)
    for feature in ('shared_load_bytes','shared_store_bytes'):
        if new['features'][feature]!=old['features'][feature]:raise ValueError('native Shared transfer counts changed')
    banks['implementation_sha256']=copy.deepcopy(bridge['new_bank_implementation_sha256'])
    banks['shared_semantics_bridge']=dict(path=str(Path(bridge_path).resolve()),sha256=sha(bridge_path),
        contract=bridge['contract'],old_bank_proof_sha256=digest(old['symbolic_banks']),
        numbers_changed=False,full_generated_image_equality_claim=False,
        interpretation='old exact transport image; only checked Shared semantics/number identity migrated')
    result=copy.deepcopy(new);result['symbolic_banks']=banks
    return result


def refresh_collection(kind,out,bridge_path):
    if kind not in SOURCES:raise ValueError('only the four preregistered observed training collections are allowed')
    out=fresh(out);source=SOURCES[kind];entries=_trusted_entries()
    known=kind.endswith('known');filename='known-labels.json' if known else 'labels.json'
    _check_sealed_source(source/'completion.json',entries);_check_sealed_source(source/filename,entries)
    if not known:_check_sealed_source(source/'micro-raw.json',entries)
    completion=read(source/'completion.json')
    key='known_labels_sha256' if known else 'labels_sha256'
    if completion.get('status')!='complete' or completion.get('no_holdout_use') is not True or sha(source/filename)!=completion[key]:
        raise ValueError('old training collection identity mismatch')
    originals=read(source/filename)
    if len(originals)!=EXPECTED_IMAGES[kind]:raise ValueError('training image matrix changed')
    bridge=_bridge(bridge_path);dependencies=_dependencies([source/'completion.json',source/filename,bridge_path,*INDICES])
    rows=[];changes=[]
    for oldrow in originals:
        directory=Path(oldrow['path'])
        if sha(directory/'kernel.cubin')!=oldrow['cubin_sha256']:raise ValueError('observed training native image changed')
        dependencies.update(_dependencies([directory/'kernel.cubin']))
        extracted=native.analyze(directory,cache_dir=out/'native-cache')
        if known:
            updated=reattach_known_shared(extracted,oldrow['native'],directory,bridge,bridge_path)
            dependencies.update(_dependencies([directory/'execution.json',directory/'kernel.ttgir',oldrow['timing_path']]))
            if sha(oldrow['timing_path'])!=oldrow['timing_sha256']:raise ValueError('observed training raw labels changed')
        elif kind=='communication':
            updated=base.attach_micro_banks(extracted,oldrow['spec'])
        else:
            # Re-check retained static LDL/STL widths and zero-Shared proof.
            updated=local.native_evidence(directory,oldrow['spec'],out/'native-cache')
        if updated['status']!='available':raise ValueError('training image became unsupported')
        oldbanks=oldrow['native']['symbolic_banks'];newbanks=updated['symbolic_banks']
        for field in ('shared_read_bank_excess','shared_write_bank_excess'):
            if newbanks[field]!=oldbanks[field]:raise ValueError('Shared bank burden changed')
        row=copy.deepcopy(oldrow);row['native']=updated;row['features']=base.model_features(updated)
        differences={name:dict(old=oldrow['features'][name],new=row['features'][name]) for name in FEATURES if oldrow['features'][name]!=row['features'][name]}
        changes.append(dict(path=str(directory),cubin_sha256=oldrow['cubin_sha256'],changed_features=differences))
        rows.append(row)
    contracts=[base.analysis_contract(row['native']) for row in rows]
    if any(c!=contracts[0] for c in contracts) or contracts[0]['domain']!=native.DOMAIN:
        raise ValueError('mixed re-extracted native/bank domains')
    output=dict(status='complete',no_new_gpu=True,no_holdout_use=True,original_labels='observed training only',
        source_completion=str(source/'completion.json'),source_completion_sha256=sha(source/'completion.json'),
        analysis_contract=contracts[0],images=len(rows),changed_images=sum(bool(c['changed_features']) for c in changes))
    if known:
        write(out/'known-labels.json',rows);output['known_labels_sha256']=sha(out/'known-labels.json')
    else:
        if sha(source/'micro-raw.json')!=completion['raw_sha256']:raise ValueError('micro raw labels changed')
        shutil.copyfile(source/'micro-raw.json',out/'micro-raw.json');dependencies.update(_dependencies([source/'micro-raw.json']))
        write(out/'labels.json',rows);output.update(labels_sha256=sha(out/'labels.json'),raw_sha256=sha(out/'micro-raw.json'))
    dependencies.update(bridge['dependencies'])
    write(out/'feature-changes.json',changes)
    write(out/'provenance.json',dict(contract=CONTRACT,implementation_sha256=IMPLEMENTATION_SHA256,kind=kind,
        dependencies=dependencies,shared_bridge_path=str(Path(bridge_path).resolve()),shared_bridge_sha256=sha(bridge_path),
        reused_timing_labels=True,recomputed_actual_native_features=True,no_new_gpu=True,no_holdout_use=True))
    output.update(provenance_sha256=sha(out/'provenance.json'),changes_sha256=sha(out/'feature-changes.json'))
    write(out/'completion.json',output)
    return out


def fit_local(known,local_collection,communication,fixed_profile_path,out):
    """Repeat the registered LOCAL-only residual fit with newly refitted nonLOCAL."""
    known,local_collection,communication=map(Path,(known,local_collection,communication));out=fresh(out)
    for directory,key,file in [(known,'known_labels_sha256','known-labels.json'),(local_collection,'labels_sha256','labels.json'),(communication,'labels_sha256','labels.json')]:
        completion=read(directory/'completion.json')
        if completion['status']!='complete' or sha(directory/file)!=completion[key]:raise ValueError('refreshed training collection changed')
    old,loc,comm=read(known/'known-labels.json'),read(local_collection/'labels.json'),read(communication/'labels.json')
    floors=base.communication_floors(comm);floors['local_transfer_bytes']=local.local_floor(loc)
    fixed=base.load_profile(fixed_profile_path)
    rows=old+[dict(x,source_run='development-communication',group='isolated-communication') for x in comm]+[dict(x,source_run='development-LOCAL',group='isolated-LOCAL') for x in loc]
    groups={}
    for row in old:groups.setdefault((row['source_run'],row['group']),[]).append(row)
    for kind in ('barrier','shared','shuffle'):groups[('micro-communication',kind)]=[comm[0]]+[x for x in comm if x['spec']['kind']==kind]
    by={x['spec']['key']:x for x in loc}
    for words in (256,1024,4096):
        selected=[x for x in loc if x['spec']['kind']=='local' and x['spec']['words']==words]
        groups[('micro-LOCAL',str(words))]=selected+[by[f'control-{n}'] for n in sorted({x['spec']['repeats'] for x in selected})]
    lower=np.asarray([fixed['coefficients'][name]/1e6 for name in FEATURES]);lower[FEATURES.index('local_transfer_bytes')]=floors['local_transfer_bytes']['floor_us_per_unit']
    design=[];targets=[];pairs=[]
    for key,items in sorted(groups.items()):
        pair=list(itertools.combinations(items,2));scale=max(float(np.median([x['median_us'] for x in items])),1e-9)
        for a,b in pair:
            delta=np.asarray([a['features'][name]-b['features'][name] for name in FEATURES],dtype=float)
            value=float(np.median(np.asarray(a['batch_medians_us'])-b['batch_medians_us']));weight=1/(scale*np.sqrt(max(1,len(pair))))
            design.append(delta*weight);targets.append((value-delta@lower)*weight)
            pairs.append(dict(group=list(key),first=a['path'],second=b['path'],delta_us=value,normalization=scale,loss_weight=weight))
    x=np.asarray([[r['features'][name] for name in FEATURES] for r in rows],dtype=float);y=np.asarray([r['median_us'] for r in rows])
    for feature,target in zip(x,y):
        weight=np.sqrt(.05/len(rows))/max(target,1e-9);design.append(feature*weight);targets.append((target-feature@lower)*weight)
    weights,solution=local.fit_one_local(np.asarray(design),np.asarray(targets),lower,fixed)
    contracts=[base.analysis_contract(row['native']) for row in rows]
    if any(c!=contracts[0] for c in contracts) or contracts[0]!=fixed['analysis_contract']:
        raise ValueError('new native/LOCAL analysis contracts differ')
    if weights['local_transfer_bytes']<=0:raise ValueError('new calibrated LOCAL fee must be positive')
    if any(weights[name]!=fixed['coefficients'][name] for name in FEATURES if name!='local_transfer_bytes'):
        raise ValueError('LOCAL-only refit changed nonLOCAL coefficients')
    profile=dict(schema_version=1,model='sm90-native-surviving-schedule-v1',features=list(FEATURES),coefficients=weights,
        ticks_per_microsecond=1000000,intercept_us=0.,training_maxima={name:int(v) for name,v in zip(FEATURES,x.max(axis=0))},
        analysis_contract=contracts[0],communication_floors={k:v for k,v in floors.items() if k!='local_transfer_bytes'},local_floor=floors['local_transfer_bytes'],
        calibration=dict(version=CONTRACT,mode='re-extracted-v2-nonLOCAL-refit-followed-by-LOCAL-only-residual',
            fixed_profile_sha256=sha(fixed_profile_path),implementation_sha256=IMPLEMENTATION_SHA256,
            known_images=len(old),communication_images=len(comm),local_images=len(loc)),
        interpretation='known-only actual CUBIN predicate-DAG repair; empirical proxy, conservative PR edges, not cycles; old labels remain observed',
        unsupported_rule='training-zero feature remains unknown; explicit fallback',no_holdout_use=True)
    write(out/'profile.json',profile);write(out/'pair-ledger.json',pairs);write(out/'training-ledger.json',rows)
    write(out/'fit.json',dict(solution,fitted_parameter_names=['local_transfer_bytes'],fixed_nonlocal_parameters=13,
        constrained_lower_bounds=lower.tolist(),training_images=len(rows),universal_parameter_identifiability=False))
    write(out/'inputs.json',dict(known_completion=str(known/'completion.json'),known_completion_sha256=sha(known/'completion.json'),
        micro_completion=str(communication/'completion.json'),micro_completion_sha256=sha(communication/'completion.json'),
        local_completion=str(local_collection/'completion.json'),local_completion_sha256=sha(local_collection/'completion.json'),
        fixed_profile=str(Path(fixed_profile_path).resolve()),fixed_profile_sha256=sha(fixed_profile_path),
        profile_sha256=sha(out/'profile.json'),fit_sha256=sha(out/'fit.json'),ledger_sha256=sha(out/'training-ledger.json'),
        pair_ledger_sha256=sha(out/'pair-ledger.json'),no_holdout_use=True))
    return out


def run(out,bridge_path):
    out=fresh(out);collections={}
    for kind in SOURCES:collections[kind]=refresh_collection(kind,out/kind,bridge_path)
    native_fit=base.fit(collections['native-known'],collections['communication'],out/'native-fit')
    local_fit=fit_local(collections['local-known'],collections['local'],collections['communication'],native_fit/'profile.json',out/'local-fit')
    dependencies=_dependencies([__file__,Path(native.__file__),ROOT/'benchmark/layout/execution/native_cost_calibration.py',
        ROOT/'benchmark/layout/execution/native_local_calibration.py',ROOT/'benchmark/layout/execution/calibration_analysis.py',bridge_path])
    for directory in collections.values():
        for path in directory.iterdir():
            if path.is_file():dependencies.update(_dependencies([path]))
        dependencies.update(read(directory/'provenance.json')['dependencies'])
    for directory in (native_fit,local_fit):
        dependencies.update(_dependencies([p for p in directory.iterdir() if p.is_file()]))
    receipt=dict(schema_version=1,contract=CONTRACT,implementation_sha256=IMPLEMENTATION_SHA256,
        analyzer_domain=native.DOMAIN,analyzer_sha256=native.ANALYZER_SOURCE_SHA256,
        bank_bridge_path=str(Path(bridge_path).resolve()),bank_bridge_sha256=sha(bridge_path),
        dependencies=dependencies,native_profile=str(native_fit/'profile.json'),local_profile=str(local_fit/'profile.json'),
        no_new_gpu=True,no_holdout_use=True,nonLOCAL_equal=True,
        old_sources={k:str(v) for k,v in SOURCES.items()},
        interpretation='new v2 native features actually re-extracted; fourteen parameters refitted then LOCAL-only residual, old timing observations only')
    write(out/'recalibration-receipt.json',receipt)
    for kind,directory in [('native',native_fit),('local',local_fit)]:
        write(out/(kind+'-profile.json'),dict(schema_version=1,contract=CONTRACT,kind=kind,
            profile_path=str(directory/'profile.json'),profile_sha256=sha(directory/'profile.json'),
            receipt_path=str(out/'recalibration-receipt.json'),receipt_sha256=sha(out/'recalibration-receipt.json')))
    return out


def _load(path):
    path=Path(path)
    if path.is_symlink():raise ValueError('regular new-stage envelope required')
    path=path.resolve()
    if not path.is_relative_to(BASE.resolve()):
        raise ValueError('new-stage recalibration envelope required')
    envelope=read(path)
    fields={'schema_version','contract','kind','profile_path','profile_sha256','receipt_path','receipt_sha256'}
    if set(envelope)!=fields or envelope['schema_version']!=1 or envelope['contract']!=CONTRACT or envelope['kind'] not in ('native','local'):
        raise ValueError('unknown predicate recalibration envelope')
    if sha(__file__)!=IMPLEMENTATION_SHA256 or sha(envelope['receipt_path'])!=envelope['receipt_sha256'] or sha(envelope['profile_path'])!=envelope['profile_sha256']:
        raise ValueError('predicate recalibration identity changed')
    receipt_path=Path(envelope['receipt_path']).resolve()
    if receipt_path.parent!=path.parent or receipt_path.name!='recalibration-receipt.json':
        raise ValueError('profile envelope must name its sibling recalibration receipt')
    receipt=read(receipt_path)
    if receipt.get('contract')!=CONTRACT or receipt.get('implementation_sha256')!=IMPLEMENTATION_SHA256 or receipt.get('no_holdout_use') is not True or receipt.get('no_new_gpu') is not True:
        raise ValueError('invalid known-only recalibration receipt')
    if receipt.get('old_sources')!={k:str(v) for k,v in SOURCES.items()} or receipt.get('analyzer_domain')!=native.DOMAIN or receipt.get('analyzer_sha256')!=native.ANALYZER_SOURCE_SHA256:
        raise ValueError('recalibration source inventory/analyzer changed')
    expected_native=str(receipt_path.parent/'native-fit/profile.json');expected_local=str(receipt_path.parent/'local-fit/profile.json')
    if receipt['native_profile']!=expected_native or receipt['local_profile']!=expected_local or path.name!=envelope['kind']+'-profile.json':
        raise ValueError('recalibration profile paths differ from the fixed experiment matrix')
    _verify_dependencies(receipt['dependencies']);bridge=_bridge(receipt['bank_bridge_path'])
    for old_index,expected in INDICES.items():
        if sha(old_index)!=expected or receipt['dependencies'].get(str(old_index.resolve()))!=expected:
            raise ValueError('recalibration no longer binds the sealed training inventory')
    trusted_entries=_trusted_entries()
    for kind,original in SOURCES.items():
        directory=receipt_path.parent/kind;filename='known-labels.json' if kind.endswith('known') else 'labels.json'
        for name in ('completion.json',filename)+(() if kind.endswith('known') else ('micro-raw.json',)):
            expected=_check_sealed_source(original/name,trusted_entries)
            if receipt['dependencies'].get(str((original/name).resolve()))!=expected:
                raise ValueError('receipt does not bind the recorded sealed training source')
        provenance=read(directory/'provenance.json');completion=read(directory/'completion.json')
        if (provenance.get('kind')!=kind or provenance.get('implementation_sha256')!=IMPLEMENTATION_SHA256 or provenance.get('no_holdout_use') is not True or completion.get('images')!=EXPECTED_IMAGES[kind]):
            raise ValueError('known-only re-extraction matrix changed')
        oldrows=read(original/filename);rows=read(directory/filename)
        if len(rows)!=len(oldrows) or len(rows)!=EXPECTED_IMAGES[kind]:raise ValueError('training rows omitted or substituted')
        for oldrow,row in zip(oldrows,rows):
            old_meta={k:v for k,v in oldrow.items() if k not in ('native','features')}
            new_meta={k:v for k,v in row.items() if k not in ('native','features')}
            if old_meta!=new_meta or base.analysis_contract(row['native'])['domain']!=native.DOMAIN or row['features']!=base.model_features(row['native']):
                raise ValueError('observed labels or re-extracted native feature contract changed')
    profile=base.load_profile(envelope['profile_path'])
    expected_path=receipt['native_profile'] if envelope['kind']=='native' else receipt['local_profile']
    if envelope['profile_path']!=expected_path or profile['analysis_contract']['domain']!=native.DOMAIN or profile['analysis_contract']['analyzer_sha256']!=native.ANALYZER_SOURCE_SHA256:
        raise ValueError('new analyzer/profile contract mismatch')
    if profile['analysis_contract']['symbolic_bank_implementation_sha256']!=bridge['new_bank_implementation_sha256']:
        raise ValueError('new Shared semantic identity mismatch')
    fixed=base.load_profile(receipt['native_profile']);loc=base.load_profile(receipt['local_profile'])
    if fixed['analysis_contract']!=loc['analysis_contract'] or any(fixed['coefficients'][name]!=loc['coefficients'][name] for name in FEATURES if name!='local_transfer_bytes'):
        raise ValueError('native/LOCAL pair changed nonLOCAL coefficients or native domain')
    if fixed['training_maxima']['local_transfer_bytes']!=0 or fixed['coefficients']['local_transfer_bytes']!=0 or loc['coefficients']['local_transfer_bytes']<=0:
        raise ValueError('LOCAL calibration ablation pair lost its intended separation')
    dependencies=dict(receipt['dependencies']);dependencies.update(_dependencies([path,envelope['receipt_path']]));dependencies.update(bridge['dependencies'])
    return profile,dependencies


def load_profile(path):return _load(path)[0]
def dependency_hashes(path):return _load(path)[1]
def base_profile_path(path):
    _load(path)
    return Path(read(path)['profile_path'])


def validate_configuration(configuration):
    """Require the new LOCAL/native ablation pair before reading fresh labels."""
    paths=[Path(configuration[name]['profile']) for name in ('native','old_native')]
    local_envelope,native_envelope=map(read,paths)
    if local_envelope.get('kind')!='local' or native_envelope.get('kind')!='native':
        raise ValueError('native must use the new LOCAL profile and old_native its v2 nonLOCAL anchor')
    if (local_envelope.get('receipt_path')!=native_envelope.get('receipt_path') or
        local_envelope.get('receipt_sha256')!=native_envelope.get('receipt_sha256')):
        raise ValueError('both ablation profiles must share the same known-only recalibration receipt')
    calibrated,dependencies=_load(paths[0]);anchor,old_dependencies=_load(paths[1])
    dependencies.update(old_dependencies)
    for profile in (calibrated,anchor):
        if profile['features']!=list(FEATURES) or set(profile['coefficients'])!=set(FEATURES) or any(type(v) is not int or v<0 for v in profile['coefficients'].values()):
            raise ValueError('nonnegative integer fourteen-feature profile required')
    if (calibrated['analysis_contract']!=anchor['analysis_contract'] or
        calibrated['training_maxima']['local_transfer_bytes']<=0 or
        calibrated['coefficients']['local_transfer_bytes']<=0 or
        anchor['training_maxima']['local_transfer_bytes']!=0 or
        anchor['coefficients']['local_transfer_bytes']!=0):
        raise ValueError('new analyzer identity and covered/unknown LOCAL ablation required')
    if any(calibrated['coefficients'][name]!=anchor['coefficients'][name] for name in FEATURES if name!='local_transfer_bytes'):
        raise ValueError('nonLOCAL ablation coefficients differ')
    return dict(contract=CONTRACT,analysis_contract=calibrated['analysis_contract'],
        local_weight=calibrated['coefficients']['local_transfer_bytes'],
        nonLOCAL_equal=True,no_new_gpu=True,no_holdout_use=True,dependencies=dependencies)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--bank-bridge',required=True,type=Path)
    args=parser.parse_args();print(run(args.out,args.bank_bridge))

if __name__=='__main__':main()
