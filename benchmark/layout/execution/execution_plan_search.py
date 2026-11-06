"""Bounded, label-free joint selection on checked layout replay packets.

The layout solver supplies the finite archive. Both strategies preview the
same layout/plan product. LI fixes its layout to the local incumbent and ranks
the same lowering options with local work. J can use any retained layout and
prices edge communication and shared storage.
"""

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time

from backend import emit
from execution_plan_cost import compare_features, copy_publication_only, extract_features, static_key
from native_static_support import check_static_shared


@dataclass(frozen=True)
class Policy:
    access: str = 'vector'
    reduction: str = 'auto'
    mma_group: int = 1
    shared: str = 'declared'
    store: str = 'direct'
    copy_batch: bool = False

    def emit(self, packet):
        return emit(packet, access_policy=self.access, reduction_policy=self.reduction,
                    mma_group_size=self.mma_group, shared_policy=self.shared,
                    store_policy=self.store, copy_batch=self.copy_batch, trace_execution=True)


BASELINE = Policy()
ATTENTION_SCOPE = {'frisk.copy': 3, 'frisk.mma': 2, 'frisk.reduce_tensor': 2, 'frisk.tile_store': 1}
COPY_BATCH = Policy(mma_group=16, copy_batch=True)
ATTENTION_ABLATIONS = (
    Policy(mma_group=16),
    COPY_BATCH,
    Policy(mma_group=16, store='warp-gf2'),
    Policy(mma_group=16, copy_batch=True, store='warp-gf2'),
)
ATTENTION_SCHEDULES = (
    COPY_BATCH,
    replace(COPY_BATCH, shared='late'),
    replace(COPY_BATCH, shared='reuse-k'),
    Policy(mma_group=16),
)


def policies_for(packet, *, attention_ablation=False, attention_schedule_ablation=False):
    names = Counter(op['name'] for op in packet['operations'])
    options = [BASELINE, Policy(access='factored'), Policy(store='shared')]
    if 'frisk.reduce_tensor' in names:
        options.append(Policy(reduction='eager'))
    full_attention = all(names[op] == count for op, count in ATTENTION_SCOPE.items())
    if attention_schedule_ablation:
        if not full_attention or attention_ablation:
            raise ValueError('schedule ablation requires the full checked Attention graph and its own mode')
        return (BASELINE,*ATTENTION_SCHEDULES)
    if attention_ablation:
        if not full_attention:
            raise ValueError('attention ablation requires the full checked Attention graph')
        return (*options, Policy(mma_group=4), Policy(mma_group=16, shared='reuse'), *ATTENTION_ABLATIONS)
    if 'frisk.mma' in names:
        options.extend((Policy(mma_group=4), Policy(mma_group=16), Policy(shared='reuse'),
                        Policy(mma_group=4, shared='reuse'), Policy(mma_group=16, shared='reuse')))
        if full_attention and all(
                op.get('attrs', {}).get('frisk.copy_contract', {}).get('path') == 'vector'
                for op in packet['operations'] if op['name'] == 'frisk.copy'):
            # Replace one policy to keep the preview product and native K4 unchanged.
            options.insert(options.index(Policy(mma_group=16))+1, COPY_BATCH)
        else:
            options.append(Policy(mma_group=4, shared='reuse', store='shared'))
    return tuple(options)


@dataclass(frozen=True)
class LayoutReplay:
    name: str
    packet: dict
    problem_sha256: str


@dataclass
class Preview:
    layout: str
    policy: Policy
    text: str
    configuration: dict
    features: dict
    program_sha256: str


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _abi(config):
    return {key: config[key] for key in ('copy_path', 'global_alignments', 'root_order', 'tma_plans')}


def _mma_swizzles(packet):
    return tuple(op['attrs']['frisk.mma_contract'][operand]['swizzle']
                 for op in packet['operations'] if op['name'] == 'frisk.mma'
                 for operand in ('a_descriptor', 'b_descriptor')
                 if operand in op.get('attrs', {}).get('frisk.mma_contract', {}))


def select_pairs(layouts, *, local_incumbent, baseline_layout, legacy_layouts, grid, block=(128, 1, 1),
                 preview_budget=96, native_budget=4, attention_ablation=False,
                 attention_schedule_ablation=False, joint_layouts=None):
    """Return the J, LI and legacy native-request queues, plus the shared audit.

    A checked-replay producer is responsible for binding each packet to the
    same prepared source identity. Failed previews are charged and retained.
    No runtime labels or compiler calibration fit are read by this function.
    """
    layouts=tuple(layouts)
    names=[item.name for item in layouts]
    if (not layouts or len(set(names)) != len(names) or local_incumbent not in names
            or baseline_layout not in names or len({item.problem_sha256 for item in layouts}) != 1
            or any(len(item.problem_sha256) != 64 or any(c not in '0123456789abcdef' for c in item.problem_sha256)
                   for item in layouts)):
        raise ValueError('layout replays do not belong to one checked prepared problem')
    if type(preview_budget) is not int or preview_budget < len(layouts) or type(native_budget) is not int or native_budget < 1:
        raise ValueError('insufficient explicit preview or native budget')
    if attention_ablation and (native_budget!=4 or local_incumbent!=baseline_layout):
        raise ValueError('attention ablation requires K4 and a common local and legacy layout')
    if attention_schedule_ablation:
        if native_budget!=4 or joint_layouts is None:
            raise ValueError('schedule ablation requires K4 and the checked J layout archive')
        joint_layouts=frozenset(joint_layouts)
        if local_incumbent not in joint_layouts or not joint_layouts.issubset(names):
            raise ValueError('schedule ablation requires the common J/LI layout control')
    options=policies_for(layouts[0].packet, attention_ablation=attention_ablation,
                         attention_schedule_ablation=attention_schedule_ablation)
    attempts, previews = [], []
    wall_start=time.perf_counter_ns()
    # Put the pinned strategy on all layouts before more expensive alternatives.
    for policy in options:
        for item in layouts:
            if len(attempts)==preview_budget:
                break
            before=time.perf_counter_ns()
            row=dict(layout=item.name, policy=asdict(policy))
            try:
                text,config=policy.emit(item.packet)
                check_static_shared(config)
                features=extract_features(text,config,grid=grid,block=block)
                identity=_digest(text.encode()+b'\0'+json.dumps(_abi(config),sort_keys=True).encode())
                previews.append(Preview(item.name,policy,text,config,features,identity))
                row.update(status='checked',program_sha256=identity,features=features)
            except (ValueError, KeyError, IndexError, TypeError) as error:
                row.update(status='unsupported',error=f'{type(error).__name__}: {error}')
            row['wall_ns']=time.perf_counter_ns()-before
            attempts.append(row)
        if len(attempts)==preview_budget:
            break
    baseline=next((x for x in previews if x.layout==baseline_layout and x.policy==BASELINE),None)
    local=next((x for x in previews if x.layout==local_incumbent and x.policy==BASELINE),None)
    if baseline is None or local is None:
        raise ValueError('checked local and legacy fallbacks required')
    if any(_abi(p.configuration)!=_abi(baseline.configuration) for p in previews):
        raise ValueError('physical ABI changed across layout/plan previews')

    def rank(pool, *, local_only):
        unique={}
        for p in sorted(pool, key=lambda p: (static_key(p.features, local=local_only), names.index(p.layout),
                                             options.index(p.policy))):
            unique.setdefault(p.program_sha256,p)
        return list(unique.values())

    by_policy={(p.layout,p.policy):p for p in previews}

    def publication_control(preview):
        control=by_policy.get((preview.layout,replace(preview.policy,copy_batch=False)))
        if control is not None and copy_publication_only(preview.features,control.features):
            return control
        return None

    def shortlist(pool, fallback):
        # Keep both the pinned route and the matched non-batched control when
        # testing a batched proposal. The fixed decision requires the latter.
        first=fallback
        if native_budget>1:
            first=next((p for p in pool
                        if (not p.policy.copy_batch or (native_budget>=3 and publication_control(p)))
                        and compare_features(p.features,fallback.features)['structural_relation']=='dominates'),fallback)
        ordered=[first]
        if first.program_sha256!=fallback.program_sha256 and native_budget>1:
            ordered.append(fallback)
        if first.policy.copy_batch:
            control=publication_control(first)
            if all(p.program_sha256!=control.program_sha256 for p in ordered):
                ordered.append(control)
        ids={p.program_sha256 for p in ordered}
        policies={p.policy for p in ordered}
        for p in pool:
            if len(ordered)==native_budget:
                break
            if p.program_sha256 not in ids and p.policy not in policies:
                ordered.append(p)
                ids.add(p.program_sha256)
                policies.add(p.policy)
        ordered.extend(p for p in pool if p.program_sha256 not in ids)
        return ordered[:native_budget]

    joint=rank(previews,local_only=False)
    informed=rank((p for p in previews if p.layout==local_incumbent),local_only=True)
    if baseline_layout not in legacy_layouts or any(name not in names for name in legacy_layouts):
        raise ValueError('legacy replay inventory differs from preview inputs')
    pinned={p.layout:p for p in previews if p.policy==BASELINE}
    legacy=list({p.program_sha256:p for p in [baseline,*(pinned[name] for name in legacy_layouts
                      if name in pinned and pinned[name].program_sha256!=baseline.program_sha256)]}.values())
    def swizzle_probe_for(first, queue):
        swizzles={item.name:_mma_swizzles(item.packet) for item in layouts}
        anchor=swizzles[first.layout]
        queued={p.program_sha256 for p in queue}
        alternatives=[p for p in joint
                      if p.program_sha256 not in queued and p.policy==first.policy
                      and p.features['scope']==first.features['scope']
                      and p.features['static']==first.features['static']
                      and len(swizzles[p.layout])==len(anchor) and swizzles[p.layout]!=anchor]
        if not alternatives:
            return None
        probe=min(alternatives,key=lambda p:sum(a!=b for a,b in zip(anchor,swizzles[p.layout])))
        audit=dict(slot=len(queue)-1,anchor_layout=first.layout,layout=probe.layout,
                   anchor_swizzles=anchor,swizzles=swizzles[probe.layout],
                   displaced_program_sha256=queue[-1].program_sha256)
        queue[-1]=probe
        return audit

    swizzle_probe=None
    joint_schedule_probe=None
    if attention_schedule_ablation:
        local_previews={p.policy:p for p in previews if p.layout==local_incumbent}
        if any(policy not in local_previews for policy in ATTENTION_SCHEDULES):
            raise ValueError('schedule ablation has an unsupported matched local plan')
        local_queue=[local_previews[policy] for policy in ATTENTION_SCHEDULES]
        used={p.program_sha256 for p in local_queue}
        if len(used)!=native_budget:
            raise ValueError('schedule ablation has equivalent matched local plans')
        joint_queue=list(local_queue)
        alternatives=[p for p in joint
                      if p.layout in joint_layouts and p.layout!=local_incumbent
                      and p.policy in ATTENTION_SCHEDULES[:3] and p.program_sha256 not in used
                      and p.features['scope']==local_queue[0].features['scope']]
        if alternatives:
            # Preserve three matched controls. Spend the fourth J slot on
            # the best different layout with K reuse, if one was proved.
            probe=min(alternatives,key=lambda p:(
                p.policy.shared!='reuse-k',
                p.features['static']['shared_allocated_peak_bytes'],
                static_key(p.features),
                names.index(p.layout)))
            joint_queue[-1]=probe
            joint_schedule_probe=dict(slot=native_budget-1,anchor_layout=local_incumbent,
                                      layout=probe.layout,policy=asdict(probe.policy))
    elif attention_ablation:
        # Pin a common arithmetic and layout control. Cross the two
        # interventions within LI, and change only the descriptor in J c3.
        local_previews={p.policy:p for p in previews if p.layout==local_incumbent}
        if any(policy not in local_previews for policy in ATTENTION_ABLATIONS):
            raise ValueError('attention ablation has an unsupported local intervention')
        local_queue=[local_previews[policy] for policy in ATTENTION_ABLATIONS]
        if len({p.program_sha256 for p in local_queue})!=native_budget:
            raise ValueError('attention ablation produced an identical intervention')
        joint_queue=list(local_queue)
        swizzle_probe=swizzle_probe_for(joint_queue[-1],joint_queue)
    else:
        joint_queue=shortlist(joint,baseline)
        local_queue=shortlist(informed,local)
        if native_budget>=4 and len(joint_queue)==native_budget and joint_queue[0].policy.mma_group>1:
            swizzle_probe=swizzle_probe_for(joint_queue[0],joint_queue)
    queues={'J':joint_queue,'LI':local_queue,'legacy':legacy[:native_budget]}
    return queues, dict(contract='frisk-joint-execution-plan-v1', scope='checked archive times checked lowering',
                        preview_budget=preview_budget,native_budget=native_budget,layouts=len(layouts),
                        policies=[asdict(p) for p in options],preview_attempts=attempts,
                        distinct_programs=len({p.program_sha256 for p in previews}),
                        native_requests={name:len(rows) for name,rows in queues.items()},
                        static_fixed={name:rows[0].program_sha256 for name,rows in queues.items()},
                        fallback_program=baseline.program_sha256,joint_swizzle_probe=swizzle_probe,
                        attention_ablation=attention_ablation,
                        attention_schedule_ablation=attention_schedule_ablation,
                        joint_schedule_probe=joint_schedule_probe,
                        wall_ns=time.perf_counter_ns()-wall_start)


def select_checked_fixture(fixture, *, preview_budget=96, native_budget=4, attention_ablation=False,
                           attention_schedule_ablation=False):
    """Load the hashed C++ replay manifest and choose a native request queue."""
    fixture=Path(fixture).resolve(strict=True)
    raw=fixture.read_bytes()
    doc=json.loads(raw)
    source=fixture.parent/'input.mlir'
    if _digest(source.read_bytes())!=doc['input_sha256']:
        raise ValueError('prepared input changed')
    layouts={}
    roles={}
    public={}
    legacy_names=[]
    joint_names=[]
    for mode in ('LI','J','legacy-K4'):
        group=doc['groups'][mode]
        if group['status']!='exported':
            raise ValueError('all requested layout strategies need a checked archive')
        for entry in group['candidates']:
            if entry['status']!='exported':
                continue
            path=Path(entry['path'])/'execution.json'
            content=path.read_bytes()
            if _digest(content)!=entry['artifacts']['execution.json']:
                raise ValueError('replayed packet changed')
            key=entry['complete_assignment_sha256']
            if key not in roles:
                roles[key]=entry['role']
                layouts[entry['role']]=LayoutReplay(entry['role'],json.loads(content),doc['input_sha256'])
            name=roles[key]
            if mode=='legacy-K4':
                legacy_names.append(name)
            elif mode=='J':
                joint_names.append(name)
            if entry['is_public_witness']:
                if mode in public and public[mode]!=name:
                    raise ValueError('ambiguous public layout witness')
                public[mode]=name
    if set(public)!={'J','LI','legacy-K4'}:
        raise ValueError('missing checked public witnesses')
    queues,audit=select_pairs(layouts.values(),local_incumbent=public['LI'],baseline_layout=public['legacy-K4'],
                             legacy_layouts=legacy_names,grid=doc['spec']['grid'],block=doc['spec']['block'],
                             preview_budget=preview_budget,native_budget=native_budget,
                             attention_ablation=attention_ablation,
                             attention_schedule_ablation=attention_schedule_ablation,
                             joint_layouts=joint_names)
    audit.update(fixture=str(fixture),fixture_sha256=_digest(raw),layout_witnesses=public)
    return queues,audit


def main():
    import argparse

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--preview-budget',type=int,default=96)
    parser.add_argument('--native-budget',type=int,default=4)
    parser.add_argument('--attention-ablation',action='store_true',
                        help='select matched K4 copy, GF(2) store and combined Attention candidates')
    parser.add_argument('--attention-schedule-ablation',action='store_true',
                        help='select matched K4 Attention Shared publication and reuse schedules')
    args=parser.parse_args()
    queues,audit=select_checked_fixture(args.fixture,preview_budget=args.preview_budget,native_budget=args.native_budget,
                                        attention_ablation=args.attention_ablation,
                                        attention_schedule_ablation=args.attention_schedule_ablation)
    args.output.mkdir()
    manifests={}
    for mode,items in queues.items():
        rows=[]
        for slot,preview in enumerate(items):
            destination=args.output/f'{mode}-c{slot}'
            destination.mkdir()
            (destination/'emitted.ttgir').write_text(preview.text)
            (destination/'emitted-config.json').write_text(json.dumps(preview.configuration,indent=2,allow_nan=False))
            rows.append(dict(slot=slot,layout=preview.layout,policy=asdict(preview.policy),
                             program_sha256=preview.program_sha256,output=str(destination)))
        manifests[mode]=rows
    (args.output/'selection.json').write_text(json.dumps(dict(audit,queues=manifests),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
