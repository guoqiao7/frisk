"""Independent whole-SSA survival audit against actual backend packets.
This is conservative static survival, not ptxas CSE, native dynamic work or a
hardware instruction/cycle model. Native feedback is audited separately.
"""
import backend
import demand_coverage_audit as emission

FEATURES=emission.FEATURE_NAMES[:-1]+('shared_warp_issues','shuffle_warp_issues','communication_alu_warp_issues')
WORK_BASIS='whole-ssa-survival-with-emitted-communication-roots-v1'


def surviving_ssa(operations,distributed,observed,plans):
    """Keep observable final output and every physically emitted communication."""
    definitions={v:op for op in operations for v in op['results'] if v in distributed}
    stores=[op for op in operations if op['name']=='frisk.tile_store']
    if len(stores)!=1:raise ValueError('one final store required')
    roots={stores[0]['operands'][0]};communication=[]
    for op,plan in plans:
        target=op['results'][0]
        if plan['mode'] in ('Shared','Shuffle') and observed.get(target):
            roots.update((op['operands'][0],target));communication.append(op)
    retained=set();pending=list(roots)
    while pending:
        value=pending.pop()
        if value in retained or value not in distributed:continue
        if value not in definitions:raise ValueError('survival root lacks distributed definition')
        retained.add(value);pending.extend(x for x in definitions[value]['operands'] if x in distributed)
    return retained,communication


def _check_features(actual,expected):
    if not isinstance(actual,dict) or set(actual)-set(FEATURES):raise ValueError('unknown surviving priced feature')
    for name,value in expected.items():
        if type(actual.get(name,0)) is not int or actual.get(name,0)!=value:raise ValueError('surviving priced feature differs: '+name)


def audit(packet,record,*,require_priced_features=True):
    # Strong old emission proof stays separate and is never weakened by survival.
    proof=emission.audit(packet,record,require_priced_features=False)
    ledger=record.get('surviving_coverage')
    if not isinstance(ledger,dict):raise ValueError('missing surviving_coverage')
    if ledger.get('proof_status')!='Proven':
        if proof['status']!='explicit-fallback' or not ledger.get('reason'):raise ValueError('unproven surviving work in exact f32 domain')
        return dict(schema_version=1,status='explicit-fallback',reason=ledger['reason'],no_gpu=True)
    if ledger.get('work_basis')!=WORK_BASIS or ledger.get('native_exact') is not False:raise ValueError('wrong static survival contract')
    if ledger.get('emitted')!=record['demand_coverage']:raise ValueError('survival emission basis substituted')
    observed={}
    def observe(value,indices):
        signature=emission.canonical_query(indices)
        if signature in observed.setdefault(value,set()):raise ValueError('duplicate emitted query')
        observed[value].add(signature)
    emitter=backend.Emitter(packet,demand_query_observer=observe);_,info=emitter.run();demand=bool(info.get('demand_lowering'))
    distributed={v['id'] for v in emitter.data['values'] if 'distributed'in v}
    if not demand:observed={i:{(r,0,0,0,0,0,0,0) for r in range(emitter.v[i]['distributed']['topology'][0])} for i in distributed}
    plans=[(op,backend.validate_redistribution(op,emitter.v)) for op in emitter.data['operations'] if op['name']=='frisk.convert_layout']
    retained,communication=surviving_ssa(emitter.data['operations'],distributed,observed,plans)
    surviving={i:q for i,q in observed.items() if i in retained}
    cpp=ledger.get('surviving')
    if not isinstance(cpp,dict) or cpp.get('exact') is not True:raise ValueError('missing proven surviving query ledger')
    diagnostics,_=emission.compare_queries(emitter.data['values'],emission.parse_values(cpp),surviving,emission.parse_boundary_queries(cpp))
    for field in ('retained_variables','removed_variables','retained_communication'):
        items=ledger.get(field)
        if not isinstance(items,list) or len(items)!=len(set(items)) or any(type(i) is not int or i<0 for i in items):raise ValueError('invalid survival diagnostic partition')
    keep_ids=set(ledger['retained_variables']);removed_ids=set(ledger['removed_variables'])
    if keep_ids&removed_ids:raise ValueError('survival variable partition overlaps')
    for value in emitter.data['values']:
        if 'distributed'not in value:continue
        for variable in value.get('layout_variable_ids',[]):
            if variable not in (keep_ids if value['id'] in retained else removed_ids):raise ValueError('survival definition diagnostic partition differs')
    materialized=sum(map(len,observed.values()));surviving_materialized=sum(map(len,surviving.values()))
    if ledger['emitted'].get('materialized_query_vectors')!=materialized or cpp.get('materialized_query_vectors')!=surviving_materialized or ledger.get('removed_query_vectors')!=materialized-surviving_materialized:
        raise ValueError('survival materialized query difference differs from actual SSA cache')
    if cpp.get('global_load_issue_groups','not-null') is not None:raise ValueError('surviving work cannot claim an unreconstructed emitter order')
    footprint=emission.access_coverage(emitter,surviving,info)
    totals={name:sum(x[name] for x in footprint) for name in ('query_vectors','warp_issues','global_sectors32','shared_bank_excess','address_alu_warp_issues')}
    if {name:sum(x[name] for x in cpp.get('accesses',[])) for name in totals}!=totals:raise ValueError('surviving access ledger differs from actual retained SSA')
    boundaries=[];by_constraint={x['constraint']:x for x in cpp['boundaries']};seen=set();scratch=0
    for op,plan in plans:
        source,target=op['operands'][0],op['results'][0]
        if plan['mode']=='Shared':scratch=max(scratch,4*plan['volume'])
        if target not in retained or not observed.get(target):continue
        actual=emission.conversion_structure(emitter,source,target,plan,surviving.get(target,set()),demand)
        ids=[i for i in emitter.v[target].get('layout_boundary_ids',[]) if i in by_constraint]
        if not ids:raise ValueError('retained physical boundary absent from surviving ledger')
        for constraint in ids:
            emission.check_boundary(by_constraint[constraint],actual,constraint)
            for field,key in (('shared_read_warp_issues','read_warp_issues'),('shared_write_warp_issues','write_warp_issues')):
                if by_constraint[constraint].get(field)!=actual[key]:raise ValueError('surviving directional Shared warp fee differs')
            boundaries.append(dict(actual,constraint=constraint));seen.add(constraint)
    for constraint,boundary in by_constraint.items():
        if constraint not in seen and (boundary.get('queries') or any(boundary.get(k,0) for k in ('bytes','barriers','address_alu_warp_issues','shuffle_warp_issues'))):raise ValueError('removed pure boundary still charged')
    expected_communication={i for op in communication for i in emitter.v[op['results'][0]].get('layout_boundary_ids',[]) if i in by_constraint}
    if set(ledger['retained_communication'])!=expected_communication:raise ValueError('retained communication diagnostic differs')
    priced,arithmetic=emission.expected_priced_features(emitter,surviving,totals,boundaries)
    if cpp.get('arithmetic_warp_issues')!=arithmetic or cpp.get('address_alu_warp_issues')!=totals['address_alu_warp_issues']+sum(b['address_alu_warp_issues'] for b in boundaries):raise ValueError('surviving arithmetic/address ledger differs')
    priced.pop('global_load_issue_groups')
    priced['alu_warp_issues']-=sum(b['address_alu_warp_issues']+b['shuffle_warp_issues'] for b in boundaries)
    priced.update(shared_warp_issues=sum(b['read_warp_issues']+b['write_warp_issues'] for b in boundaries if b['route']==3),
                  shuffle_warp_issues=sum(b['shuffle_warp_issues'] for b in boundaries),
                  communication_alu_warp_issues=sum(b['address_alu_warp_issues'] for b in boundaries))
    if require_priced_features:_check_features(record['cost'].get('features'),priced)
    if cpp.get('scratch_max_bytes')!=scratch or info['scratch_max_bytes']!=scratch:raise ValueError('declared scratch removed by survival')
    if require_priced_features and record['cost']['cost'][5]!=scratch:raise ValueError('surviving priced scratch differs')
    return dict(schema_version=1,status='passed',work_basis=WORK_BASIS,native_exact=False,retained_ssa=sorted(retained),removed_ssa=sorted(distributed-retained),
                retained_communication=sorted({i for op in communication for i in emitter.v[op['results'][0]].get('layout_boundary_ids',[])}),
                queries=diagnostics,expected_priced_features=priced,priced_features_checked=require_priced_features,declared_scratch_bytes=scratch,no_gpu=True)
