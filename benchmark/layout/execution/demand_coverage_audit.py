"""Independent actual-backend query audit for C++ GF(2) demand ledgers.
No model/cost-analysis implementation is imported. C++ affine signatures are
compared against queries observed while the existing backend actually lowers
128-thread instructions. Graph-variable aliases map to one exported SSA value.
"""
import argparse
import hashlib
import json
from pathlib import Path
import backend


def canonical_query(indices):
    if not isinstance(indices,(tuple,list)) or len(indices)!=128:
        raise ValueError('query must enumerate all128threads')
    if any(type(v) is not int or v<0 for v in indices):raise ValueError('invalid register index')
    base,bits=backend.linear(indices)
    return (base,*bits)


def parse_values(ledger):
    rows=ledger.get('values')
    if not isinstance(rows,list):raise ValueError('ledger values must be list')
    result={}
    for row in rows:
        variable=row.get('variable')
        if type(variable) is not int or variable<0 or variable in result:
            raise ValueError('invalid or duplicate ledger variable')
        queries=row.get('queries')
        if not isinstance(queries,list):raise ValueError('queries must be explicit list')
        result[variable]=set()
        for query in queries:
            terms=query.get('terms') if isinstance(query,dict) else None
            if not isinstance(terms,list) or len(terms)!=8 or any(type(t) is not int or t<0 for t in terms):
                raise ValueError('query must contain constant and7threadcolumns')
            signature=tuple(terms)
            if signature in result[variable]:raise ValueError('duplicate query within ledger variable')
            result[variable].add(signature)
    return result



def parse_boundary_queries(ledger):
    rows=ledger.get('boundaries')
    if not isinstance(rows,list):raise ValueError('ledger boundaries must be explicit list')
    converted=[]
    for row in rows:
        converted.append(dict(variable=row.get('constraint'),queries=row.get('queries')))
    return parse_values({'values':converted})


def compare_queries(values,expected,observed,expected_boundaries=None):
    expected_boundaries={} if expected_boundaries is None else expected_boundaries
    mapped={};owners={};boundary_owners={};diagnostics=[]
    for value in values:
        if 'distributed' not in value:continue
        ident=value['id'];ids=value.get('layout_variable_ids',[]);boundaries=value.get('layout_boundary_ids',[])
        for items in (ids,boundaries):
            if not isinstance(items,list) or any(type(i) is not int or i<0 for i in items) or len(items)!=len(set(items)):
                raise ValueError('missing or invalid exported layout mapping IDs')
        target=set()
        for items,owner_map,ledger_map in ((ids,owners,expected),(boundaries,boundary_owners,expected_boundaries)):
            for variable in items:
                if variable in owner_map and owner_map[variable]!=ident:
                    raise ValueError('one graph variable/boundary mapped to multiple actual SSA values')
                owner_map[variable]=ident;target.update(ledger_map.get(variable,set()))
        actual=observed.get(ident,set());mapped[ident]=target
        if not ids and not boundaries and (target or actual):raise ValueError('actual queried SSA lacks graph mapping')
        diagnostics.append(dict(ssa=ident,variables=ids,boundaries=boundaries,expected_queries=len(target),observed_queries=len(actual),
            missing=[list(x) for x in sorted(target-actual)],unexpected=[list(x) for x in sorted(actual-target)]))
    if set(observed)-set(mapped):raise ValueError('observed non-distributed or missing SSA')
    unmapped=[variable for variable,queries in expected.items() if queries and variable not in owners]
    if unmapped:raise ValueError('nonempty C++ demand variable has no exported SSA: '+str(unmapped))
    unmapped=[variable for variable,queries in expected_boundaries.items() if queries and variable not in boundary_owners]
    if unmapped:raise ValueError('nonempty C++ demand boundary has no exported SSA: '+str(unmapped))
    mismatched=[x for x in diagnostics if x['missing'] or x['unexpected']]
    if mismatched:raise ValueError('actual affine demand query mismatch: '+json.dumps(mismatched))
    return diagnostics,mapped


def access_coverage(emitter,observed,info):
    """Reconstruct active logical warp-value requests and worst root residues.
    These are model footprint units, not native/PTX-byte instruction counters.
    """
    from execution_support import normalize_packet
    result=[];global_roots=set(info['root_order'])
    for op in emitter.data['operations']:
        name=op['name']
        if name not in ('frisk.tile_load','frisk.tile_store'):continue
        read=name=='frisk.tile_load'
        value=op['results'][0] if read else op['operands'][0]
        storage=op['operands'][0] if read else op['operands'][1]
        enc=emitter.v[value]['distributed'];nr=enc['topology'][0]
        if read:
            signatures=observed.get(value,set())
            queries=[tuple(terms[0]^_xor_columns(terms[1:],t) for t in range(128)) for terms in sorted(signatures)]
        else:queries=[(reg,)*128 for reg in range(nr)]
        offsets=emitter.offsets(storage);root=emitter.storage(storage)['root']
        alignment=min(32,emitter.storage(root)['alignment']);is_global=root in global_roots
        if type(alignment) is not int or alignment<=0:raise ValueError('invalid root alignment')
        warps=sectors=banks=address_alu=0
        for indices in queries:
            logical=[enc['logical'][t*nr+i] for t,i in enumerate(indices)]
            addresses=[offsets[slot] for slot in logical]
            active=[read or enc['first'][slot]==t*nr+indices[t] for t,slot in enumerate(logical)]
            work=map_work(addresses)+2
            if not read:work+=map_work([enc['first'][slot] for slot in logical])+map_work([t*nr+indices[t] for t in range(128)])+1
            address_alu+=4*work
            max_sectors=max_banks=0
            for phase in range(0,32,alignment):
                phase_sectors=phase_banks=0
                for warp in range(4):
                    touched=set();words={};any_active=False
                    for t in range(warp*32,warp*32+32):
                        if not active[t]:continue
                        any_active=True;begin=addresses[t]+phase
                        touched.update(range(begin//32,(begin+3)//32+1))
                        for word in range(begin//4,(begin+3)//4+1):words.setdefault(word%32,set()).add(word)
                    if phase==0 and any_active:warps+=1
                    phase_sectors+=len(touched)
                    phase_banks+=max((len(x) for x in words.values()),default=1)-1
                max_sectors=max(max_sectors,phase_sectors);max_banks=max(max_banks,phase_banks)
            sectors+=max_sectors;banks+=max_banks
        result.append(dict(ssa=value,storage=storage,access='read' if read else'write',query_vectors=len(queries),
            warp_issues=warps,global_sectors32=sectors if is_global else 0,shared_bank_excess=0 if is_global else banks,address_alu_warp_issues=address_alu))
    return result


def _xor_columns(columns,thread):
    result=0
    for i,column in enumerate(columns):
        if (thread>>i)&1:result^=column
    return result



def map_work(table):
    """Number of instructions in the backend affine map/elect grammar."""
    _,columns=backend.linear(table)
    return 1+4*sum(bool(x) for x in columns)


def query_table(signature):
    return [signature[0]^_xor_columns(signature[1:],t) for t in range(128)]


def conversion_structure(emitter,source,target,plan,queries,demand):
    mode=plan['mode'];nr=plan['source_registers'];dr=plan['destination_registers']
    value=dict(source_ssa=source,target_ssa=target,route={'Register':1,'Shuffle':2,'Shared':3}[mode],
        source_vectors=0,target_vectors=len(queries),bytes=0,barriers=0,address_alu_warp_issues=0,
        receiver_mux_alu_warp_issues=0,shuffle_warp_issues=0,shared_bank_excess=0,
        shared_read_bank_excess=0,shared_write_bank_excess=0,write_warp_issues=0,read_warp_issues=0)
    if demand and mode=='Register':return value # Actual index substitution, no carrier mux.
    value.update(source_vectors=nr,target_vectors=dr)
    if mode=='Shared':
        writers=0
        for tensor,write in ((source,True),(target,False)):
            excess=warps=0
            for reg,slots in enumerate(emitter.tensor_slots(tensor)):
                offsets=[4*slot for slot in slots]
                if write:
                    owners=[(plan['global_writers'][slot]['warp']*32+plan['global_writers'][slot]['lane'])*nr+plan['global_writers'][slot]['reg'] for slot in slots]
                    current=[t*nr+reg for t in range(128)]
                    active=[owner==here for owner,here in zip(owners,current)]
                    writers+=sum(active);work=map_work(offsets)+map_work(owners)+map_work(current)+3
                else:active=[True]*128;work=map_work(offsets)+2
                value['address_alu_warp_issues']+=4*work
                for warp in range(4):
                    words={};any_active=False
                    for thread in range(warp*32,warp*32+32):
                        if not active[thread]:continue
                        any_active=True;slot=slots[thread];words.setdefault(slot%32,set()).add(slot)
                    warps+=int(any_active);excess+=max((len(x) for x in words.values()),default=1)-1
            value['shared_write_bank_excess' if write else 'shared_read_bank_excess']=excess
            value['write_warp_issues' if write else 'read_warp_issues']=warps
        value['shared_bank_excess']=value['shared_read_bank_excess']+value['shared_write_bank_excess']
        value.update(bytes=4*(writers+128*dr),barriers=2)
    else:
        network=None
        if mode=='Shuffle' and emitter.shuffle_network_policy=='auto':
            from shuffle_network import plan_shuffle_network
            network=plan_shuffle_network(plan,objective=plan.get('shuffle_network_objective','min-shfl'))
        if network is not None:
            # Count the actual independent Python emission grammar. No C++
            # cost model, predicted totals or native latency model is imported.
            trace=backend.Emitter({'values':[]})
            trace._emit_shuffle_network(network,['audit_source_'+str(r) for r in range(nr)])
            shfl=sum(line.startswith('shfl.sync.') for line in trace.code)
            value['shuffle_warp_issues']=4*shfl
            value['address_alu_warp_issues']=4*(len(trace.code)-shfl)
            value['shuffle_algorithm']=network['kind']
        for reg in ([] if network is not None else range(dr)):
            owners=[plan['destinations'][t*dr+reg]['source'] for t in range(128)]
            work=map_work([o['reg'] for o in owners])+1+2*(nr-1)
            if mode=='Shuffle':
                from redistribution import shuffle_source_support
                count=len(shuffle_source_support(plan,reg))
                work=map_work([o['reg'] for o in owners])+1+2*(count-1)+map_work([o['lane'] for o in owners])
                value['shuffle_warp_issues']+=4*count
            value['address_alu_warp_issues']+=4*work
        if mode=='Shuffle':
            value['bytes']=value['shuffle_warp_issues']*32*4
    if demand:
        for query in queries:
            if any(query[1:]):value['receiver_mux_alu_warp_issues']+=4*(map_work(query_table(query))+1+2*(dr-1))
        value['address_alu_warp_issues']+=value['receiver_mux_alu_warp_issues']
    return value


def check_boundary(cpp,actual,constraint):
    fields=('route','source_vectors','target_vectors','bytes','barriers','address_alu_warp_issues',
        'receiver_mux_alu_warp_issues','shuffle_warp_issues','shared_bank_excess','shared_read_bank_excess','shared_write_bank_excess')
    for name in fields:
        if type(cpp.get(name)) is not int or cpp[name]!=actual[name]:
            raise ValueError('actual conversion boundary '+name+' differs for constraint'+str(constraint))


def boundary_coverage(emitter,observed,ledger,demand):
    actual={};expected={x['constraint']:x for x in ledger['boundaries']};declared_scratch=0
    for op in emitter.data['operations']:
        if op['name']!='frisk.convert_layout':continue
        source=op['operands'][0];target=op['results'][0];plan=backend.validate_redistribution(op,emitter.v)
        if plan['mode']=='Shared':declared_scratch=max(declared_scratch,4*plan['volume'])
        if demand and not observed.get(target):continue
        ids=[i for i in emitter.v[target].get('layout_boundary_ids',[]) if i in expected]
        if not ids:raise ValueError('actual conversion lacks C++ boundary mapping')
        value=conversion_structure(emitter,source,target,plan,observed.get(target,set()),demand)
        for constraint in ids:
            if constraint in actual:raise ValueError('one boundary mapped to multiple physical conversions')
            entry=dict(value,constraint=constraint);check_boundary(expected[constraint],entry,constraint);actual[constraint]=entry
    missing=set(expected)-set(actual)
    if missing:raise ValueError('C++ boundaries have no actual backend conversion: '+str(sorted(missing)))
    return list(actual.values()),declared_scratch


FEATURE_NAMES=('access_warp_issues','global_sectors32','shared_bank_excess','copy_setup_units','mma_instructions',
    'reduction_element_ops','alu_warp_issues','div_warp_issues','exp_warp_issues','cast_warp_issues',
    'conversion_register_bytes','conversion_shuffle_bytes','conversion_shared_bytes','conversion_conservative_bytes',
    'synchronization_barriers','shared_read_bank_excess','shared_write_bank_excess','global_load_issue_groups')


def expected_priced_features(emitter,observed,totals,boundaries,*,load_groups=0):
    features=dict.fromkeys(FEATURE_NAMES,0);features['global_load_issue_groups']=load_groups
    features.update(access_warp_issues=totals['warp_issues'],global_sectors32=totals['global_sectors32'],
        shared_bank_excess=totals['shared_bank_excess'],alu_warp_issues=totals['address_alu_warp_issues'])
    arithmetic=0
    for op in emitter.data['operations']:
        name=op['name']
        if name in ('arith.constant','arith.mulf','arith.subf','arith.divf','math.exp'):
            work=4*len(observed.get(op['results'][0],set()));arithmetic+=work
            features[{'arith.divf':'div_warp_issues','math.exp':'exp_warp_issues'}.get(name,'alu_warp_issues')]+=work
        elif name.startswith(('arith.','math.')) and name!='arith.constant':raise ValueError('unexpected arithmetic in exact demand scope')
    for boundary in boundaries:
        features['alu_warp_issues']+=boundary['address_alu_warp_issues']+boundary['shuffle_warp_issues']
        for name in ('shared_bank_excess','shared_read_bank_excess','shared_write_bank_excess'):features[name]+=boundary[name]
        features['synchronization_barriers']+=boundary['barriers']
        if boundary['route']==2:features['conversion_shuffle_bytes']+=boundary['bytes']
        if boundary['route']==3:features['conversion_shared_bytes']+=boundary['bytes']
    return features,arithmetic


def global_load_issue_groups(code):
    """Parse the existing scalar f32 PTX emission grammar in actual order.
    Four byte loads form one request. Address construction/byte assembly and
    literal moves do not split a run; actual compute/communication/store does.
    """
    groups=requests=byte_loads=partial=cuts=0;in_run=False
    for line in code:
        text=line.strip()
        if not text or text.startswith('//'):continue
        if text.startswith('@'):
            parts=text.split(None,1)
            if len(parts)!=2:raise ValueError('malformed predicated PTX grammar')
            text=parts[1]
        opcode=text.split(None,1)[0].rstrip(';')
        if opcode=='ld.global.u8':
            byte_loads+=1;partial+=1
            if partial==4:
                if not in_run:groups+=1;in_run=True
                requests+=1;partial=0
            continue
        if opcode.startswith('ld.global.'):
            raise ValueError('load-group grammar requires scalar f32 byte loads')
        arithmetic=opcode.startswith(('mul.','sub.','div.','ex2.')) and opcode.endswith('.f32')
        cut=arithmetic or opcode.startswith(('shfl.','ld.shared.','st.shared.','bar.','st.global.'))
        if cut:
            if partial:raise ValueError('load-group boundary splits a four-byte f32 request')
            cuts+=1;in_run=False
    if partial:raise ValueError('incomplete four-byte f32 load request')
    return dict(global_load_issue_groups=groups,f32_load_requests=requests,global_byte_loads=byte_loads,cut_instructions=cuts)


def check_priced_features(priced,expected,*,legacy_17=False):
    names=FEATURE_NAMES[:-1] if legacy_17 else FEATURE_NAMES
    if not isinstance(priced,dict) or set(priced)-set(names):raise ValueError('invalid/unknown priced feature ledger')
    for name in names:
        value=expected[name]
        if type(priced.get(name,0)) is not int or priced.get(name,0)!=value:
            raise ValueError('priced feature does not match actual lowering: '+name)


def check_scratch(ledger,info,declared,record,priced):
    if ledger.get('scratch_max_bytes')!=declared or info['scratch_max_bytes']!=declared:
        raise ValueError('declared serial Shared scratch differs; dead conversions must reserve scratch')
    if priced and record['cost']['cost'][5]!=declared:raise ValueError('priced Shared scratch differs from backend allocation')


def audit(packet,record,*,require_priced_features=False):
    ledger=record.get('demand_coverage')
    if not isinstance(ledger,dict) or type(ledger.get('exact')) is not bool:
        raise ValueError('missing explicit demand_coverage ledger')
    observed={};events=[]
    def observe(value,indices):
        signature=canonical_query(indices)
        if signature in observed.setdefault(value,set()):raise ValueError('backend repeated a supposedly cached query')
        observed[value].add(signature);events.append((value,signature))
    emitter=backend.Emitter(packet,demand_query_observer=observe);text,info=emitter.run()
    if not ledger['exact']:
        if not isinstance(ledger.get('reason'),str) or not ledger['reason']:raise ValueError('fallback lacks explicit reason')
        return dict(schema_version=1,status='explicit-fallback',exact=False,reason=ledger['reason'],
            query_equivalence_applicable=False,backend_demand=info.get('demand_lowering'),no_gpu=True)
    expected=parse_values(ledger);expected_boundaries=parse_boundary_queries(ledger);demand=info.get('demand_lowering')
    if bool(demand)!=ledger.get('demand_lowering'):raise ValueError('model/backend demand mode differs')
    if not demand:
        # Eager path emits every register carrier, unlike the consumer-demand path.
        observed={v['id']:{(reg,0,0,0,0,0,0,0) for reg in range(v['distributed']['topology'][0])}
                  for v in emitter.v.values() if 'distributed'in v}
    diagnostics,mapped=compare_queries(emitter.data['values'],expected,observed,expected_boundaries)
    footprint=access_coverage(emitter,observed,info)
    boundaries,scratch=boundary_coverage(emitter,observed,ledger,bool(demand))
    totals={name:sum(x[name] for x in footprint) for name in ['query_vectors','warp_issues','global_sectors32','shared_bank_excess','address_alu_warp_issues']}
    accesses=ledger.get('accesses')
    if not isinstance(accesses,list):raise ValueError('explicit access ledger required')
    cpp={name:sum(x[name] for x in accesses) for name in totals}
    if cpp!=totals:raise ValueError('backend access footprint differs: '+json.dumps(dict(cpp=cpp,backend=totals)))
    if demand and len(events)!=demand['materialized_query_vectors']:raise ValueError('observer event count differs from backend cache')
    loads=global_load_issue_groups(emitter.code)
    if loads['f32_load_requests']!=sum(row['query_vectors'] for row in footprint if row['access']=='read'):
        raise ValueError('actual PTX f32 load request count differs from query observer')
    semantic=record.get('cost_model','');legacy_17=semantic.startswith('sm90-layout-calibrated-v4d-')
    if 'global_load_issue_groups' in ledger:
        if type(ledger['global_load_issue_groups']) is not int or ledger['global_load_issue_groups']!=loads['global_load_issue_groups']:raise ValueError('actual PTX global_load_issue_groups differs')
    elif semantic.startswith('sm90-layout-calibrated-v5d-'):
        raise ValueError('new v5d ledger lacks global_load_issue_groups')
    expected_features,arithmetic=expected_priced_features(emitter,observed,totals,boundaries,load_groups=loads['global_load_issue_groups'])
    if ledger.get('arithmetic_warp_issues')!=arithmetic:raise ValueError('actual arithmetic query work differs')
    if ledger.get('address_alu_warp_issues')!=totals['address_alu_warp_issues']+sum(b['address_alu_warp_issues'] for b in boundaries):
        raise ValueError('actual address/mux grammar work differs')
    check_scratch(ledger,info,scratch,record,require_priced_features)
    if require_priced_features:check_priced_features(record['cost']['features'],expected_features,legacy_17=legacy_17)
    return dict(schema_version=1,status='passed',exact=True,mode='actual-demand-query-observer' if demand else'eager-full-carriers',
        query_equivalence_applicable=True,values=diagnostics,backend_query_vectors=sum(len(q) for q in observed.values()),
        raw_graph_variable_queries=sum(len(q) for q in expected.values()),raw_graph_boundary_queries=sum(len(q) for q in expected_boundaries.values()),mapped_unique_queries=sum(len(q) for q in mapped.values()),
        accesses=footprint,access_totals=totals,conversion_boundaries=boundaries,shared_boundaries=[b for b in boundaries if b['route']==3],
        declared_scratch_bytes=scratch,expected_priced_features=expected_features,load_issue_grammar=loads,
        priced_feature_version='legacy17' if legacy_17 else 'demand18',priced_features_checked=require_priced_features,
        emitted_ttgir_sha256=hashlib.sha256(text.encode()).hexdigest(),no_gpu=True,
        units='logical active warp-value requests and worst root-alignment residue footprints; not native instruction/cycle measurements')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('packet',type=Path);parser.add_argument('record',type=Path)
    parser.add_argument('--out',type=Path);parser.add_argument('--require-priced-features',action='store_true');args=parser.parse_args()
    result=audit(json.loads(args.packet.read_text()),json.loads(args.record.read_text()),require_priced_features=args.require_priced_features)
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.out:args.out.write_text(text)
    else:print(text,end='')
if __name__=='__main__':main()
