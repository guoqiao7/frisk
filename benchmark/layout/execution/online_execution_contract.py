"""Shared SM90 execution-domain contract for the packed attention lowerings."""
import json
import re


def _int(value, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f'invalid execution-contract integer {value!r}')
    return value


def single_group(wait_batch=1):
    if wait_batch not in (1, 4) or type(wait_batch) is not int:
        raise ValueError('unsupported synchronous wait batch')
    return dict(version=1, kind='single', cta=256,
                groups=[dict(base=0, threads=256)], roles={k:0 for k in ('q','k','v','qk','pv','reduce')},
                buffers={k:dict(versions=1) for k in ('q','k','v')},
                publication='cta_barrier', batch=wait_batch, wait_pending=0)


# The external JSON identifies FA3's operations and buffers by integer TE DAG ID.
_OP = dict(q=0, k=4, qk=6, v=17, pv=18, reduce=8)
_BUF = dict(q=1, k=6, v=13, o=14)
_TILE_BYTES = 128*128*2

# Per-consumer issue orders in the external FA3 DAG. PV's pipeline
# stage distinguishes same-tile accumulation from QK(next) lookahead.
_FA3_CONSUMER_ORDERS = {
    'fa3_same_tile': (5,6,7,8,9,10,11,12,13,14,15,16,18),
    'fa3_qk_next': (5,6,18,7,8,9,10,12,15,11,13,14,16),
}
FA3_KINDS = frozenset(_FA3_CONSUMER_ORDERS)


def _by_id(items, field):
    table = {i[field]:i for i in items}
    if len(items) != len(table):
        raise ValueError('duplicate plan ID')
    return table


def _bind_fa3_operations(operations):
    ops = _by_id(operations, 'operation_id')
    if set(ops) != set(range(22)): raise ValueError('unknown FA3 DAG')
    roles = {name:ops[idx]['group_id'] for name,idx in _OP.items()}
    # These are the supported prologue, steady-state and drain dependencies.
    # Each group's stage/order is checked before using the fixed arithmetic template.
    before, after = [0,1,2,3], [19,20,21]
    producer = [4,17]
    pv_stage = ops[_OP['pv']]['stage']
    if type(pv_stage) is not int or pv_stage not in (1,2):
        raise ValueError('unsupported PV pipeline stage')
    kind = 'fa3_same_tile' if pv_stage==1 else 'fa3_qk_next'
    loop = _FA3_CONSUMER_ORDERS[kind]
    for ids, gid in ((before,roles['qk']),(loop,roles['qk']),
                     (after,roles['qk']),(producer,roles['k'])):
        if any(ops[i]['group_id'] != gid for i in ids):
            raise ValueError('unexpected operation placement')
        if [ops[i]['order'] for i in ids] != list(range(len(ids))):
            raise ValueError('unsupported operation issue order')
    if any(ops[i]['stage'] is not None for i in before+after) or any(
            ops[i]['stage'] != (0 if i==4 else 1) for i in (*loop,*producer) if i!=_OP['pv']):
        raise ValueError('unsupported pipeline stages')
    return kind,roles


def from_overlap(plan):
    groups = []
    base = 0
    for group in plan['groups']:
        width = 32*_int(group['warp_count'], 1, 32)
        regs = _int(group['register_count'], 24, 256)
        if regs % 8: raise ValueError('invalid register allocation')
        groups.append(dict(base=base, threads=width, regs=regs, increase=group['register_increase']))
        base += width
    kind,roles = _bind_fa3_operations(plan['operations'])
    raw = _by_id(plan['buffers'],'buffer_id')
    if set(b['buffer_id'] for b in raw.values() if b['byte_offset'] is not None) != set(_BUF.values()):
        raise ValueError('unsupported shared buffers')
    buffers = {name:dict(offset=_int(raw[i]['byte_offset'],0,227328),
                         versions=_int(raw[i]['version_count'],1,8), tile_bytes=_TILE_BYTES)
               for name,i in _BUF.items()}
    if any(raw[i]['communication']!=2 for i in _BUF.values()):
        raise ValueError('expected shared communication')
    for i,b in raw.items():
        if i not in _BUF.values() and b['version_count'] != 1: raise ValueError('unsupported private versioning')
    expected = {(0,6):('q_ready','q',2), (4,6):('k_ready','k',1),(17,18):('v_ready','v',1),
                (6,4):('k_free','k',1),(18,17):('v_free','v',1)}
    edges = {}; slots=0
    for e in plan['sync_edges']:
        name,buf,scope=expected[(e['producer_id'],e['consumer_id'])]
        free = name.endswith('_free')
        count=buffers[buf]['versions']
        if (name in edges or e['buffer_id']!=_BUF[buf] or e['byte_offset'] is not None or
            (e['scope'],e['slot_count'],e['kind'],e['completion_mode'],e['dependency_kind'],e['iteration_distance'])
            !=(scope,count,int(free),int(not free),8 if free else 1,count if free else 0)):
            raise ValueError('unsupported or inconsistent wait edge '+name)
        edges[name]=dict(first=slots,slots=count, arrivals=groups[roles['qk']]['threads'] if free else 1,
                         distance=count if free else 0)
        slots+=count
    if set(edges)!=set(v[0] for v in expected.values()): raise ValueError('missing required synchronization')
    out=dict(version=1, kind=kind, cta=base, groups=groups, roles=roles, buffers=buffers,
             edges=edges, barrier_slots=slots, arena=plan['shared_arena_bytes'],batch=8,wait_pending=0)
    validate(out)
    return out


def validate(plan):
    if plan['kind']=='single':
        if plan != single_group(plan['batch']): raise ValueError('unsupported synchronous execution contract')
        return
    if plan['kind'] not in FA3_KINDS or (plan['version'],plan['batch'],plan['wait_pending']) != (1,8,0):
        raise ValueError('unsupported FA3 intrinsic protocol')
    gs=plan['groups'];roles=plan['roles']; c=roles['qk']; p=roles['k']
    if len(gs)!=2 or {c,p}!={0,1} or roles!={'q':c,'qk':c,'pv':c,'reduce':c,'v':p,'k':p}:
        raise ValueError('unsupported FA3 group binding')
    if (gs[c]['threads'],gs[p]['threads'],gs[c]['increase'],gs[p]['increase'])!=(256,128,1,0):
        raise ValueError('unsupported MMA/producer domains')
    if [g['base'] for g in gs]!=[0,gs[0]['threads']] or plan['cta']!=384:
        raise ValueError('invalid CTA partition')
    for g in gs:
        if _int(g['regs'],24,256)%8: raise ValueError('invalid setmaxnreg')
    if sum(g['regs']*g['threads'] for g in gs)>65536: raise ValueError('register budget exceeded')
    bs=plan['buffers'];end=_int(plan['arena'],_TILE_BYTES,227328)
    spans={}
    for k,v in bs.items():
        n=_int(v['versions'],1,8); start=_int(v['offset'],0,end);stop=start+n*_TILE_BYTES
        if start%1024 or stop>end or v['tile_bytes']!=_TILE_BYTES: raise ValueError('invalid TMA ring allocation')
        spans[k]=(start,stop)
    if set(bs)!={'q','k','v','o'} or bs['q']['versions']!=1 or bs['o']['versions']!=1:
        raise ValueError('unsupported Q/O versioning')
    for i,a in enumerate(spans):
        for b in list(spans)[i+1:]:
            if max(spans[a][0],spans[b][0])<min(spans[a][1],spans[b][1]) and not ({a,b}=={'q','o'} and spans[a]==spans[b]):
                raise ValueError('live shared buffers overlap')
    edges=plan['edges'];used=[]
    if set(edges)!={'q_ready','k_ready','v_ready','k_free','v_free'}: raise ValueError('incomplete edges')
    for name,e in edges.items():
        n=bs[name[0]]['versions'];free=name.endswith('_free')
        if (e['slots'],e['arrivals'],e['distance'])!=(n,256 if free else 1,n if free else 0):
            raise ValueError('incorrect barrier versions, arrivals or reuse distance')
        used.extend(range(e['first'],e['first']+n))
    if sorted(used)!=list(range(plan['barrier_slots'])): raise ValueError('barrier slots overlap or have gaps')


def mlir_attr(value):
    if isinstance(value,dict): return '{'+','.join(k+'='+mlir_attr(v) for k,v in value.items())+'}'
    if isinstance(value,list): return '['+','.join(map(mlir_attr,value))+']'
    if type(value) is int: return str(value)+':i64'
    return json.dumps(value)


def bind_module(ir, plan):
    validate(plan)
    if 'frisk.execution_contract' in ir: raise ValueError('input already declares an execution contract')
    ir,n=re.subn(r'\bmodule\s+attributes\s*\{',lambda m:m[0]+'frisk.execution_contract='+mlir_attr(plan)+',',ir,count=1)
    if n!=1: raise ValueError('expected an attributed online module')
    return ir


def packet_contract(data):
    module,=[o for o in data['operations'] if o['name']=='builtin.module']
    plan=module['attrs'].get('frisk.execution_contract')
    if plan is not None:
        validate(plan)
        for o in data['operations']:
            if o['name'] not in ('frisk.copy','frisk.mma','frisk.reduce_tensor'): continue
            a=o['attrs'];role=a.get('frisk.execution_role')
            group=plan['groups'][plan['roles'][role] if role is not None else 0]
            if (role is None and len(plan['groups'])>1) or a.get('frisk.execution_threads')!=group['threads']:
                raise ValueError('layout certificate and execution group disagree')
    return plan
