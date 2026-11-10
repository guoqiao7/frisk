"""Pinned, reviewed projection of pure transport/proof optimizations.

Every accepted current file and sealed source anchor is byte-bound. Projection
must reproduce the complete old module AST; Shared arithmetic/transport is also
compared directly before function substitution. This is a Shared feature bridge,
not a proof that old transport timing or model coefficients remain calibrated.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
MANIFEST=ROOT/'guoqiao/optimize_plan/network_selection_execution_artifacts/development/shared-semantics-reviewed-source-r3/manifest.json'
MANIFEST_SHA256='51a159a414ee176f9734bb6c2c059dcfb8d0e275e5dac051cd8acec6b87b6ff0'
APPROVED={
 'backend.py':{'Emitter.__init__','Emitter._checked_redistribution','Emitter._emit_shuffle_network','Emitter.convert','Emitter.demand_program','Emitter.run'},
 'redistribution.py':{'_network_declaration','validate','content_digest','ProofCache','_UncacheableContent'},
 'demand_coverage_audit.py':{'conversion_structure'},
 'execution_support.py':{'SymbolicMap.__init__','SymbolicMap.first_table','SymbolicMap.logical','normalize_distribution','normalize_packet'},
 'shared_allocation.py':{'plan_shared_allocation','_extent'},
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(node):return ast.dump(node,include_attributes=False)


def dependencies():
    if sha(MANIFEST)!=MANIFEST_SHA256:raise ValueError('review manifest changed')
    m=json.loads(MANIFEST.read_text());deps={str(MANIFEST):MANIFEST_SHA256,str(Path(__file__).resolve()):sha(__file__)}
    if sha(m['anchor_index'])!=m['anchor_index_sha256']:raise ValueError('review source anchor seal changed')
    deps[m['anchor_index']]=m['anchor_index_sha256']
    sealed=json.loads(Path(m['anchor_index']).read_text())
    index={str((ROOT/r['path']).resolve()):r for r in sealed['entries']}
    for name,row in m['files'].items():
        current=ROOT/name;old=Path(m['anchor_root'])/name
        if current.is_symlink() or old.is_symlink() or sha(current)!=row['current_sha256'] or sha(old)!=row['anchor_sha256']:
            raise ValueError('unreviewed source content: '+name)
        entry=index.get(str(old.resolve()))
        if not entry or entry['sha256']!=row['anchor_sha256'] or entry['bytes']!=old.stat().st_size:
            raise ValueError('source anchor is not sealed')
        deps[str(current)]=row['current_sha256'];deps[str(old)]=row['anchor_sha256']
    for name,digest in m['helper_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('unreviewed planner helper')
        deps[str(ROOT/name)]=digest
    for path,digest in m['review_evidence_sha256'].items():
        if Path(path).is_symlink() or sha(path)!=digest:raise ValueError('review evidence changed')
        deps[str(Path(path).resolve())]=digest
    return m,deps


def _fn(tree,name,cls=None):
    body=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    return next(n for n in body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)


def _compact_storage_projection(name,new):
    """Erase only exact Global structural fast paths; retain every dense AST node."""
    imports={
        'backend.py':'from storage_addresses import RowMajorAddresses, needs_storage_normalization',
        'execution_support.py':'from storage_addresses import normalize_storage_values',
        'shared_allocation.py':'from storage_addresses import RowMajorAddresses, needs_storage_normalization',
    }
    if name not in imports:return {}
    expected=ast.parse(imports[name]).body[0]
    found=[n for n in new.body if isinstance(n,ast.ImportFrom) and n.module=='storage_addresses']
    if len(found)!=1 or dump(found[0])!=dump(expected):raise ValueError('unreviewed compact storage import')
    new.body.remove(found[0]);proofs={'exact_import_AST':hashlib.sha256(dump(expected).encode()).hexdigest()}
    if name=='backend.py':
        fn=_fn(new,'_compute_offsets','Emitter')
        guard=ast.parse('if type(a) is RowMajorAddresses:\n return a.byte_offsets()').body[0]
        if len(fn.body)<2 or dump(fn.body[1])!=dump(guard):raise ValueError('unreviewed compact offset branch')
        del fn.body[1];proofs['removed_exact_Global_offset_branch_AST']=hashlib.sha256(dump(guard).encode()).hexdigest()
    elif name=='shared_allocation.py':
        fn=_fn(new,'_extent');guard=ast.parse('if type(addresses) is RowMajorAddresses:\n return addresses.extent()').body[0]
        if len(fn.body)<2 or dump(fn.body[1])!=dump(guard):raise ValueError('unreviewed compact extent branch')
        del fn.body[1];proofs['removed_exact_Global_extent_branch_AST']=hashlib.sha256(dump(guard).encode()).hexdigest()
        fn=_fn(new,'plan_shared_allocation')
        test=ast.parse('isinstance(data, dict) and (data.get("schema_version") == 2 or needs_storage_normalization(data.get("values")))',mode='eval').body
        found=[n for n in fn.body if isinstance(n,ast.If) and dump(n.test)==dump(test)]
        if len(found)!=1:raise ValueError('unreviewed compact allocator normalization condition')
        found[0].test=ast.parse('isinstance(data, dict) and data.get("schema_version") == 2',mode='eval').body
        proofs['removed_exact_Global_normalization_predicate_AST']=hashlib.sha256(dump(test).encode()).hexdigest()
    else:
        fn=_fn(new,'normalize_packet')
        bind=ast.parse('values = normalize_storage_values(data["values"], schema_version=data["schema_version"], materialize=materialize)').body[0]
        found=[n for n in fn.body if dump(n)==dump(bind)]
        if len(found)!=1:raise ValueError('unreviewed compact normalization binding')
        fn.body.remove(found[0]);identity=ast.parse('values is data["values"]',mode='eval').body
        found=[n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp) and isinstance(n.test.op,ast.And) and len(n.test.values)==3 and dump(n.test.values[0])==dump(identity)]
        if len(found)!=1:raise ValueError('unreviewed compact normalization identity condition')
        found[0].test.values=found[0].test.values[1:]
        loads=[n for n in ast.walk(fn) if isinstance(n,ast.Name) and n.id=='values']
        comprehensions=[n for n in ast.walk(fn) if isinstance(n,ast.comprehension) and dump(n.iter)==dump(ast.Name(id='values',ctx=ast.Load()))]
        if len(loads)!=1 or len(comprehensions)!=1:raise ValueError('unreviewed compact normalized value use')
        comprehensions[0].iter=ast.parse('data["values"]',mode='eval').body
        proofs['removed_exact_Global_normalization_binding_AST']=hashlib.sha256(dump(bind).encode()).hexdigest()
    return proofs


def _direct_shared(name,old,new):
    proofs={}
    if name=='backend.py':
        def tail(tree):
            fn=_fn(tree,'convert','Emitter');test=dump(ast.parse("p['mode'] in ('Register','Shuffle')",mode='eval').body)
            places=[i for i,n in enumerate(fn.body) if isinstance(n,ast.If) and dump(n.test)==test]
            if len(places)!=1:raise ValueError('Shared transport boundary changed')
            return ast.Module(body=fn.body[places[0]+1:],type_ignores=[])
        if dump(tail(old))!=dump(tail(new)):raise ValueError('Shared transport tail changed')
        proofs['direct_Shared_tail_AST']=hashlib.sha256(dump(tail(new)).encode()).hexdigest()
        for fn in ('shared','async_copy','load','store','mma','copy','map','address','descriptor','storage','offsets','_compute_offsets'):
            if dump(_fn(old,fn,'Emitter'))!=dump(_fn(new,fn,'Emitter')):raise ValueError('Shared dependency changed: '+fn)
    elif name=='demand_coverage_audit.py':
        def branch(tree):
            found=[n for n in ast.walk(_fn(tree,'conversion_structure')) if isinstance(n,ast.If) and dump(n.test)==dump(ast.parse("mode=='Shared'",mode='eval').body)]
            if len(found)!=1:raise ValueError('Shared bank branch changed')
            return ast.Module(body=found[0].body,type_ignores=[])
        if dump(branch(old))!=dump(branch(new)):raise ValueError('Shared bank arithmetic changed')
        proofs['direct_Shared_bank_AST']=hashlib.sha256(dump(branch(new)).encode()).hexdigest()
    elif name=='shared_allocation.py':
        projected=copy.deepcopy(new);fn=_fn(projected,'plan_shared_allocation')
        if len(fn.args.kwonlyargs)!=1 or fn.args.kwonlyargs[0].arg!='redistribution_validator' or len(fn.args.kw_defaults)!=1 or dump(fn.args.kw_defaults[0])!=dump(ast.Constant(value=None)):
            raise ValueError('unreviewed allocator injection signature')
        fn.args.kwonlyargs=[];fn.args.kw_defaults=[]
        expected=ast.parse("checked_redistribution = validate_redistribution if redistribution_validator is None else redistribution_validator\nif not callable(checked_redistribution):\n raise TypeError('redistribution validator must be callable')").body
        at=1 if fn.body and isinstance(fn.body[0],ast.Expr) and isinstance(fn.body[0].value,ast.Constant) and isinstance(fn.body[0].value.value,str) else 0
        if [dump(n) for n in fn.body[at:at+2]]!=[dump(n) for n in expected]:raise ValueError('allocator validator binding changed')
        del fn.body[at:at+2]
        calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='checked_redistribution']
        if len(calls)!=1:raise ValueError('allocator validation call count changed')
        calls[0].func.id='validate_redistribution'
        if dump(projected)!=dump(old):raise ValueError('allocator arithmetic/lifetime changed')
        proofs['whole_allocator_after_exact_DI_projection_AST']=hashlib.sha256(dump(projected).encode()).hexdigest()
    return proofs


def project(name,new):
    m,deps=dependencies();short=Path(name).name
    if name not in m['files'] or short not in APPROVED:raise ValueError('unreviewed projection module')
    current=ast.parse((ROOT/name).read_text())
    if dump(new)!=dump(current):raise ValueError('projection input is not pinned live source')
    old=ast.parse((Path(m['anchor_root'])/name).read_text());reviewed=copy.deepcopy(new);compact=_compact_storage_projection(short,reviewed);direct=_direct_shared(short,old,reviewed);allowed=APPROVED[short];changes=[]
    def project_body(oldbody,newbody,scope=''):
        named={n.name:n for n in oldbody if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
        imports={dump(n):n for n in oldbody if isinstance(n,(ast.Import,ast.ImportFrom))}
        result=[]
        for node in newbody:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                key=scope+node.name;before=named.get(node.name)
                if before is None:
                    if key not in allowed:raise ValueError('unreviewed added definition: '+key)
                    changes.append(key);continue
                if isinstance(node,ast.ClassDef) and key not in allowed:
                    node=copy.deepcopy(node);node.body=project_body(before.body,node.body,key+'.')
                    if dump(node)!=dump(before):raise ValueError('unreviewed class structure: '+key)
                elif dump(node)!=dump(before):
                    if key not in allowed:raise ValueError('unreviewed function: '+key)
                    changes.append(key);node=copy.deepcopy(before)
            elif isinstance(node,(ast.Import,ast.ImportFrom)):
                before=imports.get(dump(node))
                if before is None and short=='backend.py' and isinstance(node,ast.ImportFrom) and node.module in ('redistribution','shuffle_network'):
                    found=[n for n in oldbody if isinstance(n,ast.ImportFrom) and n.module==node.module]
                    if len(found)==1:before=found[0]
                if before is None:raise ValueError('unreviewed new import')
                if dump(node)!=dump(before):
                    if short!='backend.py' or not isinstance(node,ast.ImportFrom) or node.module not in ('redistribution','shuffle_network'):
                        raise ValueError('unreviewed import change')
                    changes.append('import:'+node.module);node=copy.deepcopy(before)
            result.append(node)
        return result
    projected=ast.Module(body=project_body(old.body,copy.deepcopy(reviewed.body)),type_ignores=[])
    if dump(projected)!=dump(old):raise ValueError('whole-module projection differs: '+name)
    new.body=projected.body
    return dict(contract='reviewed-content-proof-network-compact-Global-projection-v2',manifest_sha256=MANIFEST_SHA256,
        anchor_sha256=m['files'][name]['anchor_sha256'],current_sha256=m['files'][name]['current_sha256'],
        reviewed_changes=changes,compact_Global_projection=compact,direct_shared_proofs=direct,projected_module_AST_sha256=hashlib.sha256(dump(projected).encode()).hexdigest())


def helper_proofs():
    proofs={}
    for short in ('execution_support.py','shared_allocation.py'):
        name='benchmark/layout/execution/'+short;tree=ast.parse((ROOT/name).read_text());proofs[name]=project(name,tree)
    return proofs
