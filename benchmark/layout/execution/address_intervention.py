"""Controlled source Global-address intervention + complete native static work.

Two independent proofs are required: source PTX differs only inside pure,
exclusively Global-address definition/use slices; compiled native opcode,
physical-register, predicate, control, callee and resource structure is equal.
This does not prove native address DU, dynamic lane masks/slowpath frequency,
cache/TLB misses or input-value equality per hardware lane.
"""
from __future__ import annotations
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import subprocess
import access_reuse
import execution_cost_summary as complete

CONTRACT = "frisk-source-global-address-intervention-v1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PURE_ARITY = {"mov.u32":2,"mov.b32":2,"bfe.u32":4,"neg.s32":2,"and.b32":3,"xor.b32":3,"cvt.u64.u32":2,"add.u64":3,"add.u32":3}
PURE = set(PURE_ARITY)
EXTERNAL_SPECIALS = {"%tid.x"}
TOKEN = re.compile(r"(?<![A-Za-z0-9_])%?[A-Za-z_$][A-Za-z0-9_.$]*(?![A-Za-z0-9_])")
SUPPORT_SOURCES = {"access_reuse.py":access_reuse.SOURCE_SHA256,"execution_cost_summary.py":complete.SOURCE_SHA256}
NUMERIC = re.compile(r"(?:0x[0-9a-fA-F]+|[0-9]+)")


def sha(data): return hashlib.sha256(data).hexdigest()

def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _split(text):
    parts=[];start=depth=0
    for i,c in enumerate(text):
        if c in "[{(":depth+=1
        elif c in "]})":depth-=1
        elif c=="," and not depth:parts.append(text[start:i].strip());start=i+1
        if depth<0:raise ValueError("unbalanced PTX operand")
    if depth:raise ValueError("unbalanced PTX operand")
    if text[start:].strip():parts.append(text[start:].strip())
    return parts


def parse_source(ptx):
    """One entry, complete nondebug body; no PTX opaque calls or branches."""
    text=re.sub(r"/\*.*?\*/", "", ptx, flags=re.S)
    text=re.sub(r"//[^\n]*", "", text)
    if len(re.findall(r"\.entry\s+frisk_execution\s*\(", text))!=1 or re.search(r"\.func\b",text):raise ValueError("one scalar PTX entry without opaque functions required")
    start=text.index(".visible .entry frisk_execution")
    left=text.index("{",start);depth=1;right=left+1
    while right<len(text) and depth:
        depth+=(text[right]=="{")-(text[right]=="}");right+=1
    if depth:raise ValueError("unclosed PTX function")
    header=" ".join(text[:left].split());body=text[left+1:right-1]
    parameters={}
    for declaration in _split(header[header.index("(")+1:header.rindex(")")]):
        match=re.search(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*$",declaration)
        if not declaration.startswith(".param ") or not match or match[1] in parameters:raise ValueError("bounded named PTX parameters required")
        parameters[match[1]]=declaration
    types={};declarations=[];instructions=[];labels=[];symbols=set()
    for line in body.splitlines():
        line=line.strip()
        if not line or line in {"{","}"} or line.startswith(".loc"):continue
        if line.startswith(".reg"):
            m=re.fullmatch(r"\.reg\s+(\.[A-Za-z0-9]+)\s+(.+);",line)
            if not m:raise ValueError("unknown register declaration")
            for name in _split(m[2]):
                vector=re.fullmatch(r"([%A-Za-z_$][A-Za-z0-9_$]*)<(\d+)>",name)
                names=[vector[1]+str(i) for i in range(int(vector[2]))] if vector else [name]
                for reg in names:
                    if reg in types:raise ValueError("duplicate PTX register declaration")
                    types[reg]=m[1]
            continue
        if line.startswith("."):
            declarations.append(" ".join(line.split()))
            symbol=re.search(r"\s([A-Za-z_$][A-Za-z0-9_$]*)\s*\[",line)
            if symbol:symbols.add(symbol[1])
            continue
        if line.endswith(":"):
            labels.append((len(instructions),line));continue
        m=re.fullmatch(r"(?:@(!?[%A-Za-z_$][A-Za-z0-9_.$]*)\s+)?([a-z][a-z0-9_.:]*)(?:\s+(.*?))?;",line)
        if not m:raise ValueError("unparsed PTX source statement:"+line[:100])
        op=m[2];operands=_split(m[3] or "")
        if op in PURE_ARITY and len(operands)!=PURE_ARITY[op]:raise ValueError("pure-address opcode arity differs:"+op)
        for operand in [*operands,m[1] or ""]:
            for token in TOKEN.finditer(operand):
                word=token.group()
                if word.startswith("%") and word not in types and word not in EXTERNAL_SPECIALS:raise ValueError("unknown external PTX special/register:"+word)
        if op in PURE_ARITY:
            for operand in operands[1:]:
                if operand not in types and operand not in EXTERNAL_SPECIALS and operand not in symbols and not NUMERIC.fullmatch(operand):raise ValueError("unknown pure-address symbol/literal:"+operand)
        if op.startswith("ld.param."):
            param=re.fullmatch(r"\[([A-Za-z_$][A-Za-z0-9_$]*)\]",operands[1] if len(operands)>1 else "")
            if not param or param[1] not in parameters:raise ValueError("unknown PTX entry parameter")
        if op.startswith(("bra","call","brx","jmp")):raise ValueError("source control outside straight-line scalar domain")
        store=op.startswith("st.");no_dest=store or op.startswith(("bar.","fence.","membar.","ret"))
        if not no_dest and not operands:raise ValueError("missing PTX destination")
        writes=[] if no_dest else [t.group() for t in TOKEN.finditer(operands[0]) if t.group() in types]
        if not no_dest and len(writes)!=1:raise ValueError("single scalar PTX destination required")
        reads=[]
        for oi,operand in enumerate(operands):
            if not no_dest and oi==0:continue
            role="global-address" if ((op.startswith("ld.global.") and oi==1) or (op.startswith("st.global.") and oi==0)) else "ordinary"
            reads.extend((t.group(),role) for t in TOKEN.finditer(operand) if t.group() in types)
        if m[1]:reads.extend((t.group(),"predicate") for t in TOKEN.finditer(m[1]) if t.group() in types)
        instructions.append(dict(op=op,operands=operands,predicate=m[1] or "",writes=writes,reads=reads))
    return dict(header=header,declarations=declarations,types=types,instructions=instructions,labels=labels,parameters=parameters)


def source_signature(ptx):
    """Versioned DU excludes only pure exclusive Global address definitions."""
    try:
        parsed=parse_source(ptx);ins=parsed["instructions"];types=parsed["types"]
        last={};deps=[];uses=[[] for _ in ins];global_seeds=[]
        for i,x in enumerate(ins):
            row=[]
            for reg,role in x["reads"]:
                if reg not in last:raise ValueError("uninitialized source register:"+reg)
                parent=last[reg];row.append((parent,role));uses[parent].append((i,role))
                if role=="global-address":global_seeds.append(parent)
            if x["predicate"]:
                for reg in x["writes"]:
                    if reg in last:row.append((last[reg],"conditional-retention"));uses[last[reg]].append((i,"conditional-retention"))
            deps.append(row)
            for reg in x["writes"]:last[reg]=i
        candidates=set();todo=list(global_seeds);seen=set()
        while todo:
            i=todo.pop()
            if i in seen:continue
            seen.add(i);x=ins[i]
            if x["op"] in PURE and not x["predicate"]:
                candidates.add(i);todo.extend(p for p,_ in deps[i])
        removed=deque(i for i in candidates if any(role!="global-address" and target not in candidates for target,role in uses[i]))
        while removed:
            i=removed.popleft()
            if i not in candidates:continue
            candidates.remove(i)
            for parent,_ in deps[i]:
                if parent in candidates:removed.append(parent)
        def roots_of(i):
            roots=set();todo=[i];seen=set()
            while todo:
                node=todo.pop()
                if node in seen:continue
                seen.add(node);x=ins[node]
                if x["op"].startswith("ld.param."):
                    roots.add(x["operands"][1]);continue
                if x["op"] not in PURE or x["predicate"]:raise ValueError("address root outside pure parameter/thread map:"+x["op"])
                if x["op"].startswith("mov."):
                    source=x["operands"][1]
                    if source in EXTERNAL_SPECIALS:continue
                    if source not in types and not NUMERIC.fullmatch(source):raise ValueError("address-root-unknown-external-symbol:"+source)
                todo.extend(p for p,_ in deps[node])
            return sorted(roots)
        # Canonicalize remaining virtual register *versions*, not the temporary
        # spelling. Address-only temporaries and their declarations disappear.
        versions={};serial=0;out=[];removed_literals=[];address_terminals=[]
        def renamed(operand):
            def replace(m):
                word=m.group()
                if word in types:
                    if word not in versions:raise ValueError("nonaddress source uses removed address definition")
                    return versions[word]
                return word
            return TOKEN.sub(replace,operand)
        last_all={}
        for i,x in enumerate(ins):
            if i in candidates:
                if x["op"]=="mov.u32" and NUMERIC.fullmatch(x["operands"][1]):removed_literals.append(dict(instruction=i,operand=1,opcode=x["op"]))
                if x["op"]=="and.b32" and NUMERIC.fullmatch(x["operands"][2]):removed_literals.append(dict(instruction=i,operand=2,opcode=x["op"]))
                for reg in x["writes"]:last_all[reg]=i
                continue
            operands=[]
            for oi,operand in enumerate(x["operands"]):
                global_address=(x["op"].startswith("ld.global.") and oi==1) or (x["op"].startswith("st.global.") and oi==0)
                if global_address:
                    m=re.fullmatch(r"\[([%A-Za-z_$][A-Za-z0-9_.$]*)([+-][0-9]+)?\]",operand)
                    if not m or m[1] not in last_all:raise ValueError("bounded register+literal Global address required")
                    roots=roots_of(last_all[m[1]])
                    if len(roots)!=1:raise ValueError("one pointer parameter root per Global address required")
                    tag=["Global-address",roots[0],m[2] or "+0"]
                    operands.append(tag);address_terminals.append(dict(instruction=i,root=roots[0],byte_suffix=m[2] or "+0",opcode=x["op"]))
                elif oi==0 and x["writes"]:operands.append(None)
                else:operands.append(renamed(operand))
            predicate=renamed(x["predicate"])
            retention=[]
            for reg in x["writes"]:
                if x["predicate"] and reg in versions:retention.append(versions[reg])
                name="v"+str(serial);serial+=1;versions[reg]=name;operands[0]=[name,types[reg]];last_all[reg]=i
            out.append([x["op"],predicate,operands,retention])
        # Labels have no control users in this supported source domain, while
        # barriers, predicates and every nonaddress operation stay in out.
        body=dict(header=parsed["header"],declarations=parsed["declarations"],instructions=out)
        return dict(status="available",nonaddress_sha256=sha(canonical(body)),source_instructions=len(ins),
                    removed_address_instructions=len(candidates),preserved_nonaddress_instructions=len(out),
                    address_terminals=len(address_terminals),address_terminal_order_sha256=sha(canonical(address_terminals)),
                    normalized_source_sha256=sha(canonical(parsed)),
                    proof=dict(typed_virtual_version_alpha_renaming=True,pure_exclusive_global_address_slice=True,
                               all_nonaddress_and_predicate_operations_preserved=True,global_pointer_roots_and_byte_suffixes_preserved=True,
                               address_slice_instruction_ids=sorted(candidates),address_literal_sites=removed_literals))
    except (KeyError,TypeError,ValueError,IndexError) as error:return dict(status="unsupported",reasons=[str(error)])


def native_signature(document, *, document_provider=None):
    result=(complete.analyze_document(document) if document_provider is None else
            document_provider.pure(complete.analyze_document, document))
    if result["status"]!="complete":return dict(status="unsupported",reasons=result.get("reasons",[]))
    functions=[];sites=0
    for function in document[1]:
        f={k:v for k,v in function.items() if k!="sass-instructions"};instructions=[]
        for x in function["sass-instructions"]:
            row=dict(x)
            if function["function-name"]=="frisk_execution" and x.get("opcode")=="LOP3.LUT":
                args=x.get("operands","").split(",")
                if len(args)==6 and re.fullmatch(r"0x[0-9a-fA-F]+",args[2]) and 0<=int(args[2],16)<1<<18:
                    args[2]="<address-mask-candidate>";row["operands"]=",".join(args);sites+=1
            instructions.append(row)
        f["sass-instructions"]=instructions;functions.append(f)
    return dict(status="available",skeleton_sha256=sha(canonical(functions)),normalized_mask_candidate_sites=sites,
                full_callee_sha256=sha(canonical([f for f in document[1] if f["function-name"]!="frisk_execution"])),
                semantics="every physical instruction/opcode/register/predicate/control/callee field retained except bounded entry LOP3 third-operand masks; pure-address change is independently established at source, not a native DU proof")


def summarize(directory,locality,native_raw=None,*,nvdisasm="/usr/local/cuda/bin/nvdisasm",document_provider=None):
    root=Path(directory);provenance=dict(analyzer_sha256=SOURCE_SHA256,source_dependencies=SUPPORT_SOURCES.copy(),dependencies={})
    try:
        if sha(Path(__file__).read_bytes())!=SOURCE_SHA256:raise ValueError("intervention-source-changed")
        for name,expected in SUPPORT_SOURCES.items():
            if sha((Path(__file__).parent/name).read_bytes())!=expected:raise ValueError("support-source-changed:"+name)
        if locality.get("status")!="complete" or locality.get("cost_family")!=access_reuse.FAMILY:raise ValueError("checked-locality-domain-required")
        body={k:v for k,v in locality.items() if k!="summary_sha256"}
        if locality.get("summary_sha256")!=sha(canonical(body)) or locality["provenance"]["analyzer_sha256"]!=access_reuse.SOURCE_SHA256:raise ValueError("locality-proof-integrity")
        for name,expected in locality["provenance"]["dependencies"].items():
            if sha((root/name).read_bytes())!=expected:raise ValueError("locality-input-binding:"+name)
        required=("kernel.ptx","kernel.cubin","correctness.json","image-identity.json","native-analysis.json","execution_config.json")
        raw={n:(root/n).read_bytes() for n in required};provenance["dependencies"]={n:sha(v) for n,v in raw.items()}
        image=json.loads(raw["image-identity.json"]);correct=json.loads(raw["correctness.json"]);native_old=json.loads(raw["native-analysis.json"])
        if image.get("kernel.ptx")!=sha(raw["kernel.ptx"]) or image.get("kernel.cubin")!=sha(raw["kernel.cubin"]) or correct.get("cubin_sha256")!=sha(raw["kernel.cubin"]) or correct.get("image_identity_sha256")!=sha(raw["image-identity.json"]) or correct.get("no_kernel_timing") is not True:raise ValueError("accepted-compiled-image-binding")
        if native_old.get("identity",{}).get("cubin_sha256")!=sha(raw["kernel.cubin"]):raise ValueError("native-analysis-cubin-binding")
        for name,expected in locality["provenance"].get("source_dependencies",{}).items():
            if name not in access_reuse.SOURCE_FILES or sha((Path(__file__).parent/name).read_bytes())!=expected:raise ValueError("locality-support-source-binding")
        if native_raw is None:
            tool=Path(nvdisasm).resolve(strict=True)
            if document_provider is None:
                native_raw=subprocess.check_output([str(tool),"-json",str(root/"kernel.cubin")],stderr=subprocess.PIPE,timeout=120)
            else:
                tool_identity=document_provider.tool(tool)
                native_raw=document_provider.output(tool_identity,["-json",str(root/"kernel.cubin")],root/"kernel.cubin")
            provenance["nvdisasm"]={"path":str(tool),"sha256":sha(tool.read_bytes())}
        if not isinstance(native_raw,bytes) or native_old.get("native_json_sha256")!=sha(native_raw):raise ValueError("native-document-old-binding")
        provenance["native_json_sha256"]=sha(native_raw)
        native_document=json.loads(native_raw)
        source=(source_signature(raw["kernel.ptx"].decode()) if document_provider is None else
                document_provider.pure(source_signature, raw["kernel.ptx"].decode()))
        native=native_signature(native_document,document_provider=document_provider)
        if source["status"]!="available":raise ValueError(source["reasons"][0])
        if native["status"]!="available":raise ValueError("native-static-skeleton-unavailable")
        resources=correct["physical_resources"]
        resource_signature=dict(function_attributes=resources["function_attributes"],threads_per_block=resources["threads_per_block"],
                                dynamic_shared_bytes=resources["dynamic_shared_bytes"],device_limits=resources["device_limits"])
        key=sha(canonical(dict(source=source["nonaddress_sha256"],native=native["skeleton_sha256"],resources=resource_signature)))
        result=dict(contract=CONTRACT,status="available",controlled_group_sha256=key,source=source,native=native,resources=resource_signature,
                    locality_sha256=locality["summary_sha256"],provenance=provenance,
                    claims=dict(controlled_source_global_address_intervention=True,complete_native_static_work_and_resources=True,native_address_du_proven=False,dynamic_lane_masks_or_slowpath_counts_equal=False,native_cache_or_tlb_misses_proven=False))
        for name,expected in provenance["dependencies"].items():
            if sha((root/name).read_bytes())!=expected:raise ValueError("input-changed-during-intervention-proof:"+name)
        for name,expected in SUPPORT_SOURCES.items():
            if sha((Path(__file__).parent/name).read_bytes())!=expected:raise ValueError("support-source-changed-during-proof:"+name)
        result["summary_sha256"]=sha(canonical(result));return result
    except (OSError,subprocess.SubprocessError,KeyError,TypeError,ValueError,IndexError) as error:
        result=dict(contract=CONTRACT,status="unsupported",reasons=[str(error)],provenance=provenance);result["summary_sha256"]=sha(canonical(result));return result
