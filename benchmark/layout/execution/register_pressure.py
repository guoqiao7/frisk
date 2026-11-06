"""Typed PTX virtual-register pressure with CFG liveness and syntactic DCE.

This analysis precedes ptxas scheduling/expansion/allocation. Its output is a
virtual-pressure feature, never an allocated-register count or a physical upper
bound. It does not inspect cubins, driver attributes, or training targets.
"""
import hashlib
import re
from collections import Counter

SCHEMA_VERSION = 1
REGISTER_MODEL = "typed-ptx-cfg-pressure-v1"
_REG = re.compile(r"(?<![A-Za-z0-9_$%.])(%?[A-Za-z_$][A-Za-z0-9_$]*)(?![A-Za-z0-9_$])")
_TYPES = {"pred": 0, "b8": 1, "b16": 1, "b32": 1, "b64": 2,
          "u8": 1, "u16": 1, "u32": 1, "u64": 2,
          "s8": 1, "s16": 1, "s32": 1, "s64": 2,
          "f16": 1, "f32": 1, "f64": 2}
_PURE = {"mov", "cvt", "cvta", "add", "addc", "sub", "subc", "mul", "mad", "fma", "div",
         "rem", "abs", "neg", "min", "max", "and", "or", "xor", "not", "shl", "shr",
         "bfe", "bfi", "prmt", "lop3", "setp", "set", "selp", "slct", "ex2", "lg2",
         "rcp", "sqrt", "rsqrt", "sin", "cos", "popc", "clz", "brev", "copysign", "testp"}
_NO_DEST = {"st", "bar", "fence", "membar", "cp", "prefetch", "prefetchu", "red"}


class _Unsupported(ValueError):
    pass


def _split_operands(text):
    result=[]; start=0; stack=[]
    for i,ch in enumerate(text):
        if ch in "[{(": stack.append(ch)
        elif ch in "]})":
            if not stack or "[{(".index(stack.pop()) != "]})".index(ch):
                raise _Unsupported("unbalanced instruction operand delimiters")
        elif ch=="," and not stack:
            result.append(text[start:i].strip()); start=i+1
    if stack: raise _Unsupported("unbalanced instruction operand delimiters")
    if text[start:].strip(): result.append(text[start:].strip())
    return result


def _function_body(text,entry):
    pattern=r"(?:\.visible\s+)?\.entry\s+"+re.escape(entry)+r"\s*\("
    matches=list(re.finditer(pattern,text))
    if len(matches)!=1: raise _Unsupported("expected exactly one named PTX entry")
    start=text.find("{",matches[0].end())
    if start<0: raise _Unsupported("missing entry body")
    depth=1
    for end in range(start+1,len(text)):
        depth += (text[end]=="{")-(text[end]=="}")
        if depth==0: return text[start+1:end]
    raise _Unsupported("unterminated entry body")


def _statements(body):
    # Directives/labels are line delimited. Instructions and register declarations
    # may span lines; standalone braces delimit scopes, not operands.
    pending=""; out=[]
    for line in body.splitlines():
        line=line.strip()
        if not line or line in ("{","}"): continue
        if not pending and line.endswith(":"):
            out.append(("label",line[:-1])); continue
        if not pending and line.startswith((".loc ",".loc\t",".pragma ",".pragma\t")):
            continue
        pending += " "+line
        while ";" in pending:
            item,pending=pending.split(";",1);item=item.strip()
            if item:out.append(("statement",item))
            pending=pending.strip()
            if pending in ("{","}"):pending=""
    if pending.strip(): raise _Unsupported("unsupported unterminated PTX statement")
    return out


def _parse(text,entry,max_registers,max_instructions):
    text=re.sub(r"/\*.*?\*/","",text,flags=re.S)
    text=re.sub(r"//[^\n]*","",text)
    body=_function_body(text,entry);items=_statements(body)
    registers={};types={}; instructions=[];labels={};pending_labels=[]
    for kind,value in items:
        if kind=="label":pending_labels.append(value);continue
        if value.startswith(".reg"):
            m=re.fullmatch(r"\.reg\s+\.([a-z0-9]+)\s+(.+)",value)
            if not m or m[1] not in _TYPES:raise _Unsupported("unsupported PTX register declaration")
            ty=m[1]
            for name in _split_operands(m[2]):
                vector=re.fullmatch(r"(%?[A-Za-z_$][A-Za-z0-9_$]*)<(\d+)>",name)
                count=int(vector[2]) if vector else 1
                if count<1 or count>max_registers or len(registers)+count>max_registers:
                    raise _Unsupported("register analysis budget exceeded")
                names=[vector[1]+str(i) for i in range(count)] if vector else [name]
                for name in names:
                    if not re.fullmatch(r"%?[A-Za-z_$][A-Za-z0-9_$]*",name) or name in registers:
                        raise _Unsupported("ambiguous/scoped duplicate register declaration")
                    registers[name]=len(registers);types[name]=ty
            continue
        if value.startswith((".shared ",".shared\t",".local ",".local\t")):
            continue
        if value.startswith("."):raise _Unsupported("unsupported body directive")
        for label in pending_labels:
            if label in labels:raise _Unsupported("duplicate control-flow label")
            labels[label]=len(instructions)
        pending_labels=[]
        pred=None
        if value.startswith("@"):
            fields=value.split(None,1)
            if len(fields)!=2:raise _Unsupported("malformed instruction predicate")
            pred=fields[0][1:].lstrip("!");value=fields[1]
        fields=value.split(None,1);op=fields[0];operands=_split_operands(fields[1]) if len(fields)==2 else []
        instructions.append(dict(op=op,operands=operands,predicate=pred))
        if len(instructions)>max_instructions:raise _Unsupported("instruction analysis budget exceeded")
    for label in pending_labels:labels[label]=len(instructions)
    if not instructions:raise _Unsupported("entry contains no supported instructions")
    symbols=set(labels)
    for match in re.finditer(r"\.(?:param|shared|local)\s+([^;\n,()]+)",text):
        name=re.search(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:\[[^]]+\])?\s*$",match[1])
        if name:symbols.add(name[1])
    return registers,types,instructions,labels,symbols


def analyze_ptx_pressure(ptx_text,entry="frisk_execution",max_registers=65536,max_instructions=262144,
                         max_dependency_edges=1048576,max_analysis_work=8388608):
    """Return typed pre-RA pressure or explicit unavailable for unsupported PTX.

    Predicated writes do not kill previous values. Direct CFG branches and
    loop backedges are solved to a fixed point. 64-bit virtual values consume
    two register32 words; predicates and special registers are separate.
    """
    if not isinstance(ptx_text,str):raise TypeError("PTX text must be a string")
    for value in (max_registers,max_instructions,max_dependency_edges,max_analysis_work):
        if type(value) is not int or value<1:raise ValueError("positive analysis limits required")
    work=Counter()
    def spend(category,amount=1):
        if amount>max_analysis_work-work["total"]:
            raise _Unsupported("total analysis work budget exceeded during "+category)
        work["total"]+=amount;work[category]+=amount
    base=dict(schema_version=SCHEMA_VERSION,model=REGISTER_MODEL,
              ptx_sha256=hashlib.sha256(ptx_text.encode()).hexdigest(),entry=entry,
              meaning="typed virtual pressure before ptxas scheduling/expansion/allocation; not physical registers or an upper bound",
              physical_registers_per_thread=None,physical_status="requires_compilation",
              analysis_limits=dict(max_registers=max_registers,max_instructions=max_instructions,
                  max_dependency_edges=max_dependency_edges,max_analysis_work=max_analysis_work),analysis_work=work)
    try:
        regs,types,ins,labels,symbols=_parse(ptx_text,entry,max_registers,max_instructions)
        def tokens(operand):
            return [m.group(1) for m in _REG.finditer(operand) if m.group(1) in regs]
        def mask(names):
            out=0
            for name in names:out |= 1<<regs[name]
            return out
        gpr=mask(n for n in regs if types[n]!="pred")
        wide=mask(n for n in regs if _TYPES[types[n]]==2)
        predmask=mask(n for n in regs if types[n]=="pred")
        addresses=set();opcounts=Counter()
        for i,item in enumerate(ins):
            spend("instruction_decode_visits")
            op=item["op"];root=op.split(".")[0];operands=item["operands"];predicate=item["predicate"]
            opcounts[op]+=1
            if root in ("addc","subc") or "cc" in op.split("."):
                raise _Unsupported("implicit carry condition state is unsupported")
            if root=="bar" and not op.startswith("bar.sync"):
                raise _Unsupported("only explicit CTA barrier sync is supported")
            if predicate and predicate not in regs:raise _Unsupported("undeclared instruction predicate")
            for operand in operands:
                for token in _REG.finditer(operand):
                    name=token.group(1)
                    special=(name in ("%tid","%ctaid","%ntid","%nctaid") and operand[token.end():].startswith((".x",".y",".z"))) or name in ("%laneid","%warpid","%nwarpid","%smid","%clock","%clock64","%gridid")
                    if name not in regs and name not in symbols and name!="_" and not special:
                        raise _Unsupported("undeclared register or unsupported symbol "+name)
                for memory in re.findall(r"\[([^]]+)\]",operand):addresses.update(tokens(memory))
            names=[tokens(x) for x in operands];defs=[];uses=[];pure=False;target=None
            if root in _PURE or root=="ld":
                if not operands:raise _Unsupported("missing destination operand")
                defs=names[0];uses=sum(names[1:],[])
                pure=root in _PURE or (root=="ld" and not any(x in op.split(".") for x in ("volatile","acquire","relaxed")))
            elif root=="shfl":
                if not operands:raise _Unsupported("missing shuffle destination")
                defs=names[0];uses=sum(names[1:],[])
            elif root in ("mma","wgmma"):
                if op.startswith("wgmma.mma") or root=="mma":
                    if not operands:raise _Unsupported("missing MMA destination tuple")
                    defs=names[0];uses=sum(names[1:],[])
                    if root=="wgmma":uses+=defs
                elif op.startswith(("wgmma.fence.","wgmma.commit_group.","wgmma.wait_group.")):
                    uses=sum(names,[])
                else:raise _Unsupported("unknown warpgroup instruction")
            elif root=="mbarrier":
                if op.startswith(("mbarrier.arrive.","mbarrier.try_wait.","mbarrier.test_wait.")):
                    defs=names[0];uses=sum(names[1:],[])
                elif op.startswith(("mbarrier.init.","mbarrier.inval.")):uses=sum(names,[])
                else:raise _Unsupported("unknown mbarrier instruction")
            elif root in _NO_DEST:uses=sum(names,[])
            elif root=="bra":
                if len(operands)!=1 or operands[0] not in labels:raise _Unsupported("indirect or unknown branch target")
                target=labels[operands[0]]
            elif root not in ("ret","exit","nop"):raise _Unsupported("unsupported PTX opcode "+op)
            elif operands:raise _Unsupported("unsupported control instruction operands")
            if root not in ("bra","ret","exit","nop") and not defs and root in _PURE:
                raise _Unsupported("undeclared instruction destination")
            if predicate:uses.append(predicate)
            item.update(defs=mask(defs),uses=mask(uses),pure=pure,target=target,root=root)
        # A basic block starts at every branch target or after a terminator.
        starts={0}
        for i,item in enumerate(ins):
            if item["target"] is not None:
                if item["target"]==len(ins):raise _Unsupported("branch target outside instruction body")
                starts.add(item["target"])
            if item["root"] in ("bra","ret","exit") and i+1<len(ins):starts.add(i+1)
        starts=sorted(starts)
        if len(starts)>8192:raise _Unsupported("CFG analysis budget exceeded")
        blocks=[(s,starts[j+1] if j+1<len(starts) else len(ins)) for j,s in enumerate(starts)]
        owner={i:b for b,(lo,hi) in enumerate(blocks) for i in range(lo,hi)}
        successors=[]
        for b,(lo,hi) in enumerate(blocks):
            last=ins[hi-1];succ=[]
            if last["target"] is not None:succ.append(owner[last["target"]])
            if b+1<len(blocks) and (last["root"] not in ("bra","ret","exit") or last["predicate"]):succ.append(b+1)
            successors.append(succ)
        addr=mask(addresses)
        # Include pointer bases and the integer coordinate/address calculations
        # feeding actual memory operands. This is an explanatory dependency
        # category, not a separate physical register file.
        def bit_ids(bits):
            while bits:
                one=bits&-bits;yield one.bit_length()-1;bits^=one
        # One reverse dependency graph, rather than repeatedly scanning a long
        # forward-ordered instruction chain once for every newly found base.
        dependencies={};dependency_pairs=0
        for item in ins:
            spend("dependency_instruction_visits")
            if not item["uses"] or not item["defs"]:continue
            defined=list(bit_ids(item["defs"]));used=list(bit_ids(item["uses"]))
            pairs=len(defined)*len(used)
            if pairs>max_dependency_edges-dependency_pairs:
                raise _Unsupported("address dependency edge budget exceeded")
            dependency_pairs+=pairs
            spend("dependency_build_pairs",pairs)
            for reg in defined:dependencies.setdefault(reg,set()).update(used)
        work["dependency_edges_upper_bound"]=dependency_pairs
        queue=list(bit_ids(addr));visited=set(queue)
        while queue:
            reg=queue.pop();spend("address_dependency_node_visits")
            for source in dependencies.get(reg,()):
                spend("address_dependency_edge_visits")
                if source not in visited:visited.add(source);queue.append(source)
        addr=0
        for reg in visited:addr|=1<<reg
        # Bitset work is bounded as well as visit count. A large register ID
        # does not turn one Python big-int operation into uncharged work.
        bitset_words=max(1,(len(regs)+63)//64)
        def transfer(lo,hi,live,collect=False):
            peak=(0,0,0,0);at=None;removed=0;kept=0
            for i in range(hi-1,lo-1,-1):
                spend("liveness_instruction_word_visits",bitset_words)
                work["liveness_instruction_visits"]+=1
                item=ins[i];defs=item["defs"]
                if item["pure"] and defs and not defs&live:
                    removed+=1;continue
                kept+=1
                after=live
                kill=defs if not item["predicate"] else 0
                live=(live&~kill)|item["uses"]
                # Simultaneous read/write at an instruction is conservatively
                # represented by the union, without requiring non-coalescing.
                occupied=live|after|defs
                values=((occupied&gpr).bit_count()+(occupied&wide).bit_count(),
                        (occupied&predmask).bit_count(),(occupied&wide).bit_count()*2,
                        (occupied&addr&gpr).bit_count()+(occupied&addr&wide).bit_count())
                if values[0]>peak[0]:at=i
                peak=tuple(max(x,y) for x,y in zip(peak,values))
            return live,peak,at,removed,kept
        reachable={0};pending=[0]
        while pending:
            for successor in successors[pending.pop()]:
                if successor not in reachable:reachable.add(successor);pending.append(successor)
        incoming=[0]*len(blocks);outgoing=[0]*len(blocks);iterations=0
        while True:
            changed=False;iterations+=1
            for b in range(len(blocks)-1,-1,-1):
                if b not in reachable:continue
                spend("liveness_block_visits")
                value=0
                for successor in successors[b]:
                    spend("liveness_edge_word_visits",bitset_words)
                    value|=incoming[successor]
                candidate=transfer(*blocks[b],value)[0]
                if candidate!=incoming[b] or value!=outgoing[b]:changed=True
                incoming[b]=candidate;outgoing[b]=value
            if not changed:break
            if iterations>min(65536,max(64,4*len(blocks)+4)):
                raise _Unsupported("CFG liveness did not reach fixed point within budget")
        peaks=[];dce=0;kept=0
        for b,block in enumerate(blocks):
            if b not in reachable:continue
            _,peak,at,dead,n=transfer(*block,outgoing[b],True);peaks.append((peak,at));dce+=dead;kept+=n
        peak,at=max(peaks,key=lambda x:x[0][0])
        features=dict(virtual_live_register32_peak=peak[0],virtual_live_predicate_peak=max(p[0][1] for p in peaks),
                      virtual_live_u64_register32_peak=max(p[0][2] for p in peaks),
                      virtual_live_address_register32_peak=max(p[0][3] for p in peaks),
                      virtual_declared_register32=sum(_TYPES[t] for t in types.values()),
                      virtual_declared_predicates=sum(t=="pred" for t in types.values()),
                      ptx_instruction_count=len(ins),ptx_live_instruction_count=kept,syntactic_dce_instruction_count=dce,
                      basic_blocks=len(blocks),reachable_basic_blocks=len(reachable),cfg_edges=sum(len(successors[b]) for b in reachable),
                      div_rn_f32_instructions=opcounts["div.rn.f32"],
                      global_byte_load_instructions=sum(n for op,n in opcounts.items() if op.startswith("ld.global.") and op.endswith(".u8")),
                      shared_load_instructions=sum(n for op,n in opcounts.items() if op.startswith("ld.shared.")),
                      shared_store_instructions=sum(n for op,n in opcounts.items() if op.startswith("st.shared.")),
                      barrier_instructions=sum(n for op,n in opcounts.items() if op.startswith("bar.sync")),
                      mma_instructions=sum(n for op,n in opcounts.items() if op.startswith(("mma.","wgmma.mma"))),
                      shuffle_instructions=sum(n for op,n in opcounts.items() if op.startswith("shfl.")),
                      exp2_instructions=sum(n for op,n in opcounts.items() if op.startswith("ex2.")))
        return dict(base,status="available",features=features,analysis_iterations=iterations,
                    peak_instruction_index=at,peak_opcode=ins[at]["op"] if at is not None else None,
                    liveness_boundary="union of read/write virtual values after syntactic dead pure-code removal",
                    coverage="direct-branch CFG including loops; predicated defs preserve old values; unallocated 16/32bit carriers use one word and 64bit use two; address feature follows backward syntactic dependencies")
    except _Unsupported as error:
        return dict(base,status="unavailable",features=None,reason=str(error))
