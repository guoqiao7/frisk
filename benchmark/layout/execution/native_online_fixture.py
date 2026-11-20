"""Construct a packed online-attention tile for the fixed two-warpgroup native path."""
from pathlib import Path
import math
import struct


def _copy_layout(rows, cols, threads=256):
    row_bits, col_bits = (n.bit_length()-1 for n in (rows, cols))
    thread_bits = threads.bit_length()-1
    regs = row_bits + col_bits - thread_bits
    order = list(range(regs+col_bits, regs+thread_bits)) + list(range(regs)) + list(range(regs, regs+col_bits))
    matrix = [[int(i==j) for j in range(regs+thread_bits)] for i in order]
    wg_name = ',"warp_group"' if threads>128 else ''
    wg_bits = ',1' if threads>128 else ''
    return (f'#frisk.distributed<map=#frisk.bit_linear<inputs=["register","lane","warp"{wg_name}],'
            f'input_bits=[{regs},5,2{wg_bits}],outputs=["dim0","dim1"],output_bits=[{row_bits},{col_bits}],'
            f'matrix=dense<{matrix}>:tensor<{regs+thread_bits}x{regs+thread_bits}xi1>>,topology=[{1<<regs},32,4,{threads//128},1],replication=1>')


def _storage(rows, cols, width, role):
    if role in ('Q','K'):
        base = f'((i mod 8)*128+(i floordiv 8)*1024+(j mod 64)*2+(j floordiv 64)*{rows*128})'
        swizzle = 128
    elif role=='V':
        base = f'((j mod 16)*2+(j floordiv 16)*256+(i mod 8)*32+(i floordiv 8)*{cols*16})'
        swizzle = 32
    else:
        base = f'(i*{cols*width}+j*{width})'
        swizzle = 0
    offset = base
    for bit in (16,32,64):
        if bit>=swizzle: break
        lo=f'(({base} floordiv {bit}) mod 2)'
        hi=f'(({base} floordiv {bit*8}) mod 2)'
        offset+=f'+((({lo}+{hi}) mod 2)-{lo})*{bit}'
    return (f'#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[{rows},{cols}],'
            f'outputs=["byte_offset","bit_offset"],output_extents=[{rows*cols*width},8],'
            f'map=affine_map<(i,j)->({offset},0)>>,memory_space=#frisk<memory_space {"Shared" if role else "Global"}>,'
            f'alignment={1024 if role else 128 if width==4 else 1},vector_granularity=1>')


def online_fixture(qtile, dim, rows, heads=8, kv_heads=8, causal=False, *, kv_rows=None, value_dim=None, execution_contract=None):
    from online_execution_contract import FA3_KINDS, validate
    if execution_contract is not None: validate(execution_contract)
    fa3 = execution_contract is not None and execution_contract['kind'] in FA3_KINDS
    kv_tile = 128 if fa3 else 64
    if fa3 and (qtile,dim,value_dim or dim,rows,kv_rows or rows,heads,kv_heads,causal)!=(128,128,128,8192,8192,16,16,False):
        raise ValueError('unsupported FA3 replay workload')
    kv_rows = rows if kv_rows is None else kv_rows
    if type(kv_rows) is not int or kv_rows<=0 or type(rows) is not int:
        raise ValueError('invalid packed sequence lengths')
    value_dim = dim if value_dim is None else value_dim
    if value_dim not in (32,64,128) or type(value_dim) is not int:
        raise ValueError('unsupported native online value dimension')
    kv_attr = f'kv_rows={kv_rows}:i64,' if kv_rows!=rows else ''
    if qtile not in (128,256,512) or dim not in (64,128,256) or rows<=0 or kv_heads<=0 or heads%kv_heads:
        raise ValueError('unsupported native online geometry')
    # Keep the SSA operations and copy policy common to the original native fixture.
    template=Path(__file__).with_name('native_online_q256.mlir').read_text()
    body=template[template.index('%zero ='):].splitlines()
    spec=[('Q',qtile,'Q'),('K0',kv_tile,'K'),('V0',kv_tile,'V'),('O',qtile,'')]
    columns={'Q':dim,'K0':dim,'V0':value_dim,'O':value_dim}
    types=[f'memref<{r}x{columns[v]}x{"f32" if v=="O" and not fa3 else "f16"},1>' for v,r,_ in spec]
    lines=[f'module attributes {{frisk.target="sm_90a", frisk.online_launch={{rows={rows}:i64,{kv_attr}heads={heads}:i64,kv_heads={kv_heads}:i64,causal={str(causal).lower()}}}}} {{',
           '"frisk.kernel"() <{sym_name="native_online",function_type=('+','.join(types)+')->()}> ({',
           '^bb0('+','.join(f'%{v}:{t}' for (v,_,_),t in zip(spec,types))+'):']
    for (v,r,_),t in zip(spec,types):
        w=4 if v=='O' and not fa3 else 2
        lines.append(f'frisk.storage_contract %{v} {{layout={_storage(r,columns[v],w,None)}}} : {t}')
    for (v,r,role),t in zip(spec[:3],types):
        lines.append(f'%{role.lower()}s = memref.alloc() {{alignment=1024:i64}} : '+t.replace(',1>',',3>'))
    lines+=['"frisk.parallel"() <{ranges=array<i64:1>,threads=256:i64}> ({','^bb0(%i:index):']
    for (v,r,role),t in zip(spec,types):
        lines.append(f'%g{v} = frisk.layout_view %{v} : {t} -> {t}')
        if role:
            shared=t.replace(',1>',',3>');sv=role.lower()+'s'
            lines.append(f'%s{sv} = frisk.layout_view %{sv} {{layout={_storage(r,columns[v],2,"K" if fa3 else role)}}} : {shared} -> {shared}')
    def copy(index):
        (v,r,role),t=spec[index],types[index]
        binding = f'frisk.execution_role="{role.lower()}",' if fa3 else 'frisk.execution_threads=256:i64,'
        width = execution_contract['groups'][execution_contract['roles'][role.lower()]]['threads'] if fa3 else 256
        return (f'frisk.copy %g{v}[], %s{role.lower()}s[] {{{binding}'
                f'frisk.execution_layout={_copy_layout(r,columns[v],width)},frisk.writer_policy="first_owner",'
                f'frisk.vector_bytes=1:i64,srcExtents=array<i64:{r},{columns[v]}>,dstExtents=array<i64:{r},{columns[v]}>}} : '
                + t+', '+t.replace(',1>',',3>'))
    lines.append(copy(0))
    logits=f'tensor<{qtile}x{kv_tile}xf32>'; out=f'tensor<{qtile}x{value_dim}xf32>'
    bits=struct.unpack('>I',struct.pack('>f',1/math.sqrt(dim)))[0]
    for line in body:
        if line.startswith('frisk.copy'):
            copied = copy(1 if '%gK0' in line else 2)
            lines.append(copied)
            continue
        line=line.replace('tensor<256xf32>',f'tensor<{qtile}xf32>').replace('m=256:i64',f'm={qtile}:i64')
        line=line.replace('gemm_warp_policy Square','gemm_warp_policy FullRow').replace('frisk.execution_threads=256:i64,policy=', 'frisk.execution_threads=256:i64,frisk.row_partition="contiguous",policy=')
        is_out=line.startswith(('%mfinal,','%ba1 =','%acc1 =','%out1 =','scf.yield','%denom3 =','%out =','frisk.tile_store'))
        line=line.replace('tensor<256x64xf32>',out if is_out else logits).replace('tensor<256x64xf16>',f'tensor<{qtile}x{kv_tile}xf16>')
        width=value_dim if is_out else dim
        line=line.replace('memref<256x64x',f'memref<{qtile}x{width}x').replace('memref<64x64x',f'memref<{kv_tile}x{width}x')
        if line.startswith('%zero ='):
            lines.append(f'%zeroO = arith.constant dense<0.0> : {out}')
        if line.startswith('%empty ='):
            lines.append(f'%emptyO = tensor.empty() : {out}')
        if is_out:line=line.replace('%empty :','%emptyO :').replace('%carryO=%zero)','%carryO=%zeroO)')
        line=line.replace('dense<0.125>',f'dense<0x{bits:08X}>').replace('%bound = arith.constant 16 :',f'%bound = arith.constant {(kv_rows+kv_tile-1)//kv_tile} :')
        if line.startswith('%s1 ='):line=line.replace('k=64:i64',f'k={dim}:i64')
        if line.startswith('%out1 ='):line=line.replace('n=64:i64',f'n={value_dim}:i64')
        if fa3:
            if line.startswith('%s1 ='):
                line=line.replace('n=64:i64','n=128:i64')
            if line.startswith('%out1 ='):
                line=line.replace('k=64:i64','k=128:i64')
            if 'frisk.execution_threads=256:i64' in line:
                role='qk' if line.startswith('%s1 =') else 'pv' if line.startswith('%out1 =') else 'reduce'
                line=line.replace('frisk.execution_threads=256:i64',f'frisk.execution_role="{role}"')
            if line.startswith('frisk.tile_store'):
                half_out=out.replace('xf32','xf16')
                lines.append(f'%out_half = arith.truncf %out : {out} to {half_out}')
                line=f'frisk.tile_store %out_half, %gO : {half_out}, {types[3]}'
        lines.append(line)
    ir='\n'.join(lines)+'\n'
    if execution_contract is None: return ir
    from online_execution_contract import bind_module
    return bind_module(ir,execution_contract)
