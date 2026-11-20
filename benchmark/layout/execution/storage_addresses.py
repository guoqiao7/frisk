"""Exact bounded codec for contiguous coordinate-identity Global f32 storage.

The wire descriptor remains execution identity. Internal immutable sequences are
structural address proofs, not a relaxed general Sequence input contract.
"""
from collections.abc import Sequence,Mapping
import copy,math,re
CONTRACT='contiguous-identity-addresses-v1';FIELDS={'contract','shape','element_bits','base_bit','strides_bits'};MAX_VOLUME=65536
class RowMajorAddresses(Sequence):
 __slots__=('_shape','_strides','_count')
 def __init__(self,description):
  if type(description) is not dict or set(description)!=FIELDS:raise ValueError('exact compact-address schema required')
  if type(description['contract']) is not str or description['contract']!=CONTRACT:raise ValueError('unknown compact-address contract')
  shape=description['shape'];strides=description['strides_bits']
  if type(shape) is not list or len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape):raise ValueError('first version requires positive static rank2 shape')
  count=math.prod(shape)
  if count>MAX_VOLUME:raise ValueError('compact-address volume outside supported domain')
  if type(description['element_bits']) is not int or description['element_bits']!=32 or type(description['base_bit']) is not int or description['base_bit']!=0:raise ValueError('first version requires f32/base0')
  if type(strides) is not list or any(type(v) is not int for v in strides) or strides!=[32*shape[1],32]:raise ValueError('exact row-major strides required')
  object.__setattr__(self,'_shape',tuple(shape));object.__setattr__(self,'_strides',tuple(strides));object.__setattr__(self,'_count',count)
 def __setattr__(self,key,value):raise AttributeError('immutable compact-address snapshot')
 def __len__(self):return self._count
 def __getitem__(self,index):
  if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
  if type(index) is not int:raise TypeError('integer address index required')
  if index<0:index+=len(self)
  if not 0<=index<len(self):raise IndexError('address index outside compact domain')
  row,col=divmod(index,self._shape[1]);return dict(begin_bit=32*index,end_bit=32*(index+1),root=[row,col],view=[row,col])
 def __deepcopy__(self,memo):memo[id(self)]=self;return self
 def description(self):return dict(contract=CONTRACT,shape=list(self._shape),element_bits=32,base_bit=0,strides_bits=list(self._strides))
 def complete_state(self):return (CONTRACT,self._shape,32,0,self._strides,self._count)
 def extent(self):return (4*self._count,32*self._count)
 def byte_offsets(self):return RowMajorByteOffsets(self._count)
class RowMajorByteOffsets(Mapping):
 __slots__=('_count',)
 def __init__(self,count):
  if type(count) is not int or not 1<=count<=MAX_VOLUME:raise ValueError('validated offset count required')
  object.__setattr__(self,'_count',count)
 def __setattr__(self,k,v):raise AttributeError('immutable offset map')
 def __len__(self):return self._count
 def __iter__(self):return iter(range(self._count))
 def __getitem__(self,index):
  if type(index) is not int or not 0<=index<self._count:raise KeyError(index)
  return 4*index

_TYPE = re.compile(r'memref<(\d+)x(\d+)xf32, (?:(strided<\[(\d+), (\d+)\](?:, offset: (0))?>), )?(1|#frisk<memory_space Global>)>')
_LAYOUT = re.compile(r'#frisk\.storage<map = #frisk\.affine_layout<inputs = (\[[^\]]*\]), input_extents = (\[[^\]]*\]), outputs = (\[[^\]]*\]), output_extents = (\[[^\]]*\]), map = affine_map<\(d0, d1\) -> \((.*), 0\)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>')

def _linear_expression(text):
    # A bounded shunting-yard parser: no eval, symbols, division or arbitrary AST.
    # Its positive-literal multiplication grammar mirrors the C++ exact matcher.
    if len(text)>131072:raise ValueError('oversized compact affine expression')
    operators=[];values=[];position=0;expect_operand=True;nodes=0
    def apply():
        op=operators.pop()
        if len(values)<2:raise ValueError('invalid compact affine expression')
        right,rliteral=values.pop();left,lliteral=values.pop()
        if op=='+':result=tuple(x+y for x,y in zip(left,right))
        elif lliteral and left[2]>0:result=tuple(left[2]*x for x in right)
        elif rliteral and right[2]>0:result=tuple(right[2]*x for x in left)
        else:raise ValueError('compact multiplication requires a positive literal')
        if any(not 0<=x<(1<<63) for x in result):raise ValueError('affine coefficient overflow')
        values.append((result,False))
    while position<len(text):
        match=re.match(r'\s*(d[01]|[0-9]+|[()+*])',text[position:])
        if match is None:raise ValueError('unsupported compact affine expression')
        token=match.group(1);position+=match.end()
        if token=='(':
            if not expect_operand:raise ValueError('missing compact affine operator')
            operators.append(token)
        elif token==')':
            if expect_operand:raise ValueError('empty compact affine operand')
            while operators and operators[-1]!='(':apply()
            if not operators:raise ValueError('unbalanced compact affine expression')
            operators.pop()
        elif token in ('+','*'):
            if expect_operand:raise ValueError('missing compact affine operand')
            while operators and operators[-1]!='(' and (operators[-1]=='*' or token=='+'):apply()
            operators.append(token);expect_operand=True;nodes+=1
        else:
            if not expect_operand:raise ValueError('missing compact affine operator')
            if token in ('d0','d1'):values.append(((1,0,0) if token=='d0' else (0,1,0),False))
            else:
                number=int(token)
                if number>=(1<<63):raise ValueError('affine integer overflow')
                values.append(((0,0,number),True))
            expect_operand=False;nodes+=1
        if nodes>4095:raise ValueError('compact affine node budget exceeded')
    if expect_operand:raise ValueError('missing compact affine operand')
    while operators:
        if operators[-1]=='(':raise ValueError('unbalanced compact affine expression')
        apply()
    if len(values)!=1:raise ValueError('invalid compact affine expression')
    return values[0][0]

def _shape(value):
    text=value.get('type');match=_TYPE.fullmatch(text) if type(text) is str else None
    if match is None:raise ValueError('compact storage requires rank2 Global f32 memref')
    shape=tuple(map(int,match.groups()[:2]))
    if match.group(3) is not None and (int(match.group(4)),int(match.group(5)))!=(shape[1],1):
        raise ValueError('compact memref descriptor must have zero offset and row-major strides')
    return shape

def _metadata(value,table):
    import json
    storage=value['storage'];shape=_shape(value)
    if shape!=table._shape or type(storage.get('alignment')) is not int or storage['alignment']!=1:
        raise ValueError('compact storage shape/alignment mismatch')
    text=storage.get('layout');match=_LAYOUT.fullmatch(text) if type(text) is str else None
    if match is None:raise ValueError('unsupported compact storage metadata')
    try:inputs,extents,outputs,out_extents=map(json.loads,match.groups()[:4])
    except (TypeError,ValueError) as error:raise ValueError('invalid compact layout metadata') from error
    if (type(inputs) is not list or len(inputs)!=2 or any(type(x) is not str or not x for x in inputs) or len(set(inputs))!=2 or
        type(extents) is not list or any(type(x) is not int for x in extents) or tuple(extents)!=shape or
        outputs!=['byte_offset','bit_offset'] or out_extents!=[4*len(table),8] or
        any(type(x) is not int for x in out_extents) or _linear_expression(match.group(5))!=(4*shape[1],4,0)):
        raise ValueError('compact storage metadata is not the declared row-major map')

def needs_storage_normalization(values):
    return type(values) is list and any(type(v) is dict and type(v.get('storage')) is dict and
        ('addresses_compact' in v['storage'] or type(v['storage'].get('addresses')) is RowMajorAddresses) for v in values)

def _dense_equals(table,addresses):
    if type(addresses) is not list or len(addresses)!=len(table):return False
    for i,address in enumerate(addresses):
        if (type(address) is not dict or set(address)!={'begin_bit','end_bit','root','view'} or
            any(type(address[k]) is not int for k in ('begin_bit','end_bit')) or
            any(type(address[k]) is not list or len(address[k])!=2 or any(type(x) is not int for x in address[k]) for k in ('root','view')) or
            address!=table[i]):return False
    return True

def normalize_storage_values(values, *, schema_version, materialize=False):
    """Strictly snapshot compact descriptors; ordinary dense values pass through."""
    if not needs_storage_normalization(values):return values
    by_id={};tables={};raw=set()
    for value in values:
        if type(value) is not dict or type(value.get('id')) is not int or value['id']<0 or value['id'] in by_id:
            raise ValueError('unique exact execution identities required')
        by_id[value['id']]=value
        storage=value.get('storage')
        if type(storage) is not dict:continue
        if 'addresses_compact' in storage:
            if type(schema_version) is not int or schema_version!=2:raise ValueError('compact storage wire requires schema2')
            if 'addresses' in storage:raise ValueError('addresses and addresses_compact are mutually exclusive')
            tables[value['id']]=RowMajorAddresses(storage['addresses_compact']);raw.add(value['id'])
        elif type(storage.get('addresses')) is RowMajorAddresses:
            table=storage['addresses']
            if type(table._shape) is not tuple or type(table._strides) is not tuple or type(table._count) is not int or any(type(x) is not int for x in table._shape+table._strides):raise ValueError('invalid compact internal state types')
            checked=RowMajorAddresses(table.description())
            if table.complete_state()!=checked.complete_state():raise ValueError('invalid compact internal state')
            tables[value['id']]=table
    for identity,table in tables.items():
        value=by_id[identity];storage=value['storage'];_metadata(value,table);root=storage.get('root')
        if type(root) is not int or root not in by_id or type(by_id[root].get('storage')) is not dict:
            raise ValueError('compact storage root missing')
        rv=by_id[root];rs=rv['storage']
        if type(rs.get('root')) is not int or rs['root']!=root or _shape(rv)!=table._shape or rv.get('type')!=value.get('type'):
            raise ValueError('compact storage canonical root/type mismatch')
        _metadata(rv,table)
        if root in tables:
            if tables[root].complete_state()!=table.complete_state():raise ValueError('compact root geometry mismatch')
        elif not _dense_equals(table,rs.get('addresses')):raise ValueError('compact root dense address proof mismatch')
    if not raw and not materialize:return values
    result=[]
    for value in values:
        identity=value['id']
        if identity not in tables:result.append(value);continue
        out=dict(value);storage=copy.deepcopy(value['storage']);storage.pop('addresses_compact',None)
        storage['addresses']=list(tables[identity]) if materialize else tables[identity]
        out['storage']=storage;result.append(out)
    return result

def global_root_spans(data,roots):
    """Validate the packet and return exact Global byte extents for CTA lifting."""
    from execution_support import normalize_packet
    normalized=normalize_packet(data);values={v['id']:v for v in normalized['values']};spans=[]
    for root in roots:
        storage=values[root].get('storage',{})
        if storage.get('root')!=root or not storage.get('addresses'):
            raise ValueError('Global root storage declaration is incomplete')
        addresses=storage['addresses']
        if type(addresses) is RowMajorAddresses:spans.append(addresses.extent()[0]);continue
        if any(type(a.get('begin_bit')) is not int or type(a.get('end_bit')) is not int or a['begin_bit']<0 or a['end_bit']<=a['begin_bit'] or a['begin_bit']%8 or a['end_bit']%8 for a in addresses):
            raise ValueError('Global tile footprint must be a nonnegative byte range')
        spans.append(max(a['end_bit'] for a in addresses)//8)
    return spans
