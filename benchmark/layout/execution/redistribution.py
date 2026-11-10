"""Check the exported C++ redistribution proof against complete finite maps.
Conversion-internal elected writers are distinct from external first_owner ABI.
"""
import math


def shuffle_source_support(plan, register):
 """Exact elected source-register support over all four warps.

 For an affine GF(2) owner map the support is a coset of the image of the
 seven thread columns. Reconstruct that image and check it against the full
 finite owner domain; an arbitrary validated owner table uses its exact set.
 No source/lane election, plan field or payload representation is changed.
 """
 if plan.get('mode')!='Shuffle':raise ValueError('shuffle support requires Shuffle')
 dr=plan['destination_registers'];nr=plan['source_registers']
 if type(register) is not int or not 0<=register<dr or len(plan['destinations'])!=128*dr:
  raise ValueError('shuffle support destination domain')
 table=[plan['destinations'][t*dr+register]['source']['reg'] for t in range(128)]
 if any(type(x) is not int or not 0<=x<nr for x in table):raise ValueError('shuffle support source domain')
 actual=set(table);base=table[0];basis={}
 for bit in range(7):
  column=table[1<<bit]^base
  while column:
   pivot=column.bit_length()-1
   if pivot not in basis:basis[pivot]=column;break
   column^=basis[pivot]
 image=[0]
 for column in basis.values():image += [x^column for x in image]
 coset={base^x for x in image}
 return tuple(sorted(coset if coset==actual else actual))

def affine_warp_sections(plan):
 """Replace a piecewise local-first owner by the canonical same-warp section.

 Selecting the least physical preimage is linear on the image of a linear
 layout. Verify both affine holder tables over the finite destination domain.
 Replicas carry the same immutable SSA payload.
 """
 from access_plan import linear
 nr,dr=plan['source_registers'],plan['destination_registers']
 source=plan['source_slots']
 first={}
 for index,slot in enumerate(source):
  first.setdefault((index//(32*nr),slot),index%(32*nr))
 sections=[]
 for register in range(dr):
  holders=[first[(thread//32,plan['destinations'][thread*dr+register]['slot'])]
           for thread in range(128)]
  lanes,regs=zip(*(divmod(holder,nr) for holder in holders))
  linear(lanes)
  linear(regs)
  sections.append((lanes,regs))
 return sections


def _network_declaration(plan, require):
 import json
 from shuffle_network import plan_shuffle_network,network_resource_proxy
 objective=plan.get('shuffle_network_objective','min-shfl')
 require(type(objective) is str and objective in ('min-shfl','min-work'),'unknown shuffle network objective')
 if 'shuffle_network' not in plan and 'shuffle_network_resource_proxy' not in plan:return
 expected=plan_shuffle_network(plan,objective=objective)
 def same(a,b):
  try:return a==b and json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False)
  except (ValueError,TypeError):return False
 if 'shuffle_network' in plan:require(same(plan['shuffle_network'],expected),'shuffle network declaration differs from full owner proof')
 if 'shuffle_network_resource_proxy' in plan:require(same(plan['shuffle_network_resource_proxy'],network_resource_proxy(expected)),'shuffle network resource proxy differs from emitted definitions')


def validate(operation, values):
 def require(ok,why):
  if not ok:raise ValueError('redistribution: '+why)
 try:
  p=operation['redistribution'];a,b=[values[i] for i in (operation['operands'][0],operation['results'][0])];s=a['distributed'];d=b['distributed']
  if 'symbolic' in s or 'symbolic' in d:
   from execution_support import normalize_distribution
   s=normalize_distribution(s);d=normalize_distribution(d)
  require('xf32' in a['type'] and 'xf32' in b['type'],'f32 payload required')
  require(s['shape']==d['shape'] and s['topology'][1:]==d['topology'][1:]==[32,4,1,1],'unsupported topology/shape')
  nr,dr=s['topology'][0],d['topology'][0];volume=math.prod(s['shape'])
  if 'source_slots_kind' in p:
   require(p['source_slots_kind']=='source-distributed-symbolic-v1' and 'source_slots' not in p,'compact source kind')
   require(p['mode']=='Register' and p['scratch_bytes']==0,'compact source Register-only route')
   require(0<volume<=65536 and 0<nr<=65536 and 0<dr and 128*dr<=65536,'compact source/destination domain')
   require((p['threads'],p['warps'],p['source_registers'],p['destination_registers'],p['volume'])==(128,4,nr,dr,volume),'header mismatch')
   require(p['payload']==dict(bits=32,words=[dict(bit_offset=0,live_bits=32)]),'unsupported payload')
   require(len(s['logical'])==128*nr and len(d['logical'])==128*dr,'map size')
   from execution_support import LogicalTable
   mapping=s['logical'].mapping if isinstance(s['logical'],LogicalTable) else None
   def compact_index(owner):
    require(set(owner)=={'reg','lane','warp','warp_group'} and all(type(x)==int for x in owner.values()),'owner fields')
    require(0<=owner['reg']<nr and 0<=owner['lane']<32 and 0<=owner['warp']<4 and owner['warp_group']==0,'owner bounds')
    return (owner['warp']*32+owner['lane'])*nr+owner['reg']
   require(len(p['global_writers'])==volume and len(s['first'])==volume,'writer coverage')
   for slot,owner in enumerate(p['global_writers']):
    index=compact_index(owner)
    require(s['logical'][index]==slot and index==s['first'][slot],'canonical global writer')
   require(len(p['destinations'])==128*dr,'destination coverage')
   for index,entry in enumerate(p['destinations']):
    thread=index//dr;slot=d['logical'][index];owner=entry['source'];source_index=compact_index(owner)
    require(entry['slot']==slot and s['logical'][source_index]==slot,'destination coordinate')
    require(source_index//nr==thread,'compact owner is not thread-local')
    if mapping is not None:
     preferred=mapping.local_owner(slot,thread)
    else:
     preferred=next((reg for reg in range(nr) if s['logical'][thread*nr+reg]==slot),None)
    require(owner['reg']==preferred,'preferred local owner')
   _network_declaration(p,require)
   return p
  require(0<volume<=65536 and 0<nr<=512 and 0<dr<=512,'bounded domain')
  require((p['threads'],p['warps'],p['source_registers'],p['destination_registers'],p['volume'])==(128,4,nr,dr,volume),'header mismatch')
  require(p['payload']==dict(bits=32,words=[dict(bit_offset=0,live_bits=32)]),'unsupported payload')
  require(len(s['logical'])==128*nr and len(d['logical'])==128*dr,'map size')
  require(p['source_slots']==s['logical'] and set(s['logical'])==set(range(volume)) and set(d['logical'])==set(range(volume)),'logical coverage')
  local={};warp={}
  for i,slot in enumerate(s['logical']):
   local.setdefault((slot,i//nr),i);warp.setdefault((slot,i//nr//32),i)
  def index(o):
   require(set(o)=={'reg','lane','warp','warp_group'} and all(type(x)==int for x in o.values()),'owner fields')
   require(0<=o['reg']<nr and 0<=o['lane']<32 and 0<=o['warp']<4 and o['warp_group']==0,'owner bounds')
   return (o['warp']*32+o['lane'])*nr+o['reg']
  require(len(p['global_writers'])==volume,'writer coverage')
  for slot,o in enumerate(p['global_writers']):require(s['logical'][index(o)]==slot,'writer coordinate')
  require(len(p['destinations'])==128*dr,'destination coverage');mode='Register'
  for i,e in enumerate(p['destinations']):
   j=index(e['source']);t=i//dr;slot=d['logical'][i]
   require(e['slot']==slot and s['logical'][j]==slot,'destination coordinate')
   require(j==local.get((slot,t),warp.get((slot,t//32),j)),'preferred owner')
   if j//nr//32!=t//32:mode='Shared'
   elif j//nr!=t and mode=='Register':mode='Shuffle'
  require(p['mode']==mode,'wrong route')
  require(p['scratch_bytes']==(volume*4 if mode=='Shared' else 0),'scratch mismatch')
  require(mode!='Shuffle' or sum(len(shuffle_source_support(p,r)) for r in range(dr))<=65536,'shuffle emitted-work limit')
  _network_declaration(p,require)
  return p
 except (KeyError,TypeError,IndexError) as e:raise ValueError('malformed redistribution') from e


class _UncacheableContent(Exception):
 pass

def content_digest(value):
 """Conservative typed content identity; never deserialize pickle data.

 Dict order and aliasing may cause extra misses, never weaker equality. Exact
 known algebra objects are serialized from their complete state, not id/repr.
 Unknown custom objects use the ordinary full validator without memoization.
 """
 import hashlib,pickle
 from execution_support import LogicalTable,SymbolicMap,_ImageBasis
 from storage_addresses import RowMajorAddresses
 algebra=(LogicalTable,SymbolicMap,_ImageBasis)
 class Sink:
  def __init__(self):self.digest=hashlib.sha256()
  def write(self,data):self.digest.update(data);return len(data)
 class ContentPickler(pickle.Pickler):
  def reducer_override(self,obj):
   if type(obj) is RowMajorAddresses:
    return (dict,([("__typed_class__","storage_addresses.RowMajorAddresses"),("__state__",obj.complete_state())],))
   if type(obj) in algebra:
    name=type(obj).__module__+'.'+type(obj).__qualname__
    return (dict,([('__typed_class__',name),('__state__',dict(vars(obj)))],))
   if obj is dict:return NotImplemented
   raise _UncacheableContent('custom object is not a checked content value')
 sink=Sink()
 try:ContentPickler(sink,protocol=5).dump(value)
 except (_UncacheableContent,pickle.PickleError,TypeError,AttributeError,RuntimeError):return None
 return sink.digest.hexdigest()


class ProofCache:
 """Bounded single-emission success receipts keyed by complete current content.

 Every lookup hashes the entire operation and both values again. Failed proofs
 are not cached, callers never receive cached mutable plans, and mutation of a
 returned plan forces a fresh proof. Default direct validate() remains uncached.
 """
 def __init__(self,max_entries=256,validator=None):
  if type(max_entries) is not int or max_entries<1:raise ValueError('proof cache bound')
  self.max_entries=max_entries;self.validator=validate if validator is None else validator
  self.receipts={};self.hits=0;self.misses=0;self.uncacheable=0
  self._validator_identity=(self.validator,getattr(self.validator,"__code__",None))
 def validate(self,operation,values):
  if self._validator_identity[0] is not self.validator or self._validator_identity[1] is not getattr(self.validator,"__code__",None):
   self.receipts.clear();self._validator_identity=(self.validator,getattr(self.validator,"__code__",None))
  try:
   source=values[operation['operands'][0]];target=values[operation['results'][0]]
   key=content_digest(('frisk-redistribution-full-content-proof-v1',operation,source,target))
  except (KeyError,IndexError,TypeError):key=None
  if key is not None and key in self.receipts:
   self.hits+=1;return operation['redistribution']
  self.misses+=1
  result=self.validator(operation,values)
  if key is None:self.uncacheable+=1
  elif result is operation.get('redistribution') and len(self.receipts)<self.max_entries:self.receipts[key]=True
  return result
