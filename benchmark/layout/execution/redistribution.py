"""Check the exported C++ redistribution proof against complete finite maps.
Conversion-internal elected writers are distinct from external first_owner ABI.
"""
import math

def validate(operation, values):
 def require(ok,why):
  if not ok:raise ValueError('redistribution: '+why)
 try:
  p=operation['redistribution'];a,b=[values[i] for i in (operation['operands'][0],operation['results'][0])];s=a['distributed'];d=b['distributed']
  require('xf32' in a['type'] and 'xf32' in b['type'],'f32 payload required')
  require(s['shape']==d['shape'] and s['topology'][1:]==d['topology'][1:]==[32,4,1,1],'unsupported topology/shape')
  nr,dr=s['topology'][0],d['topology'][0];volume=math.prod(s['shape'])
  require(0<volume<=65536 and 0<nr<=256 and 0<dr<=256,'bounded domain')
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
  require(mode!='Shuffle' or nr*dr<=65536,'shuffle code-size limit')
  return p
 except (KeyError,TypeError,IndexError) as e:raise ValueError('malformed redistribution') from e
