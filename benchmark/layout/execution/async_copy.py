"""Finite-domain validation and plans for the bounded async Copy backend."""
import math

def addresses(v,shape):
 s=v['storage'];a=s['addresses'];result={}
 for x in a:
  point=tuple(x['view'])
  if len(point)!=len(shape) or any(c<0 or c>=n for c,n in zip(point,shape)) or point in result:raise ValueError('duplicate/outside Copy coordinate')
  if x['begin_bit']%8 or x['end_bit']-x['begin_bit']!=16:raise ValueError('Copy requires f16 16-bit endpoints')
  result[point]=x['begin_bit']//8
 if len(result)!=math.prod(shape) or 'xf16' not in v['type'].split(',')[0]:raise ValueError('incomplete f16 Copy map')
 return result

def plan(o,values):
 c=o['attrs']['frisk.copy_contract'];path=c['path'];shape=[64,16]
 required={'version','path','target','vector_bytes','valid_shape','guarded','issuer','completion'}|({'descriptor'} if path=='tma' else set())
 if set(c)!=required or c['version']!=1 or c['target']!='sm_90a' or c['valid_shape']!=shape or c['guarded']!=0:raise ValueError('unsupported async Copy contract')
 if o['attrs']['frisk.writer_policy']!='first_owner':raise ValueError('Copy writer policy')
 src,dst=[values[x] for x in o['operands'][:2]];ss,ds=src['storage'],dst['storage']
 if 'memory_space Global' not in ss['layout'] or 'memory_space Shared' not in ds['layout']:raise ValueError('async Copy direction')
 if ss['alignment']<128 or ds['alignment']<256:raise ValueError('async root alignment')
 sa,da=addresses(src,shape),addresses(dst,shape)
 for view in (src,dst):
  root=values[view['storage']['root']]['storage']
  if root['alignment']!=view['storage']['alignment']:raise ValueError('Copy root/view alignment guarantee differs')
  root_entries={tuple(x['view']):(x['begin_bit'],x['end_bit']) for x in root['addresses']}
  if len(root_entries)!=len(root['addresses']):raise ValueError('duplicate root footprint')
  for entry in view['storage']['addresses']:
   if root_entries.get(tuple(entry['root']))!=(entry['begin_bit'],entry['end_bit']):raise ValueError('Copy view exceeds or differs from actual root footprint')
 intervals=sorted((x['begin_bit'],x['end_bit']) for x in dst['storage']['addresses'])
 if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):raise ValueError('Copy destination physical collision')
 if path=='cp.async':
  width=c['vector_bytes'];e=o['execution'];n=e['topology'][0]
  if c['issuer']!='execution-map' or c['completion']!='commit-wait-and-cta-visibility' or width not in (4,8,16) or e['topology'][1:]!=[32,4,1,1] or len(e['logical'])!=128*n or n%(width//2):raise ValueError('cp.async width/issuer/completion/topology')
  groups=[];coverage=[]
  for reg in range(0,n,width//2):
   sr=[];dr=[]
   for tid in range(128):
    slots=e['logical'][tid*n+reg:tid*n+reg+width//2];points=[divmod(x,16) for x in slots]
    if any(e['first'][x]!=tid*n+reg+i for i,x in enumerate(slots)):raise ValueError('cp.async first owner')
    sb,db=sa[points[0]],da[points[0]]
    if sb%width or db%width or any(sa[p]!=sb+2*i or da[p]!=db+2*i for i,p in enumerate(points)):raise ValueError('cp.async noncontiguous/aligned vector')
    sr.append(sb);dr.append(db);coverage.extend(points)
   groups.append((sr,dr))
  if len(coverage)!=1024 or len(set(coverage))!=1024:raise ValueError('cp.async coverage')
  return {'path':path,'width':width,'groups':groups}
 if path!='tma' or c['issuer']!='elected-cta-thread' or c['completion']!='mbarrier-and-async-proxy' or c['vector_bytes']!=16:raise ValueError('TMA issuer/completion/width')
 d=c['descriptor']
 if set(d)!={'box','global_strides','source_offset','destination_offset','swizzle_bytes','interleave'} or d['box']!=[16,64] or d['global_strides']!=[32] or d['interleave']!='none' or d['swizzle_bytes'] not in (0,32):raise ValueError('unsupported TMA descriptor')
 so,do=d['source_offset'],d['destination_offset'];sw=d['swizzle_bytes']
 if so<0 or so%16 or do<0 or do%(sw*8 if sw else 128):raise ValueError('TMA descriptor offset alignment')
 for point in sa:
  flat=point[0]*16+point[1];offset=do+flat*2;expected=offset^(((offset>>7)&(sw//16-1))<<4) if sw else offset
  if sa[point]!=so+flat*2 or da[point]!=expected:raise ValueError('TMA actual map differs from descriptor')
 return {'path':path,'descriptor':d,'source_root':ss['root'],'bytes':2048}
