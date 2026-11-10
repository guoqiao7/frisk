import copy, unittest
import backend

def case(kind='register',replica=False):
 nr=8*int(replica) if replica else 8
 src=[t*8+r%8 for t in range(128) for r in range(nr)]
 def dest(t,r):
  if kind=='register':return t*8+(r^1)
  if kind=='shuffle':return (t^1)*8+r
  if kind=='dependent':return (t^1)*8+(r^(t&1))
  if kind=='shared':return (t^32)*8+r
  return t*8+r
 dst=[dest(t,r) for t in range(128) for r in range(8)]
 def owner(i):return dict(reg=i%nr,lane=(i//nr)%32,warp=i//nr//32,warp_group=0)
 writers=[owner(src.index(s)) for s in range(1024)]
 destinations=[]
 for i,s in enumerate(dst):
  t=i//8;poss=[j for j,x in enumerate(src) if x==s]
  local=[j for j in poss if j//nr==t];warp=[j for j in poss if j//nr//32==t//32]
  destinations.append(dict(source=owner((local or warp or poss)[0]),slot=s))
 mode='Shared' if kind=='shared' else 'Shuffle' if kind in ('shuffle','dependent') else 'Register'
 p=dict(mode=mode,threads=128,warps=4,source_registers=nr,destination_registers=8,volume=1024,scratch_bytes=4096 if mode=='Shared' else 0,payload=dict(bits=32,words=[dict(bit_offset=0,live_bits=32)]),destinations=destinations,source_slots=src,global_writers=writers)
 values={i:dict(id=i,type='tensor<64x16xf32>',distributed=dict(topology=[n,32,4,1,1],shape=[64,16],logical=slots,first=[slots.index(s) for s in range(1024)])) for i,n,slots in [(0,nr,src),(1,8,dst)]}
 return dict(name='frisk.convert_layout',operands=[0],results=[1],redistribution=p),values

class WarpSectionTests(unittest.TestCase):
 def test_affine_warp_section_uses_immutable_same_warp_replicas(self):
  from access_plan import linear
  from redistribution import affine_warp_sections
  source=[2*(thread//32)+(thread&1) for thread in range(128)]
  destinations=[2*(thread//32)+((thread>>1)&1) for thread in range(128)]
  local_first=[thread%32 if source[thread]==slot else slot%2
               for thread,slot in enumerate(destinations)]
  with self.assertRaises(ValueError):linear(local_first)
  plan=dict(source_registers=1,destination_registers=1,source_slots=source,
            destinations=[dict(slot=slot) for slot in destinations])
  (lanes,regs),=affine_warp_sections(plan)
  self.assertEqual(tuple((thread>>1)&1 for thread in range(128)),lanes)
  self.assertEqual((0,)*128,regs)
  self.assertEqual(destinations,[source[32*(t//32)+lane] for t,lane in enumerate(lanes)])


class Redistribution(unittest.TestCase):
 def test_full_owner_maps_and_instructions(self):
  for kind in ('identity','register','shuffle','dependent','shared'):
   for replica in (False,2,4):
    op,values=case(kind,replica)
    self.assertTrue(hasattr(backend,'validate_redistribution'),'checked route decoder absent')
    p=backend.validate_redistribution(op,values)
    for i,d in enumerate(p['destinations']):
     s=d['source'];j=((s['warp_group']*4+s['warp'])*32+s['lane'])*p['source_registers']+s['reg']
     self.assertEqual(values[0]['distributed']['logical'][j],values[1]['distributed']['logical'][i])
    e=backend.Emitter(dict(values=list(values.values())));e.reg[0]=['s'+str(i) for i in range(p['source_registers'])];e.scratch=0;e.convert(op);code='\n'.join(e.code)
    self.assertEqual('shfl.sync.' in code,p['mode']=='Shuffle')
    self.assertEqual('st.shared.b32' in code,p['mode']=='Shared')
    self.assertEqual(code.count('bar.sync 0;'),2 if p['mode']=='Shared' else 0)
    if p['mode']=='Shuffle':self.assertFalse(any('@' in line for line in code.splitlines() if 'shfl.sync' in line))
 def test_four_lane_replica_election(self):
  self.assertTrue(hasattr(backend.Emitter,'elect'),'physical-index election absent')
  e=backend.Emitter(dict(values=[]))
  pred=e.elect([(t//4)*4*8 for t in range(128)],8,0)
  self.assertIn('setp.eq.u32','\n'.join(e.code))
  self.assertTrue(pred)
 def test_scratch_checked_and_route_aware(self):
  self.assertTrue(hasattr(backend,'reserve_shared'),'checked reservation absent')
  self.assertEqual(backend.reserve_shared(5,16,8,64),24)
  for args in ((2**64-1,1,4,2**64-1),(0,65,4,64),(0,1,3,64)):
   with self.assertRaises(ValueError):backend.reserve_shared(*args)
 def test_corrupt_plan_rejected(self):
  for change in ('owner','missing','slot','mode','scratch','writer'):
   op,v=case('shared');p=op['redistribution']
   if change=='owner':p['destinations'][0]['source']['lane']=32
   if change=='missing':p['destinations'].pop()
   if change=='slot':p['source_slots'][0]=1024
   if change=='mode':p['mode']='Register'
   if change=='scratch':p['scratch_bytes']=2**64
   if change=='writer':p['global_writers'][1]=p['global_writers'][0]
   self.assertTrue(hasattr(backend,'validate_redistribution'),'checked route decoder absent')
   with self.assertRaises(ValueError):backend.validate_redistribution(op,v)
if __name__=='__main__':unittest.main()
