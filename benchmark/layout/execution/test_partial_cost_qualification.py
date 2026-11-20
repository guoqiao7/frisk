import unittest,copy
import partial_cost_qualification as q
class ArchiveContractTest(unittest.TestCase):
 def fixture(self,mode='joint'):
  total=dict(probes=10,relationship=5,complete_scores=3,refinement=2,refinement_probes=2)
  e=lambda h,l,j:dict(complete_assignment_sha256=h*64,prepared_digest='pkg',conversion_policy='independent',certificate_verified=True,cost=dict(cost=[1]*9,saturated=False),probes_at_observation=10,refinement_at_observation=2,local_incumbent=l,joint_incumbent=j)
  entries=[e('a',True,mode=='local')]
  if mode=='joint':entries.append(e('b',False,True))
  c=dict(version='shared-complete-candidate-archive-v1',limit=65536,refinement_limit=256,capacity=4,total=total,local_end=total,local_terminal_probes=16384 if mode=='joint' else 65536,local_terminal_refinement=256,retained=len(entries),local_status='complete-witness',joint_status='complete-witness' if mode=='joint' else'not-requested-local-control')
  c['local']=dict(selection_mode='informed-producer-first',local_objective='own-operations-input-repairs-and-resources-v2',incumbent_full_score_budget_charged=True,total=total)
  c['joint']=dict(total=total)
  for i,e in enumerate(entries):e['index']=i
  return dict(prepared_digest='pkg',complete_assignment_sha256=entries[-1]['complete_assignment_sha256'],candidate_archive_search=c,total=total,budget_contract=dict(kind='shared-total-search-probes-v1',search_performed=True,limit=65536,refinement_limit=256,used=10,refinement_used=2)),dict(schema_version=1,capacity=4,prepared_digest='pkg',entries=entries,search=c)
 def test_valid_shared_and_local_receipts(self):
  for m in ('joint','local'):
   r,a=self.fixture(m);self.assertEqual(q.check_archive(r,a,m,4)['total']['probes'],10)
 def test_reset_counters_rejected(self):
  r,a=self.fixture();r['candidate_archive_search']['local_end']=dict(probes=11,relationship=6,complete_scores=3,refinement=2,refinement_probes=2)
  with self.assertRaises(ValueError):q.check_archive(r,a,'joint',4)
 def test_unaccounted_work_rejected(self):
  r,a=self.fixture();r['candidate_archive_search']['total']['probes']+=1
  with self.assertRaises(ValueError):q.check_archive(r,a,'joint',4)
 def test_uncertified_candidate_rejected(self):
  r,a=self.fixture();a['entries'][0]['certificate_verified']=False
  with self.assertRaises(ValueError):q.check_archive(r,a,'joint',4)
 def test_full_incumbent_cannot_be_omitted(self):
  r,a=self.fixture();r['complete_assignment_sha256']='c'*64
  with self.assertRaises(ValueError):q.check_archive(r,a,'joint',4)
 def test_archive_duplicate_not_feature_dedup(self):
  r,a=self.fixture();a['entries'][1]['complete_assignment_sha256']='a'*64;r['complete_assignment_sha256']='a'*64
  with self.assertRaises(ValueError):q.check_archive(r,a,'joint',4)
 def test_local_cannot_hide_joint_work(self):
  r,a=self.fixture('local');r['candidate_archive_search']['joint_status']='complete-witness'
  with self.assertRaises(ValueError):q.check_archive(r,a,'local',4)
 def test_free_archive_observation_rejected(self):
  r,a=self.fixture();a['entries'][0]['probes_at_observation']=99
  with self.assertRaises(ValueError):q.check_archive(r,a,'joint',4)
 def test_outer_budget_and_index_tamper_rejected(self):
  r,a=self.fixture();r['budget_contract']['used']=0
  with self.assertRaises(ValueError):q.check_archive(r,a,'joint',4)
  r,a=self.fixture();a['entries'][0]['index']=True
  with self.assertRaises(ValueError):q.check_archive(r,a,'joint',4)
 def test_static_tie_prefers_actual_incumbent(self):
  r,a=self.fixture();self.assertEqual(q.static_fallback_entry(r,a)['complete_assignment_sha256'],r['complete_assignment_sha256'])
 def test_refinement_ablations_share_pool_and_isolate_factors(self):
  import unittest.mock as mock
  import test_decision_priority as fixtures
  profile=fixtures.profile();pool=dict(legacy='legacy',modes={})
  analyses={};evidence={}
  for mode in ('joint','local'):
   arms=[mode+'-'+a for a in 'abc'];pool['modes'][mode]=dict(arms=arms,fallback=arms[0],static_order=arms)
   for arm,sectors,frame,conv in zip(arms,(100,10,10),(6928,80,0),(0,1,2)):
    analyses[arm]=dict(arm=arm);evidence[arm]=fixtures.evidence(arm,sectors,frame,100,conv)
  analyses['legacy']={}
  def cost(native,profile):return fixtures.interval(native['arm'],10,profile)
  with mock.patch('partial_cost_selection.cost_interval',side_effect=cost):
   decisions=q.calculate_decisions(pool,analyses,profile,profile,evidence)
  self.assertEqual(set(decisions),set(q.suite.POLICY_LABELS))
  for mode in ('joint','local'):
   self.assertEqual(decisions[mode+'-interval-only']['selected_source'],mode+'-a')
   self.assertEqual(decisions[mode+'-frame-only']['selected_source'],mode+'-a')
   self.assertEqual(decisions[mode+'-gf2-tie']['selected_source'],mode+'-b')
   self.assertEqual(decisions[mode+'-tie-frame']['selected_source'],mode+'-b')
   self.assertEqual(decisions[mode+'-affine-frame']['selected_source'],mode+'-b')
   expected=set(pool['modes'][mode]['arms'])
   for rule in ('interval-only','gf2-tie','frame-only','tie-frame','affine-frame'):
    self.assertEqual(set(decisions[mode+'-'+rule]['intervals']),expected)
   self.assertEqual(decisions[mode+'-static']['selected_source'],mode+'-a')
 def test_cfg_fallback_uses_frame_without_fake_native_bound(self):
  import unittest.mock as mock
  import test_decision_priority as fixtures
  profile=fixtures.profile();pool=dict(legacy='legacy',modes={});analyses={'legacy':{}};evidence={}
  for mode in ('joint','local'):
   arms=[mode+'-'+a for a in 'ab'];pool['modes'][mode]=dict(arms=arms,fallback=arms[0],static_order=arms)
   for arm,frame,conv in zip(arms,(6928,80),(0,1)):
    analyses[arm]=dict(arm=arm);evidence[arm]=fixtures.evidence(arm,100,frame,100,conv)
  def cost(native,profile):return fixtures.interval(native['arm'],0,profile,unsupported=True)
  with mock.patch('partial_cost_selection.cost_interval',side_effect=cost):d=q.calculate_decisions(pool,analyses,profile,profile,evidence)
  for mode in ('joint','local'):
   self.assertEqual(d[mode+'-gf2-tie']['selected_source'],mode+'-a')
   self.assertEqual(d[mode+'-frame-only']['selected_source'],mode+'-b')
   self.assertEqual(d[mode+'-affine-frame']['selected_source'],mode+'-b')
   self.assertEqual(d[mode+'-affine-frame']['status'],'fallback')
if __name__=='__main__':unittest.main()
