import unittest
import decision_priority as p
import resource_decision as r


def evidence(arm,counts,primary=1):
    x=dict(schema_version=1,contract=p.CONTRACT,implementation_sha256=p.IMPLEMENTATION_SHA256,
        kind='candidate-evidence',arm=arm,static_cost_tuple=[primary]+[0]*8,
        static_model_id='test',static_profile_sha256='0'*64,frequency_mode='unit',prepared_digest='same',
        native_sha256='0'*64,cubin_sha256='1'*64,bindings={'correctness.json':'2'*64},
        gf2=dict(status='available',global_sectors32=1),local=dict(status='available',bytes_per_thread=counts[1]),uses_gpu_timing=False)
    x['compiled_resource_frontier']=dict(contract=r.CONTRACT,implementation_sha256=r.IMPLEMENTATION_SHA256,
        counts=dict(zip(r.DIMENSIONS,counts)),limits_sha256='3'*64,max_warps_per_sm=64,
        cubin_sha256='1'*64,correctness_sha256='2'*64,uses_gpu_timing=False)
    return p._seal(x)


def matrix(rows):
    return p._seal(dict(schema_version=1,contract=p.AFFINE_CONTRACT,implementation_sha256=p.IMPLEMENTATION_SHA256,
        kind='affine-matrix',status='available',rows={a:dict(status='available',base_ticks=b,counts={}) for a,b in rows.items()},
        dimensions=[],unsupported_candidates=[],coefficient_domain={},uses_gpu_timing=False,basis=p.BASIS))

class ResourceDecisionTest(unittest.TestCase):
    def test_dominated_resource_candidate_removed(self):
        ev={'a':evidence('a',(64,10,1024,32)),'b':evidence('b',(32,10,1024,16))}
        out=r.pareto_fallback(ev,['a','b'])['payload']
        self.assertEqual(out['frontier'],['b']);self.assertEqual(out['domination_witnesses'],{'a':'b'})
    def test_reg_local_tradeoff_incomparable(self):
        ev={'a':evidence('a',(32,100,0,0)),'b':evidence('b',(255,10,0,56))}
        out=r.pareto_fallback(ev,['a','b'])['payload']
        self.assertEqual(set(out['frontier']),{'a','b'});self.assertFalse(out['domination_witnesses'])
        d=p.select_affine(r.extend_affine_matrix(matrix({'a':1,'b':1}),ev),fallback_arm='a',priority=['a','b'])['payload']
        self.assertEqual(d['status'],'fallback');self.assertEqual(d['universal_argmin_members'],[])
    def test_resource_preference_never_escapes_primary_minimum(self):
        ev={'a':evidence('a',(255,100,100,56),1),'b':evidence('b',(1,0,0,0),2)}
        self.assertEqual(r.pareto_fallback(ev,['a','b'])['payload']['frontier'],['a'])
    def test_added_unknown_axis_only_restricts_certificate(self):
        ev={'a':evidence('a',(255,100,0,56)),'b':evidence('b',(32,0,0,0))}
        base=matrix({'a':1,'b':2})
        self.assertEqual(p.select_affine(base,fallback_arm='b',priority=['a','b'])['payload']['universal_argmin_members'],['a'])
        d=p.select_affine(r.extend_affine_matrix(base,ev),fallback_arm='b',priority=['a','b'])['payload']
        self.assertEqual(d['status'],'fallback')
        for a,w in d['necessity_counterexamples'].items():
            rows=r.extend_affine_matrix(base,ev)['payload']['rows'];other=w['opponent'];theta=w['coefficients']
            cost=lambda x:rows[x]['base_ticks']+sum(rows[x]['counts'][k]*v for k,v in theta.items())
            self.assertGreater(cost(a),cost(other))
    def test_whole_pool_and_device_identity_are_strict(self):
        ev={'a':evidence('a',(32,0,0,0)),'b':evidence('b',(32,0,0,0))}
        ev['b']['payload']['compiled_resource_frontier']['limits_sha256']='4'*64
        ev['b']['payload_sha256']=p.digest(ev['b']['payload'])
        with self.assertRaises(ValueError):r.resource_rows(ev)
    def test_missing_or_tampered_proof_fails_closed(self):
        ev={'a':evidence('a',(32,0,0,0))}
        ev['a']['payload']['compiled_resource_frontier']['counts']['registers_per_thread']=True
        ev['a']['payload_sha256']=p.digest(ev['a']['payload'])
        with self.assertRaises(ValueError):r.resource_rows(ev)

if __name__=='__main__':unittest.main()
