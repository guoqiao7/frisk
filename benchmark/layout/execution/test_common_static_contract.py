import copy
import hashlib
import unittest
import decision_priority as p
from test_resource_decision import evidence


def contract():
    names=['feature'+str(i) for i in range(20)]
    x=dict(contract='sm90-network-static-common-v1',grammar='ordered-surviving-network-source-issues-v1',
        frequency_mode='unit',weight_provenance='migrated-engineering-proxy-not-refit',
        units='nonnegative-score-ticks-per-feature-unit-not-cycles',feature_order=names,
        coefficients={n:i for i,n in enumerate(names)})
    lines=[x[n] for n in ('contract','grammar','frequency_mode','weight_provenance','units')]
    x['sha256']=hashlib.sha256(('\n'.join(lines+[n+'='+str(x['coefficients'][n]) for n in names])+'\n').encode()).hexdigest()
    return x


def comparison(arm,objective):
    e=evidence(arm,(32,0,0,0));v=e['payload'];v.update(common_static_contract=contract(),
        static_base_model_id='auto',static_base_profile_sha256='0'*64,
        static_model_id=objective,static_effective_model_id=objective,
        static_profile_sha256=('1' if objective=='auto' else '2')*64,
        static_effective_profile_sha256=('1' if objective=='auto' else '2')*64)
    e['payload_sha256']=p.digest(v);return e

class CommonStaticContractTest(unittest.TestCase):
    def test_compares_units_without_erasing_actual_identity(self):
        ev={'a':comparison('a','auto'),'b':comparison('b','work')}
        self.assertEqual(set(p._evidence(ev)),{'a','b'})
        self.assertNotEqual(ev['a']['payload']['static_model_id'],ev['b']['payload']['static_model_id'])
    def test_changed_weights_without_rehash_rejected(self):
        c=contract();c['coefficients']['feature0']=99
        with self.assertRaises(ValueError):p.common_static_contract(c)
    def test_unit_and_boolean_coefficients_rejected(self):
        c=contract();c['coefficients']['feature0']=True
        with self.assertRaises(ValueError):p.common_static_contract(c)
        c=contract();c['units']='cycles'
        with self.assertRaises(ValueError):p.common_static_contract(c)
    def test_actual_effective_identity_mismatch_rejected(self):
        e=comparison('a','work');e['payload']['static_effective_model_id']='auto';e['payload_sha256']=p.digest(e['payload'])
        with self.assertRaises(ValueError):p._evidence({'a':e})
    def test_mixed_frequency_rejected(self):
        e=comparison('a','auto');e['payload']['frequency_mode']='dynamic';e['payload_sha256']=p.digest(e['payload'])
        with self.assertRaises(ValueError):p._evidence({'a':e})

if __name__=='__main__':unittest.main()
