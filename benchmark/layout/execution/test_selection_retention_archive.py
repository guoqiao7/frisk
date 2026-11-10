import copy
import unittest
import network_archive as na
import selection_retention_archive as ra
from test_network_archive import fixture as v2_fixture


def fixture(changed=True, mode='J'):
    record, archive = v2_fixture(changed=changed, mode=mode)
    search = archive['search'];search['version'] = archive['version'] = ra.VERSION
    for entry in archive['entries']:
        entry['retention_structure'] = dict(contract=ra.STRUCTURE_CONTRACT,proven=False,reason='unavailable finite target',
            communication_signature=[],cross_thread_boundaries=0,boundaries=[],ownership=[dict(variable=0,proven=False,active=True,logical_bits=0,
            register_nullity=0,carrier_columns=[[] for _ in range(5)],register_basis=[],warp_basis=[])])
        if entry['network_variant']:
            entry['retention_reason'] = 'fixed-min-work-template'
        elif entry['local_incumbent']:
            entry['retention_reason'] = 'local-incumbent-pin';entry['origins'] = ['local-incumbent']
        elif entry['joint_incumbent']:
            entry['retention_reason'] = 'joint-incumbent-pin';entry['origins'] = ['joint-incumbent']
        else:
            entry['retention_reason'] = 'paid-feature-frontier-diversity';entry['origins'] = ['local-completion']
    pin_count = sum(e['local_incumbent'] or e['joint_incumbent'] for e in archive['entries'])
    search.update(paid_complete_observations=10,observed_duplicates=2,frontier_rejections=0,capacity_discards=0)
    reasons = {key:0 for key in ra.DISCARD_REASONS}
    reasons.update({'final-pin-identity-duplicate':pin_count, 'final-native-capacity':4})
    examples = [dict(reason='final-native-capacity',assignment_stable_ordinals=[i],
        feature_values=[i]*len(ra.AXES),origin='local-completion',main_score=i,
        probes_at_observation=10+i) for i in range(4)]
    search['retention'] = dict(contract=ra.CONTRACT,policy=ra.POLICY,reservoir_capacity=32,
        reservoir_peak=8,observer_working_set_peak=8,pending_incoming_capacity=1,
        feature_axes=list(ra.AXES),observations_with_recognized_features=8,
        feature_values_are_physical_register_predictions=False,
        feature_zero_contract=ra.ZERO_CONTRACT,identity_duplicate_scope=ra.IDENTITY_SCOPE,
        extra_candidate_scores=0,certificate_rebuilds=8-pin_count,
        structure_contract=ra.STRUCTURE_CONTRACT,structural_frontier=ra.STRUCTURAL_FRONTIER,
        structure_proven_observations=0,structure_unknown_observations=8,
        structure_layout_cache_hits=10,structure_layout_cache_misses=4,
        structure_distance_cache_hits=0,structure_distance_cache_misses=0,
        discard_reason_counts=reasons,discard_examples=examples)
    return record,archive


class SelectionRetentionArchiveTest(unittest.TestCase):
    def test_v3_all_modes_and_variant_outcomes_share_fixed_budget_proof(self):
        for mode in ('J','LI'):
            for changed in (False,True):
                with self.subTest(mode=mode,changed=changed):
                    self.assertEqual(len(ra.check_archive(*fixture(changed,mode),mode)),4)
    def test_exact_new_search_args_and_zero_search_replay(self):
        self.assertEqual(ra.search_args(),na.search_args()+['--archive-retention=paid-feature-frontier-v1'])
        record,archive=fixture();entry=archive['entries'][3]
        self.assertEqual(ra.replay_args(entry),['--network-objective=min-work'])
        replay=dict(entry,status='success',selection_bypassed=True,
            budget_contract=dict(search_performed=False,limit=65536,refinement_limit=256,used=0,refinement_used=0))
        self.assertTrue(ra.check_replay(replay,entry))
        entry['retention_reason']='local-incumbent-pin'
        with self.assertRaises(ValueError):ra.check_replay(replay,entry)
    def test_static_only_ablation_has_its_own_explicit_contract(self):
        record,archive=fixture();receipt=archive['search']['retention']
        receipt.update(contract=ra.STATIC_CONTRACT,policy=ra.STATIC_POLICY,structural_frontier='disabled; static-only feature ablation')
        self.assertEqual(len(ra.check_archive(record,archive,'J',expected_policy=ra.STATIC_POLICY)),4)
        with self.assertRaises(ValueError):ra.check_archive(record,archive,'J')
        self.assertEqual(ra.search_args(policy=ra.STATIC_POLICY)[-1],'--archive-retention='+ra.STATIC_POLICY)
    def test_v2_contract_does_not_admit_v3_fields_or_version(self):
        self.assertEqual(len(na.check_archive(*v2_fixture(),'J')),4)
        record,archive=fixture()
        with self.assertRaises(ValueError):na.check_archive(record,archive,'J')
        archive['version']=archive['search']['version']=na.VERSION
        with self.assertRaises(ValueError):na.check_archive(record,archive,'J')
    def test_retention_schema_axes_resource_and_score_claims_are_strict(self):
        attacks=[('unknown',0),('policy','legacy-hamming-v2'),('contract','v0'),
            ('reservoir_capacity',33),('pending_incoming_capacity',2),
            ('feature_values_are_physical_register_predictions',True),
            ('extra_candidate_scores',1),('extra_candidate_scores',False),
            ('feature_zero_contract','zero means free'),('identity_duplicate_scope','features'),
            ('reservoir_peak',33),('observer_working_set_peak',9)]
        for key,value in attacks:
            record,archive=fixture();archive['search']['retention'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):ra.check_archive(record,archive,'J')
        record,archive=fixture();archive['search']['retention']['feature_axes'].reverse()
        with self.assertRaises(ValueError):ra.check_archive(record,archive,'J')
    def test_paid_reconstruction_discard_mass_and_reason_cannot_drift(self):
        for attack in ('observed','duplicate','frontier','capacity','reconstruction','final-mass','examples','example-key','example-probe','example-reason','entry-reason'):
            record,archive=fixture();search=archive['search'];retention=search['retention']
            if attack=='observed':search['paid_complete_observations']=17
            elif attack=='duplicate':search['observed_duplicates']=11
            elif attack=='frontier':search['frontier_rejections']=1
            elif attack=='capacity':search['capacity_discards']=1
            elif attack=='reconstruction':retention['certificate_rebuilds']+=1
            elif attack=='final-mass':retention['discard_reason_counts']['final-native-capacity']+=1
            elif attack=='examples':retention['discard_examples'].pop()
            elif attack=='example-key':retention['discard_examples'][0]['assignment_stable_ordinals']=[]
            elif attack=='example-probe':retention['discard_examples'][0]['probes_at_observation']=1000
            elif attack=='example-reason':retention['discard_examples'][0]['reason']='bounded-reservoir-dominated'
            elif attack=='entry-reason':archive['entries'][2]['retention_reason']='local-incumbent-pin'
            with self.subTest(attack=attack),self.assertRaises(ValueError):ra.check_archive(record,archive,'J')
    def test_li_cannot_import_another_search_or_relabel_a_pin(self):
        record,archive=fixture(mode='LI');archive['entries'][2]['origins']=['joint-search']
        with self.assertRaises(ValueError):ra.check_archive(record,archive,'LI')
        record,archive=fixture();archive['entries'][0]['origins']=['local-completion']
        with self.assertRaises(ValueError):ra.check_archive(record,archive,'J')
    def test_gf2_columns_canonical_images_kernel_and_inclusion_are_independent_proofs(self):
        def attribute(raw):
            return '#frisk.distributed<map = #frisk.bit_linear<inputs = ["register", "lane"], input_bits = [1, 1], outputs = ["d"], output_bits = [2], matrix = dense<"0x'+raw+'"> : tensor<2x2xi1>>, topology = [2, 2, 1, 1, 1], replication = 1>'
        owners=[dict(variable=0,proven=True,active=True,logical_bits=2,register_nullity=0,
            carrier_columns=[[1],[2],[],[],[]],register_basis=[1],warp_basis=[1,2]),
            dict(variable=1,proven=True,active=True,logical_bits=2,register_nullity=0,
            carrier_columns=[[2],[1],[],[],[]],register_basis=[2],warp_basis=[1,2])]
        packet=dict(assignment=[dict(variable=0,candidate=dict(attribute=attribute('09'))),
            dict(variable=1,candidate=dict(attribute=attribute('06')))],retention_structure=dict(
            contract=ra.STRUCTURE_CONTRACT,proven=True,reason='',ownership=owners,
            communication_signature=[1],cross_thread_boundaries=1,
            boundaries=[dict(constraint=0,source=0,target=1,register_defect=1,warp_defect=0,scope=1)]))
        self.assertTrue(ra._structure(packet)['proven'])
        for attack in ('column','basis','kernel','defect','scope','signature','missing-owner'):
            changed=copy.deepcopy(packet);structure=changed['retention_structure']
            if attack=='column':structure['ownership'][0]['carrier_columns'][0]=[2]
            elif attack=='basis':structure['ownership'][0]['warp_basis']=[2,1]
            elif attack=='kernel':structure['ownership'][0]['register_nullity']=1
            elif attack=='defect':structure['boundaries'][0]['register_defect']=0
            elif attack=='scope':structure['boundaries'][0]['scope']=2
            elif attack=='signature':structure['communication_signature']=[0]
            else:structure['ownership'].pop()
            with self.subTest(attack=attack),self.assertRaises(ValueError):ra._structure(changed)
    def test_common_network_identity_proof_is_not_bypassed(self):
        record,archive=fixture();archive['entries'][3]['complete_program_sha256']='0'*64
        with self.assertRaises(ValueError):ra.check_archive(record,archive,'J')


class SourceGrammarRoutingTests(unittest.TestCase):
    @staticmethod
    def routed(eligible):
        record, archive = fixture()
        if not eligible:
            record, archive = copy.deepcopy((record,archive))
            record['candidate_archive_search'] = archive['search']
            archive['version'] = archive['search']['version'] = na.VERSION
            for entry in archive['entries']:
                entry.pop('retention_reason');entry.pop('retention_structure')
            receipt = archive['search']['retention']
            receipt.update(contract='paid-fullcost-hamming-v2',policy='legacy-hamming-v2',
                reservoir_capacity=4,reservoir_peak=4,observer_working_set_peak=5,
                structural_frontier='disabled; static-only feature ablation',
                structure_proven_observations=0,structure_unknown_observations=0,
                structure_layout_cache_hits=0,structure_layout_cache_misses=0,
                structure_distance_cache_hits=0,structure_distance_cache_misses=0)
            reasons = receipt['discard_reason_counts']
            reasons.update({'legacy-hamming-capacity':4,'final-pin-identity-duplicate':0,'final-native-capacity':2})
            receipt['discard_examples'] = receipt['discard_examples'][:2]
            archive['search']['capacity_discards'] = 4
        gate = dict(contract=ra.SOURCE_GRAMMAR_CONTRACT,status='eligible' if eligible else 'unknown',
            reason='' if eligible else 'unknown operation math.log',source_eligibility_only=True,
            native_output_guaranteed=False,checked_operations=20,checked_variables=10,
            checked_candidates=48,kernel_count=1,parallel_count=1,threads=128,grid=[1],
            source_target='sm_90a',execution_domain='sm90-scalar-f32-dag-v1',grammar=ra.SOURCE_GRAMMAR)
        archive['search']['retention_resolution'] = dict(contract=ra.ROUTING_CONTRACT,
            requested=ra.ROUTING_POLICY,effective=ra.POLICY if eligible else 'legacy-hamming-v2',
            resolved_before_search=True,resolver_calls=1,additional_searches=0,
            native_output_guaranteed=False,gate=gate)
        record['candidate_archive_search'] = copy.deepcopy(archive['search'])
        return record, archive

    def test_actual_v2_or_v3_representation_is_checked_without_relabeling(self):
        for eligible in (False,True):
            record, archive = self.routed(eligible)
            self.assertEqual(len(ra.check_archive(record,archive,'J',expected_policy=ra.ROUTING_POLICY)),4)
            self.assertEqual(archive['version'],ra.VERSION if eligible else na.VERSION)
        self.assertEqual(ra.search_args(policy=ra.ROUTING_POLICY)[-1],
                         '--archive-retention='+ra.ROUTING_POLICY)

    def test_unknown_source_cannot_force_gf2_or_claim_native_proof_or_extra_search(self):
        attacks = [('route','effective',ra.POLICY),('route','native_output_guaranteed',True),
            ('route','additional_searches',1),('route','resolver_calls',2),
            ('route','resolver_calls',True),('route','resolved_before_search',False),
            ('gate','status','eligible'),('gate','reason',''),
            ('gate','source_eligibility_only',False),('gate','native_output_guaranteed',True),
            ('gate','unvalidated','field'),('gate','checked_candidates',-1),('gate','grammar','unrelated grammar')]
        for where,key,value in attacks:
            record,archive=self.routed(False);route=archive['search']['retention_resolution']
            (route if where=='route' else route['gate'])[key]=value
            record['candidate_archive_search']=copy.deepcopy(archive['search'])
            with self.subTest(where=where,key=key),self.assertRaises(ValueError):
                ra.check_archive(record,archive,'J',expected_policy=ra.ROUTING_POLICY)

    def test_legacy_recipe_does_not_ignore_new_observer_or_resource_receipts(self):
        for key,value in [('extra_candidate_scores',999),('contract','fake-v3'),
            ('feature_values_are_physical_register_predictions',True),('reservoir_capacity',32),
            ('structure_proven_observations',8),('unknown','silently ignored')]:
            record,archive=self.routed(False);archive['search']['retention'][key]=value
            record['candidate_archive_search']=copy.deepcopy(archive['search'])
            with self.subTest(key=key),self.assertRaises(ValueError):
                ra.check_archive(record,archive,'J',expected_policy=ra.ROUTING_POLICY)
        record,archive=self.routed(False);archive['search']['paid_complete_observations']=999999
        record['candidate_archive_search']=copy.deepcopy(archive['search'])
        with self.assertRaises(ValueError):ra.check_archive(record,archive,'J',expected_policy=ra.ROUTING_POLICY)

    def test_eligible_launch_domain_and_entire_candidate_domain_are_required(self):
        for key,value in [('threads',64),('grid',[2]),('kernel_count',2),('parallel_count',2),
            ('execution_domain','unknown'),('source_target','sm_80'),('checked_variables',0),
            ('checked_candidates',0),('reason','no DIV is not proof')]:
            record,archive=self.routed(True)
            archive['search']['retention_resolution']['gate'][key]=value
            record['candidate_archive_search']=copy.deepcopy(archive['search'])
            with self.subTest(key=key),self.assertRaises(ValueError):
                ra.check_archive(record,archive,'J',expected_policy=ra.ROUTING_POLICY)
        record,archive=self.routed(True);archive['version']=archive['search']['version']=na.VERSION
        record['candidate_archive_search']=copy.deepcopy(archive['search'])
        with self.assertRaises(ValueError):ra.check_archive(record,archive,'J',expected_policy=ra.ROUTING_POLICY)


if __name__=='__main__':unittest.main()
