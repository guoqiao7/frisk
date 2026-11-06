import copy
import hashlib
import unittest
import network_archive as na


def fixture(changed=True, mode='J', unavailable=False):
    weights = {n: 1 for n in na.FEATURES}
    for n in ('shared_bank_excess', 'conversion_register_bytes', 'conversion_shuffle_bytes',
              'conversion_shared_bytes', 'conversion_conservative_bytes'):
        weights[n] = 0
    common = dict(contract='sm90-network-static-common-v1', grammar='ordered-surviving-network-source-issues-v1',
        frequency_mode='legacy', weight_provenance='migrated-engineering-proxy-not-refit',
        units='nonnegative-score-ticks-per-feature-unit-not-cycles', feature_order=list(na.FEATURES), coefficients=weights)
    lines = [common[n] for n in ('contract', 'grammar', 'frequency_mode', 'weight_provenance', 'units')]
    common['sha256'] = hashlib.sha256(('\n'.join(lines + [n + '=' + str(weights[n]) for n in na.FEATURES]) + '\n').encode()).hexdigest()
    profile = dict(schema_version=1, model='sm90-layout-network-surviving-proxy-v1', coefficients=weights,
        provenance='migration fixture; no fit', network_objective='min-shfl', weight_provenance=common['weight_provenance'])
    profile_sha, model_id = na._profile(profile, common, 'min-shfl')
    def entry(index, attr, primary, local=False, joint=False, variant=False):
        objective = 'min-work' if variant else 'min-shfl';p = dict(profile, network_objective=objective)
        ph, mid = na._profile(p, common, objective)
        e = dict(index=index, assignment=[dict(variable=0, name='value', candidate=dict(attribute=attr, id=hashlib.sha256(attr.encode()).hexdigest()))],
            instruction_bindings=[], reduction_bindings=[], cost=dict(cost=[primary] + [0]*8, features={'alu_warp_issues': primary}, saturated=False),
            prepared_digest='a'*64, conversion_policy='independent', local_incumbent=local, joint_incumbent=joint,
            probes_at_observation=96 if variant else 20, refinement_at_observation=0, certificate_verified=True,
            network_objective=objective, network_variant=variant, cost_model=mid, cost_profile=p, cost_profile_sha256=ph,
            static_base_model_id=model_id, static_base_profile_sha256=profile_sha, static_effective_model_id=mid,
            static_effective_profile_sha256=ph, common_static_contract=common,
            origins=['fixed-min-work-template'] if variant else ['paid-complete-score'])
        e['complete_assignment_sha256'] = na._digest({n:e[n] for n in ('assignment','instruction_bindings','reduction_bindings')}, True)
        e['complete_program_sha256'] = na._digest({n:e[n] for n in ('complete_assignment_sha256','network_objective','cost_model')}, True)
        return e
    entries = [entry(0,'a',180,local=True),entry(1,'b',168,joint=mode=='J'),entry(2,'c',185),entry(3,'d',190)]
    columns = [8,16,4,1,2,32,64,128,0,0] if changed else [8,2,4,1,16,32,64,128,0,0]
    a,b = na._recompute_pair(na._json(dict(registers=8,affine_base=0,affine_columns=columns)))
    v = dict(contract=na.CONTRACT,reserved_score_probes=1,layout_search_limit=65535,variant_scores=int(changed),
        variant_plan_checks=1,native_capacity=4,objective='min-work',auto_original_indices=[0,1,2,3],static_first_auto_index=1,
        pins=[dict(original_index=0,retained_slot=0,local_incumbent=True,joint_incumbent=False,static_first=False,assignment=entries[0]['assignment']),
              dict(original_index=1,retained_slot=1,local_incumbent=False,joint_incumbent=mode=='J',static_first=True,assignment=entries[1]['assignment'])],
        plan_check_wall_ns=1,plan_comparison_proven=not unavailable,plan_changed=changed,compared_shuffle_boundaries=1,
        plan_comparison_reason='proven maps' if not unavailable else 'unsupported finite target',
        boundary_plans=[dict(constraint=0,source=0,target=1,source_registers=8,target_registers=8,**{'min-shfl':a,'min-work':b})],
        status='min-work-replaces-fourth-auto' if changed else 'unavailable-plan-proof-keep-auto' if unavailable else 'identical-plans-keep-auto',
        unused_reserved_score_probes=int(not changed),budget_limit_restored=65536,budget_refinement_unchanged=256,
        resource_proxy_is_physical_register_prediction=False)
    if changed:
        v.update(score_wall_ns=1,replaced_auto_original_index=3,replaced_auto_assignment=entries[3]['assignment'],variant_slot=3)
        entries[3] = entry(3,'b',144,variant=True)
    total = dict(probes=96 if changed else 95, refinement=0, relationship=80, complete_scores=16 if changed else 15,refinement_probes=0)
    search = dict(version=na.VERSION,capacity=4,retained=4,limit=65536,refinement_limit=256,local_terminal_refinement=256,
        local_terminal_probes=16384 if mode=='J' else 65535,total=total,
        local_end=dict(probes=20,refinement=0,relationship=15,complete_scores=5,refinement_probes=0),
        local_status='complete-witness',joint_status='complete-witness' if mode=='J' else 'not-requested-local-control',network_variant=v)
    if mode == 'LI':
        search['local_end'] = dict(probes=95,refinement=0,relationship=80,complete_scores=15,refinement_probes=0)
        stage1 = dict(probes=48,refinement=0,relationship=40,complete_scores=8,refinement_probes=0)
    else:
        stage1 = dict(probes=10,refinement=0,relationship=8,complete_scores=2,refinement_probes=0)
    stage2 = {n: search['local_end'][n]-stage1[n] for n in stage1}
    search['local'] = dict(selection_mode='informed-producer-first',
        local_objective='own-operations-input-repairs-and-resources-v2',
        incumbent_full_score_budget_charged=True,legal_return=True,search_complete=False,
        local_cost_changed_during_completion=False,stage2_status='budget-exhausted-witness-retained',
        stage1=stage1,stage2=stage2,total=copy.deepcopy(search['local_end']))
    if mode == 'J':
        search['joint'] = dict(legal_return=True,total={n:total[n]-(int(changed) if n in ('probes','complete_scores') else 0) for n in total})
    else:search['joint'] = dict(status='not-requested-local-control')
    record = dict(status='success',selection=mode,conversion_policy='independent',network_objective='min-shfl',prepared_digest='a'*64,
        cost_model=model_id,cost_profile=profile,cost_profile_sha256=profile_sha,static_base_model_id=model_id,
        static_base_profile_sha256=profile_sha,static_effective_model_id=model_id,static_effective_profile_sha256=profile_sha,
        common_static_contract=common,frequency_mode='legacy',prepared_cost_model=model_id,prepared_cost_profile_sha256=profile_sha,candidate_archive_search=search,total=total,
        budget_contract=dict(kind='shared-total-search-probes-v1',search_performed=True,limit=65536,refinement_limit=256,
                             used=total['probes'],refinement_used=0))
    for n in ('assignment','instruction_bindings','reduction_bindings','complete_assignment_sha256','complete_program_sha256'):
        record[n] = copy.deepcopy(entries[1][n])
    archive=dict(schema_version=1,version=na.VERSION,prepared_digest='a'*64,cost_model=model_id,capacity=4,entries=entries,
        search=search,export_contract='fixed replay')
    return record,archive


class NetworkArchiveTest(unittest.TestCase):
    def check(self, pair, mode='J'):
        return na.check_archive(*pair,mode,4)
    def test_real_network_tradeoff_and_both_fair_modes(self):
        for mode in ('J','LI'):
            entries=self.check(fixture(mode=mode),mode)
            self.assertEqual(len(entries),4);self.assertEqual(entries[3]['network_objective'],'min-work')
            self.assertEqual(entries[1]['complete_assignment_sha256'],entries[3]['complete_assignment_sha256'])
            self.assertNotEqual(entries[1]['complete_program_sha256'],entries[3]['complete_program_sha256'])
            self.assertEqual(na.replay_args(entries[3]),['--network-objective=min-work'])
    def test_partial_pool_adds_variant_without_inventing_four_auto(self):
        r,a=fixture();auto,variant=a['entries'][1],a['entries'][3]
        auto.update(index=0,local_incumbent=True);variant['index']=1;a['entries']=[auto,variant]
        v=a['search']['network_variant'];v.update(auto_original_indices=[0],static_first_auto_index=0,variant_slot=1,
            status='min-work-added-within-K4',pins=[dict(original_index=0,retained_slot=0,local_incumbent=True,
            joint_incumbent=True,static_first=True,assignment=auto['assignment'])])
        del v['replaced_auto_original_index'];del v['replaced_auto_assignment'];a['search']['retained']=2
        self.assertEqual(len(self.check((r,a))),2)
        v['status']='min-work-replaces-fourth-auto'
        with self.assertRaises(ValueError):self.check((r,a))
    def test_identical_and_typed_unavailable_keep_four_auto(self):
        for unavailable in (False,True):
            entries=self.check(fixture(changed=False,unavailable=unavailable))
            self.assertTrue(all(not e['network_variant'] for e in entries))
    def test_fixed_capacity_search_args(self):
        self.assertEqual(na.search_args(),['--candidate-archive=4','--network-archive-v2=true'])
        for x in (True,3,5):
            with self.assertRaises(ValueError):na.search_args(x)
    def test_no_extra_native_or_rescore_budget(self):
        for key,value in [('variant_scores',0),('unused_reserved_score_probes',1),('layout_search_limit',65536),
                          ('native_capacity',5),('budget_refinement_unchanged',257)]:
            with self.subTest(key=key):
                r,a=fixture();a['search']['network_variant'][key]=value
                with self.assertRaises(ValueError):self.check((r,a))
        r,a=fixture();a['entries'].append(copy.deepcopy(a['entries'][-1]))
        with self.assertRaises(ValueError):self.check((r,a))
    def test_pin_identity_and_counterpart_cannot_be_forged(self):
        for change in ('pin','counterpart','origin','slot'):
            with self.subTest(change=change):
                r,a=fixture();v=a['search']['network_variant']
                if change=='pin':v['pins'][0]['original_index']=3
                if change=='counterpart':a['entries'][3]['assignment'][0]['candidate']['attribute']='different'
                if change=='origin':a['entries'][3]['origins']=['gpu-label-winner']
                if change=='slot':v['variant_slot']=2
                with self.assertRaises(ValueError):self.check((r,a))
    def test_whole_plan_and_owner_proof_is_recomputed(self):
        for change in ('count','step','map','label_only'):
            with self.subTest(change=change):
                r,a=fixture();v=a['search']['network_variant'];p=v['boundary_plans'][0]
                if change=='count':p['min-work']['counts']['shfl']+=1
                if change=='step':p['min-work']['steps'][0]['lane_bit']=4
                if change=='map':p['min-work']['affine_base']=1
                if change=='label_only':p['min-work']=copy.deepcopy(p['min-shfl'])
                with self.assertRaises(ValueError):self.check((r,a))
    def test_actual_objective_profile_and_common_units_are_bound(self):
        for change in ('id','objective','weight','common','hash','program'):
            with self.subTest(change=change):
                r,a=fixture();e=a['entries'][3]
                if change=='id':e['cost_model']=a['entries'][0]['cost_model']
                if change=='objective':e['network_objective']='min-shfl'
                if change=='weight':e['cost_profile']['coefficients']['shuffle_warp_issues']+=1
                if change=='common':e['common_static_contract']=copy.deepcopy(e['common_static_contract']);e['common_static_contract']['units']='cycles'
                if change=='hash':e['cost_profile_sha256']='b'*64
                if change=='program':e['complete_program_sha256']='b'*64
                with self.assertRaises(ValueError):self.check((r,a))
    def test_consistently_rehashed_foreign_base_profile_is_rejected(self):
        r,a=fixture();e=a['entries'][3];e['cost_profile']=copy.deepcopy(e['cost_profile']);e['cost_profile']['provenance']='different base'
        ph,mid=na._profile(e['cost_profile'],e['common_static_contract'],'min-work')
        bh,bid=na._profile(dict(e['cost_profile'],network_objective='min-shfl'),e['common_static_contract'],'min-shfl')
        e.update(cost_model=mid,static_effective_model_id=mid,cost_profile_sha256=ph,static_effective_profile_sha256=ph,
                 static_base_model_id=bid,static_base_profile_sha256=bh)
        e['complete_program_sha256']=na._digest({n:e[n] for n in ('complete_assignment_sha256','network_objective','cost_model')},True)
        with self.assertRaises(ValueError):self.check((r,a))
        r,a=fixture();r['prepared_cost_profile_sha256']='b'*64
        with self.assertRaises(ValueError):self.check((r,a))
    def test_replay_has_zero_search_and_actual_profile(self):
        r,a=fixture();e=a['entries'][3];replay=dict(e,status='success',selection_bypassed=True,
            budget_contract=dict(search_performed=False,limit=65536,refinement_limit=256,used=0,refinement_used=0))
        self.assertTrue(na.check_replay(replay,e))
        replay['budget_contract']['used']=1
        with self.assertRaises(ValueError):na.check_replay(replay,e)
    def test_replay_rejects_independently_drifted_or_missing_program_declaration(self):
        r,a=fixture();e=a['entries'][3]
        replay=dict(e,status='success',selection_bypassed=True,
            budget_contract=dict(search_performed=False,limit=65536,refinement_limit=256,used=0,refinement_used=0))
        self.assertTrue(na.check_replay(replay,e))
        for value in ('0'*64,None):
            mutated=copy.deepcopy(replay)
            if value is None:del mutated['complete_program_sha256']
            else:mutated['complete_program_sha256']=value
            with self.subTest(value=value),self.assertRaises(ValueError):na.check_replay(mutated,e)
    def test_strict_schema_booleans_and_cost_feature_receipt(self):
        for change in ('unknown','bool','cost','unpaid','mode'):
            r,a=fixture()
            if change=='unknown':a['entries'][0]['extra']=0
            if change=='bool':a['entries'][0]['network_variant']=0
            if change=='cost':a['entries'][0]['cost']['cost'][0]+=1
            if change=='unpaid':a['entries'][0]['probes_at_observation']=65536
            if change=='mode':r['selection']='LI'
            with self.subTest(change=change),self.assertRaises(ValueError):self.check((r,a))


    def test_informed_scope_and_charge_cannot_be_synchronously_relabelled(self):
        for mode in ('J','LI'):
            for field,value in (('selection_mode','naive-local'),('local_objective','omit-input-repairs'),
                                ('incumbent_full_score_budget_charged',False),('incumbent_full_score_budget_charged',1)):
                r,a=fixture(mode=mode);r,a=copy.deepcopy((r,a))
                r['candidate_archive_search']['local'][field]=value;a['search']=r['candidate_archive_search']
                with self.subTest(mode=mode,field=field,value=value),self.assertRaises(ValueError):self.check((r,a),mode)
    def test_stage_phase_and_refinement_counters_are_conserved(self):
        for mode in ('J','LI'):
            for attack in ('local-total','stage-sum','refinement','extra-phase','joint-end'):
                if attack=='joint-end' and mode!='J':continue
                r,a=fixture(mode=mode);c=a['search']
                target=c['local']['total'] if attack=='local-total' else c['local']['stage2'] if attack=='stage-sum' else c['total'] if attack=='extra-phase' else c['joint']['total'] if attack=='joint-end' else c['local']['stage1']
                if attack=='refinement':target['refinement']=1
                else:target['probes']+=1;target['complete_scores']+=1
                if attack=='extra-phase':r['budget_contract']['used']=c['total']['probes']
                with self.subTest(mode=mode,attack=attack),self.assertRaises(ValueError):self.check((r,a),mode)
    def test_budget_incumbent_and_auxiliary_reordering_remain_valid(self):
        for mode in ('J','LI'):
            for changed in (False,True):
                r,a=fixture(changed=changed,mode=mode);local=a['search']['local']
                local['search_complete']=False;local['local_cost_changed_during_completion']=True
                local['stage2_status']='budget-exhausted-witness-retained'
                self.assertEqual(len(self.check((r,a),mode)),4)
    def test_typed_paid_sample_fallback_is_not_a_complete_local_witness(self):
        r,a=fixture(mode='LI');c=a['search'];c['local_status']='no-local-witness'
        c['local'].update(legal_return=False,failure_kind='search-budget',failure='retained paid full candidate')
        a['entries'][0]['local_incumbent']=False;c['network_variant']['pins']=c['network_variant']['pins'][1:]
        self.assertEqual(len(self.check((r,a),'LI')),4)
        for field,value in (('failure_kind','fatal-integrity'),('failure',''),('legal_return',True)):
            rr,aa=copy.deepcopy((r,a));aa['search']['local'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.check((rr,aa),'LI')
        rr,aa=copy.deepcopy((r,a));aa['search']['local_status']='invented-success'
        with self.assertRaises(ValueError):self.check((rr,aa),'LI')
    def test_actual_incumbent_and_joint_constructed_witness_are_retained(self):
        r,a=fixture();r['complete_program_sha256']='0'*64
        with self.assertRaises(ValueError):self.check((r,a))
        r,a=fixture();a['search']['joint_components']=[dict(has_seed_witness=True,
            seed_start_cost=dict(cost=[1]+[0]*8,saturated=False),
            selected_component_cost=dict(cost=[2]+[0]*8,saturated=False))]
        with self.assertRaises(ValueError):self.check((r,a))


if __name__=='__main__':unittest.main()
