import copy
from collections import Counter,defaultdict
import inspect
import unittest
import abi_offset_probe as probe
import selection_quality as quality


def fixture():
    images={arm:dict(cubin_sha256=probe.CUBINS[arm],execution_config_sha256='a'*64,
        image_identity_sha256='b'*64,execution_identity='c'*64,physical_abi_sha256='d'*64) for arm in probe.ARMS}
    contents={str(seed):dict(input_hashes=[str(seed)*64]*3,reference_sha256=str(seed+3)*64) for seed in probe.SEEDS}
    spec=dict(key=probe.KEY,shape=[512,128],abi_stratum='weak-vector',threads=128,dtype='float32',tail='div6',case_id='no-conflict')
    rows=probe.schedule(images,contents)
    measurement=dict(batches=108,warmup=100,pilot_nodes=100,target_graph_ms=5,max_nodes=4096,min_nodes=1,replays=5,
        bootstrap_resamples=10000,seed=probe.RANDOM_SEED,foreign_pid_rule='invalidate whole paired block; retain and never replace',
        confidence='diagnostic paired block intervals; no promotion or holdout claim',
        batch_provider_contract=probe.BATCH_CONTRACT,batch_plan={probe.GROUP:rows})
    return dict(schema_version=1,contract=probe.CONTRACT,role='known-image-confounding-probe-not-fresh-qualification',
        spec=spec,images=images,input_contents=contents,random_seed=probe.RANDOM_SEED,schedule=rows,
        provenance={},measurement=measurement,statistical_unit='whole paired block; event samples are dependent repetitions')


class AbiOffsetProbeTest(unittest.TestCase):
    def test_original_default_input_and_offset_path_is_unchanged(self):
        self.assertIsNone(inspect.signature(quality.measure).parameters['batch_provider'].default)
        for b in range(10):
            for abi in ('weak-vector','strong-cpasync','strong-tma'):
                value=quality.measurement_batch_plan({'abi_stratum':abi},b)
                self.assertEqual(value,dict(input_batch=b,input_offset=1+b%3 if abi=='weak-vector' else 128*(b%3),output_offset=1+b%3,metadata={},arm_order=None))
        with self.assertRaises(TypeError):quality.measure(None,None,batch_provider=7)
    def test_crossed_factors_are_complete_and_order_balanced(self):
        plan=fixture();self.assertIs(probe.validate_plan(plan),plan);cells=defaultdict(list);by_repeat=defaultdict(set)
        for row in plan['schedule']:
            cells[row['metadata']['cell_id']].append(row);by_repeat[row['metadata']['replicate_block']].add(row['metadata']['cell_id'])
        self.assertEqual(len(cells),27);self.assertEqual({len(x) for x in by_repeat.values()},{27})
        for rows in cells.values():
            self.assertEqual(len(rows),4);self.assertEqual(Counter(tuple(row['arm_order']) for row in rows),{probe.ARMS:2,tuple(reversed(probe.ARMS)):2})
        self.assertEqual(plan['schedule'],probe.schedule(plan['images'],plan['input_contents']))
    def test_input_seed_is_independent_of_both_offsets(self):
        plan=fixture();by_seed=defaultdict(set);by_offsets=defaultdict(set);hashes=defaultdict(set)
        for row in plan['schedule']:
            seed=row['input_batch'];io,oo=row['input_offset'],row['output_offset'];by_seed[seed].add((io,oo));by_offsets[io,oo].add(seed);hashes[seed].add(tuple(row['metadata']['input_content']['input_hashes']))
        self.assertEqual({len(x) for x in by_seed.values()},{9});self.assertEqual({len(x) for x in by_offsets.values()},{3});self.assertEqual({len(x) for x in hashes.values()},{1})
    def test_provider_is_evaluated_once_per_registered_batch_and_cannot_drift(self):
        plan=fixture();groups={probe.GROUP:[dict(spec=plan['spec'],arm=arm) for arm in probe.ARMS]};calls=[];provided=probe.provider(plan)
        def tracked(spec,b):calls.append(b);return provided(spec,b)
        resolved=quality.measurement_batch_plans({'measurement':plan['measurement']},groups,tracked)
        self.assertEqual(calls,list(range(108)));self.assertEqual(resolved[probe.GROUP],plan['schedule'])
        changed=copy.deepcopy(plan['schedule'][0]);changed['input_offset']=3 if changed['input_offset']!=3 else 1
        def wrong(spec,b):return changed if b==0 else provided(spec,b)
        with self.assertRaises(ValueError):quality.measurement_batch_plans({'measurement':plan['measurement']},groups,wrong)
        with self.assertRaises(ValueError):quality.measurement_batch_plans({'measurement':plan['measurement']},groups)
        with self.assertRaises(ValueError):quality.measurement_batch_plans({'measurement':dict(batches=108)},groups,provided)
    def test_strict_batch_parameter_types_and_plan_budget(self):
        plan=fixture();valid=plan['schedule'][0]
        for field,value in [('input_batch',True),('input_batch',0.0),('input_offset',True),('input_offset',257),('output_offset',-1),('arm_order',['joint-c0','joint-c0']),('metadata',{'not_finite':float('nan')})]:
            row=copy.deepcopy(valid);row[field]=value
            with self.subTest(field=field,value=value),self.assertRaises((ValueError,TypeError)):quality.measurement_batch_plan(plan['spec'],0,lambda s,b:row)
        for mutation in ('count','order','factor'):
            changed=copy.deepcopy(plan)
            if mutation=='count':changed['measurement']['batches']=107
            elif mutation=='order':changed['schedule'][0]['arm_order'].reverse()
            else:changed['schedule'][0]['input_batch']=9
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):probe.validate_plan(changed)


if __name__=='__main__':unittest.main()
