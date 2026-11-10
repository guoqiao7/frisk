import unittest
import network_selection_analysis as a

class PoolDecompositionTest(unittest.TestCase):
    def test_bad_pool_and_bad_ranking_are_separate(self):
        group=dict(paired_batches=10,method=dict(bootstrap_seed=1,bootstrap_samples=1000),
            image_batch_medians_us={'fast':[1.]*10,'middle':[2.]*10,'slow':[4.]*10})
        pool=dict(modes={'joint':dict(arms=['j1','j2']),'local':dict(arms=['l1'])})
        aliases={'j1':dict(execution_identity='middle'),'j2':dict(execution_identity='slow'),'l1':dict(execution_identity='fast')}
        choices={'joint-main':dict(selected_source='j2'),'local-main':dict(selected_source='l1')}
        r=a.decompose(group,pool,aliases,choices)['modes']
        self.assertEqual(r['joint']['pool_coverage_batch_oracle_loss']['paired_median'],2.)
        self.assertEqual(r['joint']['selections']['joint-main']['within_pool_batch_oracle_loss']['paired_median'],2.)
        self.assertEqual(r['joint']['selections']['joint-main']['finite_union_batch_oracle_loss']['paired_median'],4.)
        self.assertEqual(r['local']['pool_coverage_batch_oracle_loss']['paired_median'],1.)
    def test_aliases_do_not_inflate_pool(self):
        group=dict(paired_batches=10,method=dict(bootstrap_seed=2,bootstrap_samples=1000),image_batch_medians_us={'one':[1.]*10})
        pool=dict(modes={'joint':dict(arms=['a','b']),'local':dict(arms=['c'])})
        aliases={x:dict(execution_identity='one') for x in 'abc'}
        r=a.decompose(group,pool,aliases,{'joint-main':dict(selected_source='a'),'local-main':dict(selected_source='c')})
        self.assertEqual(r['modes']['joint']['unique_images'],1)
    def test_escaped_or_missing_candidate_fails(self):
        group=dict(paired_batches=10,method=dict(bootstrap_seed=2,bootstrap_samples=1000),image_batch_medians_us={'one':[1.]*10})
        pool=dict(modes={'joint':dict(arms=['a']),'local':dict(arms=['c'])});aliases={x:dict(execution_identity='one') for x in 'abc'}
        with self.assertRaises(ValueError):a.decompose(group,pool,aliases,{'joint-main':dict(selected_source='b')})

if __name__=='__main__':unittest.main()
