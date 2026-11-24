import unittest

from fa3_query_tile import select_query_geometry, launch_features, rank_cta_dispatch, select_query_tile


class QueryTileTest(unittest.TestCase):
    def test_partial_grouped_query_uses_guarded_writer(self):
        from fa3_query_tile import select_query_geometry
        pick=select_query_geometry(1409,1409,16,False,guarded_query=True)
        self.assertTrue(all(c["eligible_writers"]==["register"] for c in pick["candidates"]))
        self.assertEqual(pick["cta_dispatch"],"forward")
        with self.assertRaises(ValueError):
            select_query_geometry(1409,1409,16,False,"tma",guarded_query=True)

    def test_reverse_feedback_uses_issued_cta_order(self):
        from fa3_query_tile import dispatch_query_tiles, launch_costs
        self.assertEqual(list(dispatch_query_tiles(3,2,True)),[2,2,1,1,0,0])
        self.assertEqual(list(dispatch_query_tiles(3,2,False)),[0,1,2,0,1,2])
        for rows,heads in ((1025,7),(1281,6),(1921,5)):
            costs=launch_costs(rows,rows,heads,True,64,"register",cta_dispatch="reverse")
            self.assertEqual(min(costs,key=costs.get),"late_k")
            forward=launch_costs(rows,rows,heads,True,64,"register",cta_dispatch="forward")
            self.assertEqual(min(forward,key=forward.get),"same_tile")
        with self.assertRaises(ValueError):
            launch_costs(1024,1024,8,False,64,"register",cta_dispatch="reverse")

    def test_two_adjacent_q64_ctas_share_one_diagonal_kv_tile(self):
        q64 = launch_features(256, 256, 1, True, 64, sm_count=4)
        q128 = launch_features(256, 256, 1, True, 128, sm_count=4)
        self.assertEqual(q64, (1.0, 1, 2, 1.5))
        self.assertEqual(q128, (1.0, 1, 2, 0.75))

    def test_rectangular_trip_count_caps_at_kv_length(self):
        features = launch_features(320, 129, 1, True, 64, sm_count=5)
        self.assertEqual(features, (1.0, 1, 2, 8 / 5))

    def test_tile_eligibility(self):
        self.assertEqual(select_query_tile(192, 256, 2, True), 64)
        self.assertIn(select_query_tile(193, 256, 2, True), (64,128))
        with self.assertRaises(ValueError):
            select_query_tile(193, 256, 2, True,"tma")

    def test_dispatch_residency_and_rectangular_plateau(self):
        # Two heads, four tiles/head, three resident slots.
        result = rank_cta_dispatch(512,512,2,True,128,sm_count=3)
        self.assertEqual(result["critical_service"], {"forward":18,"reverse":16})
        self.assertEqual(result["selected"],"reverse")
        for args in [(512,512,1,True,128,4),(512,128,4,True,128,3),
                     (512,512,4,False,128,3)]:
            *shape,sm=args
            self.assertEqual(rank_cta_dispatch(*shape,sm_count=sm)["selected"],"forward")

    def test_joint_pair_beats_each_single_change(self):
        choice=select_query_geometry(1921,1921,5,True)
        self.assertEqual((choice["query_tile"],choice["cta_dispatch"]),(64,"reverse"))
        self.assertEqual(len(choice["candidates"]),4)
        self.assertTrue(all(c["eligible_writers"]==["register"] for c in choice["candidates"]))
        fixed=select_query_geometry(1921,1921,5,True,cta_dispatch="forward")
        self.assertEqual((fixed["query_tile"],fixed["cta_dispatch"]),(128,"forward"))
        with self.assertRaises(ValueError):
            select_query_geometry(1921,1921,5,True,"tma")
        regular=select_query_geometry(256,256,2,False)
        self.assertEqual({c["cta_dispatch"] for c in regular["candidates"]},{"forward"})

    def test_register_and_tma_budget_contracts(self):
        self.assertEqual(select_query_tile(256, 256, 2, False), 64)
        self.assertEqual(select_query_tile(896, 896, 18, True), 128)
