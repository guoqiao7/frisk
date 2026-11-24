import unittest

from fa3_transport import FA3Transport, canonical, rank_transport_events, register_store_work


def contract(output="bulk_shared_to_global"):
    buf = dict(tile_bytes=32768, versions=2)
    moves = {role: dict(path="tma_tensor_4d", box=[64,128,1,1], boxes=2,
                        element_bytes=2, buffer=buf, generic_copy=dict(path="vector"))
             for role in ("q","k","v")}
    moves["o"] = dict(path=output, buffer=buf)
    plan = dict(kind="fa3_qk_next", buffers={"k":buf,"v":buf}, batch=8, wait_pending=0)
    return FA3Transport(canonical(dict(moves=moves,execution=plan)))


class TransportCostTest(unittest.TestCase):
    def test_causal_work_uses_native_path_and_valid_tiles(self):
        bound = contract()
        receipt = rank_transport_events(bound,896,896,True,2,heads=18)
        self.assertEqual(receipt["selected"][0], "same_tile")
        work = receipt["transport_work"]
        self.assertEqual([work[r]["issued_instructions"] for r in ("q","k","v","o")],
                         [252,1008,1008,126])
        self.assertEqual(work["k"]["requested_bytes"],504*32768)
        self.assertEqual(bound.fingerprint, receipt["transport_sha256"])

    def test_rectangular_tail_and_output_once(self):
        receipt = rank_transport_events(contract(),512,257,True,1,heads=3)
        work = receipt["transport_work"]
        self.assertEqual(work["v"]["tiles"], 3*(1+2+3+3))
        self.assertEqual(work["o"]["tiles"],12)

    def test_incompatible_profile_does_not_receive_tma_timings(self):
        with self.assertRaisesRegex(ValueError,"no calibrated"):
            rank_transport_events(contract("register_store_u32"),896,896,True,1,heads=18)
        facts=contract().facts
        facts["moves"]["k"]["path"]="vector"
        with self.assertRaisesRegex(ValueError,"no calibrated"):
            rank_transport_events(FA3Transport(canonical(facts)),896,896,True,1,heads=18)

    def test_expect_tx_matches_issued_boxes(self):
        facts=contract().facts
        facts["moves"]["v"]["boxes"]=1
        with self.assertRaisesRegex(ValueError,"transaction bytes"):
            rank_transport_events(FA3Transport(canonical(facts)),384,384,False,1,heads=16)


    def test_register_service_replaces_bulk_once_per_cta(self):
        cost = dict(bytes=32768,warp_stores=256,sector_requests=2048,width_bytes=4,per_lane_stores=32)
        facts=contract("register_store_u32").facts
        facts["moves"]["o"].update(store_work=cost,buffer=None)
        register=FA3Transport(canonical(facts))
        result=rank_transport_events(register,896,896,True,1,heads=18)
        self.assertEqual(result["selected"],["qk_next"])
        self.assertEqual(result["transport_work"]["o"]["issued_instructions"],126*256)
        other=rank_transport_events(register,896,2048,False,1,heads=18)
        self.assertEqual(other["transport_work"]["o"],result["transport_work"]["o"])

    def test_causal_partial_output_accounts_for_guarded_rows(self):
        facts=contract("register_store_u32").facts
        facts["moves"]["o"].update(
            store_work=dict(bytes=32768,warp_stores=256,sector_requests=2048,width_bytes=4),
            tail_store_work=dict(bytes=256,warp_stores=16,sector_requests=16))
        result=rank_transport_events(FA3Transport(canonical(facts)),257,600,True,1,heads=2)
        self.assertEqual(result["transport_work"]["o"]["requested_bytes"],257*2*256)
        self.assertEqual(result["transport_work"]["o"]["issued_instructions"],2*(2*256+16))
        with self.assertRaises(ValueError):
            rank_transport_events(contract(),257,600,True,1,heads=2)

    def test_store_work_uses_actual_lane_addresses(self):
        self.assertEqual(register_store_work({"logical":list(range(64))},32)["sector_requests"],4)
        tail = register_store_work({"logical":list(range(64))},32,8)
        self.assertEqual((tail["warp_stores"],tail["sector_requests"],tail["bytes"]),(1,1,16))
        with self.assertRaisesRegex(ValueError,"holder map"):
            register_store_work({"logical":list(range(63,-1,-1))},32)

    def test_joint_writeback_budget_requires_one_layout(self):
        from native_fa3_execution import rank_fa3_writebacks
        pair={}
        for writer,path in (("tma","bulk_shared_to_global"),("register","register_store_u32")):
            facts=contract(path).facts
            facts.update(execution=facts["execution"], anchors={},graph_sha256="graph",writeback=writer)
            facts["moves"]["o"]["holder_sha256"]="same-holders"
            if writer=="register":
                facts["moves"]["o"]["store_work"]=dict(bytes=32768,warp_stores=256,sector_requests=2048,width_bytes=4)
            pair[writer]=FA3Transport(canonical(facts))
        receipt,chosen=rank_fa3_writebacks(pair,896,896,True,1,heads=18)
        self.assertEqual(receipt["selected"],["register-qk_next"])
        self.assertEqual(len(chosen),1)
        facts=pair["register"].facts
        facts["moves"]["o"]["holder_sha256"]="other-holders"
        pair["register"]=FA3Transport(canonical(facts))
        with self.assertRaisesRegex(ValueError,"share a solved layout"):
            rank_fa3_writebacks(pair,896,896,True,2,heads=18)
