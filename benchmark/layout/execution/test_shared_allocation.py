import copy
import unittest
from unittest.mock import patch
from shared_allocation import plan_shared_allocation, reserve_shared


def storage(identity, size, alignment=1, space="Shared"):
    return dict(id=identity, type="memref<"+str(size)+"xi8>", storage=dict(
        root=identity, alignment=alignment,
        layout="#frisk.storage<memory_space "+space+">",
        addresses=[dict(begin_bit=0, end_bit=size*8, view=[0], root=[0])]))


def declaration(identity):
    return dict(name="frisk.storage_contract", operands=[identity], results=[], attrs={})


def simple(*values):
    return dict(schema_version=1, values=list(values), operations=[declaration(x["id"]) for x in values])


def tma_fixture(copies=1, trailing_bytes=0):
    values=[]
    for identity,space,alignment in [(0,"Global",128),(1,"Shared",256)]:
        addresses=[dict(begin_bit=(i*16+j)*16,end_bit=(i*16+j+1)*16,view=[i,j],root=[i,j])
                   for i in range(64) for j in range(16)]
        values.append(dict(id=identity,type="memref<64x16xf16,1>",storage=dict(
            root=identity,alignment=alignment,layout="#frisk.storage<memory_space "+space+">",addresses=addresses)))
    result=simple(*values)
    contract=dict(version=1,path="tma",target="sm_90a",vector_bytes=16,valid_shape=[64,16],guarded=0,
                  issuer="elected-cta-thread",completion="mbarrier-and-async-proxy",
                  descriptor=dict(box=[16,64],global_strides=[32],source_offset=0,destination_offset=0,
                                  swizzle_bytes=0,interleave="none"))
    result["operations"] += [dict(name="frisk.copy",operands=[0,1],results=[],
                                  attrs={"frisk.copy_contract":copy.deepcopy(contract),"frisk.writer_policy":"first_owner"})
                             for _ in range(copies)]
    if trailing_bytes:
        result["values"].append(storage(2,trailing_bytes));result["operations"].append(declaration(2))
    return result


class SharedAllocationTest(unittest.TestCase):
    def test_no_allocation_has_no_fictitious_intercept_or_padding(self):
        plan=plan_shared_allocation(dict(schema_version=1,values=[],operations=[]))
        self.assertEqual(plan["physical_shared_bytes"],0)
        self.assertEqual(plan["static_shared_bytes"],0)
        self.assertIsNone(plan["tma_barrier_offset"])
    def test_declaration_footprint_includes_holes_and_alignment(self):
        data=simple(storage(0,3),storage(1,5,16))
        plan=plan_shared_allocation(data)
        self.assertEqual([x["offset"] for x in plan["allocations"]],[0,16])
        self.assertEqual(plan["root_payload_bytes"],8)
        self.assertEqual(plan["root_alignment_padding_bytes"],13)
        self.assertEqual(plan["physical_shared_bytes"],21)
    def test_serial_scratch_uses_maximum_and_counts_alignment_once(self):
        data=simple(storage(0,3))
        data["operations"] += [dict(name="frisk.convert_layout",operands=[],results=[],attrs={}) for _ in range(3)]
        # The layout proof checker is an independent contract boundary; this
        # unit checks allocation reuse, while the 103-image input oracle checks
        # actual route proofs without mocking them.
        with patch("shared_allocation.validate_redistribution",side_effect=[
                dict(mode="Shared",scratch_bytes=64),dict(mode="Shared",scratch_bytes=128),
                dict(mode="Register",scratch_bytes=0)]):
            plan=plan_shared_allocation(data)
        self.assertEqual(plan["scratch_offset"],4)
        self.assertEqual(plan["scratch_alignment_padding_bytes"],1)
        self.assertEqual(plan["scratch_max_bytes"],128)
        self.assertEqual(plan["static_shared_bytes"],132)
        self.assertEqual(plan["route_counts"],dict(Register=1,Shuffle=0,Shared=2))

    def test_unused_declared_root_keeps_capacity(self):
        plan=plan_shared_allocation(simple(storage(0,4096)))
        self.assertEqual(plan["physical_shared_bytes"],4096)
        self.assertEqual(plan["allocations"][0]["lifetime"],"whole-kernel-declared")
    def test_alias_contracts_never_double_allocate(self):
        root=storage(0,8);alias=storage(1,4);alias["storage"]["root"]=0
        alias["storage"]["addresses"][0].update(begin_bit=16,end_bit=48)
        data=simple(root,alias);data["operations"].append(dict(name="frisk.tile_load",operands=[1],results=[],attrs={}))
        plan=plan_shared_allocation(data)
        self.assertEqual(len(plan["allocations"]),1)
        self.assertEqual(plan["physical_shared_bytes"],8)
    def test_view_beyond_root_is_rejected(self):
        root=storage(0,8);alias=storage(1,16);alias["storage"]["root"]=0
        data=simple(root,alias);data["operations"].append(dict(name="frisk.tile_load",operands=[1],results=[],attrs={}))
        with self.assertRaisesRegex(ValueError,"exceeds root"):plan_shared_allocation(data)
    def test_root_identity_and_undeclared_endpoint_rejected(self):
        data=simple(storage(0,4));data["values"][0]["storage"]["root"]=9
        with self.assertRaises(ValueError):plan_shared_allocation(data)
        data=simple(storage(0,4));data["operations"]=[dict(name="frisk.tile_load",operands=[0],results=[],attrs={})]
        with self.assertRaisesRegex(ValueError,"no matching declared root"):plan_shared_allocation(data)
    def test_tma_barriers_are_distinct_from_tensor_roots(self):
        plan=plan_shared_allocation(tma_fixture(3))
        self.assertEqual(plan["shared_root_bytes"],2048)
        self.assertEqual(plan["tma_barrier_bytes"],24)
        self.assertEqual(plan["physical_shared_bytes"],2072)
        self.assertEqual([x["offset"] for x in plan["tma_barriers"]],[2048,2056,2064])
    def test_tma_alignment_gap_counts_towards_limit(self):
        data=tma_fixture(3,3);plan=plan_shared_allocation(data)
        self.assertEqual(plan["static_shared_bytes"],2051)
        self.assertEqual(plan["tma_barrier_offset"],2056)
        self.assertEqual(plan["tma_alignment_padding_bytes"],5)
        self.assertEqual(plan["physical_shared_bytes"],2080)
        data["shared_limit_bytes"]=2079
        with self.assertRaisesRegex(ValueError,"budget exceeded"):plan_shared_allocation(data)
        data["shared_limit_bytes"]=2080
        self.assertEqual(plan_shared_allocation(data)["physical_shared_bytes"],2080)
    def test_unknown_async_completion_and_overlap_are_not_serially_reused(self):
        data=tma_fixture();data["operations"][-1]["attrs"]["frisk.copy_contract"]["completion"]="overlapped-token"
        with self.assertRaises(ValueError):plan_shared_allocation(data)
        for ranges,threads in [([2],128),([1],256),([True],128)]:
            data=simple(storage(0,16));data["operations"].append(dict(name="frisk.parallel",operands=[],results=[],attrs=dict(ranges=ranges,threads=threads)))
            with self.assertRaises(ValueError):plan_shared_allocation(data)
        data=simple(storage(0,16));data["pipeline_stages"]=2
        with self.assertRaises(ValueError):plan_shared_allocation(data)
    def test_unknown_control_flow_does_not_get_static_zero(self):
        data=simple(storage(0,16));data["operations"].append(dict(name="scf.for",operands=[],results=[],attrs={}))
        with self.assertRaisesRegex(ValueError,"unsupported lowering"):plan_shared_allocation(data)
    def test_malformed_integer_and_overflow_are_fail_closed(self):
        for limit in [-1,True,2**64]:
            data=simple(storage(0,4));data["shared_limit_bytes"]=limit
            with self.assertRaises(ValueError):plan_shared_allocation(data)
        for root_identity in [True, -1, 2**64]:
            data=simple(storage(0,4));data["values"][0]["storage"]["root"]=root_identity
            with self.assertRaises(ValueError):plan_shared_allocation(data)
        for align in [0,3,True,2**64]:
            with self.assertRaises(ValueError):plan_shared_allocation(simple(storage(0,4,align)))
        for args in [(2**64-1,0,8,2**64-1),(0,16,4,15),(0,2**64,4,2**64-1)]:
            with self.assertRaises(ValueError):reserve_shared(*args)
    def test_duplicate_value_identity_and_empty_footprint_are_rejected(self):
        with self.assertRaises(ValueError):plan_shared_allocation(simple(storage(0,4),storage(0,4)))
        value=storage(0,4);value["storage"]["addresses"]=[]
        with self.assertRaises(ValueError):plan_shared_allocation(simple(value))


if __name__=="__main__":unittest.main()
