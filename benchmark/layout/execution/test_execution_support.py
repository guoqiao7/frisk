"""Capability and compact-map regressions independent of GPU execution."""
import copy
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from execution_support import (assess, assess_file, normalize_packet, normalize_distribution,
                               SymbolicMap, LogicalTable)


def distribution(shape, topology, rows):
    hardware_widths=[x.bit_length()-1 for x in topology]
    logical_widths=[x.bit_length()-1 for x in shape]
    return dict(shape=shape,topology=topology,replication=math.prod(topology)//math.prod(shape),
                symbolic=dict(version="gf2-canonical-v1",hardware_bits=sum(hardware_widths),
                              logical_bits=sum(logical_widths),hardware_widths=hardware_widths,
                              logical_widths=logical_widths,rows_hex=[format(x,"x") for x in rows]))


def packet():
    addresses=[dict(view=[i],root=[i],begin_bit=32*i,end_bit=32*i+32) for i in range(128)]
    values=[dict(id=i,type="memref<128xf32, 1>",storage=dict(root=i,alignment=1,layout="memory_space Global",addresses=copy.deepcopy(addresses))) for i in range(4)]
    for i in (4,5):
        values.append(dict(id=i,type="tensor<128xf32>",distributed=dict(shape=[128],topology=[1,32,4,1,1],replication=1,logical=list(range(128)),first=list(range(128)))))
    def op(name,operands=(),results=(),**attrs):return dict(name=name,operands=list(operands),results=list(results),attrs=attrs)
    ops=[op("builtin.module",**{"frisk.target":"sm_90a"}),op("frisk.kernel")]
    ops += [op("frisk.storage_contract",[i]) for i in range(4)]
    ops += [op("frisk.parallel",ranges=[1],threads=128),op("frisk.tile_load",[0],[4]),
            op("arith.mulf",[4,4],[5]),op("frisk.tile_store",[5,3]),op("frisk.end"),op("frisk.end")]
    return dict(schema_version=1,values=values,operations=ops)


class CompactMapTests(unittest.TestCase):
    def test_nonzero_dependent_columns_choose_canonical_owner(self):
        # Columns [3,1,2], all nonzero and distinct; third is dependent.
        mapping=SymbolicMap(distribution([4],[2,2,2,1,1],[3,5]))
        self.assertEqual([mapping.logical(i) for i in range(8)],[0,3,1,2,2,1,3,0])
        self.assertEqual([mapping.first(i) for i in range(4)],[0,2,3,1])

    def test_random_mixed_maps_and_rowmajor_flatten_match_enumeration(self):
        rng=random.Random(83413)
        for trial in range(24):
            columns=[1<<i for i in range(5)]+[0]*4
            for _ in range(40):
                a,b=rng.sample(range(9),2)
                if rng.randrange(2):columns[a],columns[b]=columns[b],columns[a]
                else:columns[a]^=columns[b]
            rows=[sum(((column>>r)&1)<<c for c,column in enumerate(columns)) for r in range(5)]
            d=distribution([4,8],[4,32,4,1,1],rows);mapping=SymbolicMap(d)
            expected=[]
            for h in range(512):
                out=[sum(((h>>c)&1)*((columns[c]>>r)&1) for c in range(9))%2 for r in range(5)]
                axis0=out[0]+2*out[1];axis1=out[2]+2*out[3]+4*out[4]
                expected.append(axis0*8+axis1)
            self.assertEqual([mapping.logical(h) for h in range(512)],expected)
            self.assertEqual([mapping.first(slot) for slot in range(32)],[expected.index(slot) for slot in range(32)])
            for thread in range(128):
                for slot in set(expected[thread*4:thread*4+4]):
                    self.assertEqual(mapping.local_owner(slot,thread),expected[thread*4:thread*4+4].index(slot))

    def test_large_holder_table_is_lazy_and_source_snapshot_owned(self):
        d=distribution([64,64],[4096,32,4,1,1],[1<<i for i in range(12)])
        with patch.object(LogicalTable,"__iter__",side_effect=AssertionError("unexpected holder expansion")):
            normalized=normalize_distribution(d)
        table=normalized["logical"]
        self.assertIsInstance(table,LogicalTable);self.assertEqual(len(table),524288)
        self.assertEqual(len(normalized["first"]),4096)
        self.assertEqual(table[4095],4095);self.assertEqual(table[-1],4095)
        self.assertEqual(table[:3],[0,64,128])
        # Mutating caller-owned metadata must not mutate an already-normalized map.
        d["symbolic"]["rows_hex"][0]="0"
        self.assertEqual(table[1],64)

    def test_materialization_is_explicit_and_small_oracle_equivalent(self):
        d=distribution([4],[2,2,2,1,1],[3,5]);normalized=normalize_distribution(d,materialize=True)
        self.assertIsInstance(normalized["logical"],list)
        self.assertEqual(normalized["logical"],[0,3,1,2,2,1,3,0])
        raw=dict(schema_version=2,values=[dict(id=0,type="tensor<4xf32>",distributed=d)],operations=[])
        self.assertEqual(normalize_packet(raw)["schema_version"],1)
        self.assertIsInstance(normalize_packet(raw)["values"][0]["distributed"]["logical"],LogicalTable)

    def test_bad_rank_replication_width_and_schema_rejected(self):
        original=distribution([4],[2,2,2,1,1],[3,5])
        variants=[]
        d=copy.deepcopy(original);d["symbolic"]["rows_hex"][1]="0";variants.append(d)
        d=copy.deepcopy(original);d["replication"]=1;variants.append(d)
        d=copy.deepcopy(original);d["symbolic"]["hardware_widths"]=[3,0,0,0,0];variants.append(d)
        d=copy.deepcopy(original);d["symbolic"]["rows_hex"][0]="8";variants.append(d)
        d=copy.deepcopy(original);d["symbolic"]["future"]=True;variants.append(d)
        for d in variants:
            with self.assertRaises(ValueError):SymbolicMap(d)

    def test_unit_axes_and_zero_logical_bits(self):
        d=distribution([1,1],[1,32,4,1,1],[]);mapping=SymbolicMap(d)
        self.assertEqual(mapping.first(0),0)
        self.assertTrue(all(mapping.logical(h)==0 for h in range(128)))
        self.assertEqual(mapping.local_owner(0,17),0)


class SupportAssessmentTests(unittest.TestCase):
    def test_actual_emitter_acceptance_is_scoped_and_does_not_mutate_input(self):
        original=packet();saved=copy.deepcopy(original)
        result=assess(original,include_emission=True)
        self.assertEqual(result["status"],"supported",result)
        self.assertEqual(result["claim"],"checked-emission-only")
        self.assertIn("ld.global.u8",result["ttgir"])
        self.assertEqual(result["execution_config"]["emitted_operations"],len(original["operations"]))
        self.assertEqual(original,saved)
        self.assertEqual(assess(original)["packet_sha256"],result["packet_sha256"])
        self.assertNotIn("ttgir",assess(original))

    def test_static_shared_support_rejects_total_capacity_as_static_limit(self):
        for size, supported in ((32768, True), (49152, True), (65536, False)):
            with patch("backend.emit", return_value=("ttgir", dict(static_shared_bytes=size, shared_limit_bytes=227328))):
                result=assess(packet(), include_emission=True)
            self.assertEqual(result["supported"], supported, result)
            if supported:
                self.assertEqual(result["execution_config"]["static_shared_bytes"], size)
                self.assertIn("static_shared_domain", result)
            else:
                self.assertEqual(result["code"], "static-shared-limit")
                self.assertEqual(result["status"], "unsupported")
                self.assertNotIn("ttgir", result)

    def test_invalid_emitted_static_shared_is_not_a_supported_packet(self):
        with patch("backend.emit", return_value=("ttgir", dict(static_shared_bytes=True))):
            result=assess(packet())
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["code"], "invalid-emitted-config")

    def test_unsupported_operation_retains_actual_backend_reason(self):
        data=packet();data["operations"].append(dict(name="math.log",operands=[],results=[],attrs={}))
        result=assess(data)
        self.assertEqual(result["status"],"unsupported",result)
        self.assertEqual(result["stage"],"backend-emission")
        self.assertIn("unsupported lowering operation",result["reason"])
        self.assertFalse(result["supported"])

    def test_invalid_envelope_and_unused_first_owner_are_rejected(self):
        variants=[]
        data=packet();data["values"][4]["distributed"]["first"][0]=1;variants.append(data)
        data=packet();data["operations"][7]["operands"]=[999];variants.append(data)
        data=packet();data["values"][5]["id"]=4;variants.append(data)
        data=packet();data["operations"][0]["attrs"]["frisk.target"]="sm_80";variants.append(data)
        data=packet();data["schema_version"]=True;variants.append(data)
        for data in variants:
            result=assess(data);self.assertEqual(result["status"],"invalid",result);self.assertFalse(result["supported"])

    def test_nonfinite_json_and_unreadable_input_are_typed(self):
        data=packet();data["extra"]=float("nan")
        self.assertEqual(assess(data)["status"],"invalid")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.json";path.write_text("{")
            result=assess_file(path)
            self.assertEqual(result["code"],"unreadable-packet")


class CompactRedistributionTests(unittest.TestCase):
    def fixture(self, source_registers=1024):
        from redistribution import validate
        source=distribution([1024],[source_registers,32,4,1,1],[1<<i for i in range(10)])
        destination=dict(shape=[1024],topology=[8,32,4,1,1],replication=1,logical=list(range(1024)),first=list(range(1024)))
        def owner(reg,thread):return dict(reg=reg,lane=thread%32,warp=thread//32,warp_group=0)
        plan=dict(mode="Register",threads=128,warps=4,source_registers=source_registers,
                  destination_registers=8,volume=1024,scratch_bytes=0,
                  payload=dict(bits=32,words=[dict(bit_offset=0,live_bits=32)]),
                  source_slots_kind="source-distributed-symbolic-v1",
                  destinations=[dict(source=owner(slot,slot//8),slot=slot) for slot in range(1024)],
                  global_writers=[owner(slot,0) for slot in range(1024)])
        values={0:dict(id=0,type="tensor<1024xf32>",distributed=source),
                1:dict(id=1,type="tensor<1024xf32>",distributed=destination)}
        operation=dict(name="frisk.convert_layout",operands=[0],results=[1],attrs={},redistribution=plan)
        return operation,values

    def test_compact_register_validates_without_source_holder_iteration(self):
        from redistribution import validate
        operation,values=self.fixture()
        with patch.object(LogicalTable,"__iter__",side_effect=AssertionError("source enumeration")):
            self.assertIs(validate(operation,values),operation["redistribution"])

    def test_forged_global_local_and_route_plans_are_rejected(self):
        from redistribution import validate
        variants=[]
        operation,values=self.fixture();operation["redistribution"]["global_writers"][5]["lane"]=1;variants.append((operation,values))
        operation,values=self.fixture();operation["redistribution"]["destinations"][5]["source"]["lane"]^=1;variants.append((operation,values))
        operation,values=self.fixture();operation["redistribution"]["destinations"][5]["slot"]^=1;variants.append((operation,values))
        operation,values=self.fixture();operation["redistribution"]["mode"]="Shuffle";variants.append((operation,values))
        operation,values=self.fixture(2048);operation["redistribution"]["destinations"][5]["source"]["reg"]+=1024;variants.append((operation,values))
        for operation,values in variants:
            with self.assertRaises(ValueError):validate(operation,values)


if __name__=="__main__":unittest.main()
