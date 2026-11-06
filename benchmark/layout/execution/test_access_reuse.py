"""Algebraic image/coset locality checks against independent bounded enumeration."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import access_reuse as reuse
from test_storage_addresses import fixture


def explicit_counts(maps, width, block, scope):
    width = min(width, len(maps)); total = potential = 0
    for first in range(0, len(maps), width):
        group = maps[first:first+width]
        for warp in range(4) if scope == "warp" else range(1):
            union, summed = set(), 0
            for base, columns in group:
                touched = set()
                for t in range(warp * 32, warp * 32 + 32) if scope == "warp" else range(128):
                    addr = base
                    for i, c in enumerate(columns):
                        if t >> i & 1: addr ^= c
                    touched.update(range(addr // block, (addr + 3) // block + 1))
                union.update(touched); summed += len(touched)
            total += len(union); potential += summed - len(union)
    return total, potential


def packet_and_record():
    packet = fixture()
    for v in packet["values"]:
        if "distributed" in v:
            v["distributed"]["shape"] = [2, 64]; v["type"] = "tensor<2x64xf32>"
            v["layout_variable_ids"] = [v["id"]-4]
    record = {"surviving_coverage": {"proof_status":"Proven", "work_basis":"whole-ssa-survival-with-emitted-communication-roots-v1", "native_exact":False,
              "surviving":{"values":[{"variable":i,"queries":[{"terms":[0]*8}]} for i in (0,1)],"boundaries":[]}}}
    return packet, record


class AlgebraTest(unittest.TestCase):
    def compare(self, maps):
        for b in reuse.GRANULES:
            for w in reuse.WINDOWS:
                for scope in reuse.SCOPES:
                    actual = reuse.count_windows(maps,w,b,scope)
                    self.assertEqual((actual["footprint"],actual["potential"]),explicit_counts(maps,w,b,scope), (b,w,scope,maps))

    def test_random_affine_maps_with_dependent_and_zero_columns(self):
        rng = random.Random(61318)
        for _ in range(12):
            columns = [rng.randrange(1<<15)<<2 for _ in range(7)]
            columns[rng.randrange(7)] = 0
            qcols = [rng.randrange(1<<15)<<2 for _ in range(3)]
            maps = [(0x48 ^ reuse._apply(qcols,q),tuple(columns)) for q in range(8)]
            self.compare(maps)
            self.assertEqual(reuse.count_windows(maps,4,128,"cta")["projected_tags_enumerated"],0)

    def test_nonaffine_query_constants_use_exact_quotient_cosets(self):
        columns = (4,8,16,32,64,128,256)
        maps = [(x,columns) for x in (0,128,8,1024,512)]
        self.compare(maps)
        self.assertIn("common-image-quotient-cosets",reuse.count_windows(maps,4,128,"cta")["methods"])

    def test_varying_thread_images_use_bounded_tag_union(self):
        maps = [(0,(4,8,16,32,64,128,256)), (512,(4,4,0,8,16,32,64)), (1024,(0,0,0,0,0,0,0))]
        self.compare(maps)
        self.assertIn("bounded-projected-image-tag-union",reuse.count_windows(maps,4,32,"cta")["methods"])

    def test_cta_union_keeps_cross_warp_coset_overlap(self):
        # Every warp has the same 128-byte image when its two columns live
        # inside the lane image; four independent warp counts cannot replace it.
        maps = [(0,(4,8,16,32,64,4,8))]
        warp = reuse.count_windows(maps,1,128,"warp")["footprint"]
        cta = reuse.count_windows(maps,1,128,"cta")["footprint"]
        self.assertEqual(warp,4); self.assertEqual(cta,1)

    def test_pages_capture_warp_partition_information_not_in_warp_sector_sum(self):
        close = [(0,(16,32,64,128,256,4,8))]
        spread = [(0,(4,8,16,32,64,65536,131072))]
        self.assertEqual(reuse.count_windows(close,1,128,"cta")["footprint"],reuse.count_windows(spread,1,128,"cta")["footprint"])
        self.assertLess(reuse.count_windows(close,1,4096,"cta")["footprint"],reuse.count_windows(spread,1,4096,"cta")["footprint"])

    def test_quotient_representatives_are_basis_order_invariant(self):
        cols = [0b1100,0b1010,0b0110,0]
        for x in range(16):
            self.assertEqual(reuse.reduce_coset(x,reuse.basis(cols)),reuse.reduce_coset(x,reuse.basis(list(reversed(cols)))))
        self.assertEqual(len(reuse.basis(cols)),2)

    def test_invalid_alignment_and_enumeration_budget_fail_closed(self):
        for maps in [[(1,(4,8,16,32,64,128,256))], [(0,(4,8))], [(0,(True,8,16,32,64,128,256))]]:
            with self.assertRaises(ValueError): reuse.count_windows(maps,1,32,"cta")
        maps=[(0,(32,64,128,256,512,1024,2048)),(0,(4,8,16,32,64,128,256))]
        with mock.patch.object(reuse,"MAX_ENUMERATED_BLOCK_TAGS",0),self.assertRaisesRegex(ValueError,"enumeration-budget"):
            reuse.count_windows(maps,2,32,"cta")


class DomainTest(unittest.TestCase):
    def analyze(self, packet=None, record=None):
        p,r = packet_and_record()
        return reuse.analyze_packet(packet or p,record or r,ordered_queries={("read",4):[(0,)*8],("write",5):[(0,)*8]},retained_ssa={4,5})

    def test_strict_compact_and_explicit_maps(self):
        r = self.analyze();self.assertEqual(r["status"],"complete",r)
        self.assertEqual(len(r["features"]),72);self.assertEqual(len(reuse.FOOTPRINT_FEATURES),36)
        self.assertEqual(r["features"]["gf2_reuse_cta_b4096_w1_footprint"],2)
        self.assertIn("no weak-ABI runtime base alignment assumed",r["scope"]["address_origin"])
        self.assertEqual(r["features"]["gf2_reuse_cta_b4096_w32_potential"],0)

    def test_nondyadic_noncontiguous_attention_or_unproved_inputs_are_unsupported(self):
        p,r=packet_and_record()
        cases=[]
        q=copy.deepcopy(p);q["values"][4]["distributed"]["logical"][3],q["values"][4]["distributed"]["logical"][5]=5,3;cases.append(q)
        q=copy.deepcopy(p);q["values"][0]["storage"]["addresses_compact"]["base_bit"]=32;cases.append(q)
        q=copy.deepcopy(p);q["values"][4]["distributed"]["shape"]=[3,64];cases.append(q)
        q=copy.deepcopy(p);q["operations"].insert(0,{"name":"frisk.mma","attrs":{},"operands":[],"results":[]});cases.append(q)
        q=copy.deepcopy(p);q["values"][5]["distributed"]["first"][0]=1;cases.append(q)
        for q in cases:
            result=self.analyze(q,r);self.assertEqual(result["status"],"unsupported",result);self.assertIsNone(result["features"])
        r["surviving_coverage"]["proof_status"]="Unknown"
        self.assertEqual(self.analyze(p,r)["status"],"unsupported")


class ReceiptTest(unittest.TestCase):
    def case(self, directory):
        root=Path(directory);p,r=packet_and_record();emitter=reuse._TraceEmitter(p);text,cfg=emitter.run()
        files={"execution.json":json.dumps(p).encode(),"record.json":json.dumps(r).encode(),"kernel.ttgir":text.encode(),"kernel.cubin":b"explicit CPU fake compiled-image token", "emitted-config.json":json.dumps(cfg).encode(),"execution_config.json":json.dumps(cfg).encode(),"surviving-coverage-audit.json":json.dumps(dict(status="passed",native_exact=False,retained_ssa=[4,5])).encode()}
        for name,data in files.items():(root/name).write_bytes(data)
        sha=lambda name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        (root/"image-identity.json").write_text(json.dumps({n:sha(n) for n in ("execution.json","kernel.ttgir","kernel.cubin","execution_config.json")}))
        (root/"compile.json").write_text(json.dumps(dict(execution_sha256=sha("execution.json"),cubin_sha256=sha("kernel.cubin"))))
        return root

    def test_exact_full_emission_receipt_and_payload_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.case(directory);a=reuse.summarize(root);b=reuse.summarize(root)
            self.assertEqual(a["status"],"complete",a);self.assertEqual(a,b)
            value=dict(a);digest=value.pop("summary_sha256")
            self.assertEqual(digest,hashlib.sha256(reuse._canonical(value)).hexdigest())
            self.assertFalse(a["emission_proof"]["native_order_proven"])
            self.assertEqual(len(a["provenance"]["dependencies"]),9)

    def test_reemission_and_query_ledger_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.case(directory);(root/"kernel.ttgir").write_text("changed TTGIR")
            identity=json.loads((root/"image-identity.json").read_text());identity["kernel.ttgir"]=hashlib.sha256((root/"kernel.ttgir").read_bytes()).hexdigest();(root/"image-identity.json").write_text(json.dumps(identity))
            self.assertIn("full-reemitted-ttgir-mismatch",reuse.summarize(root)["reasons"])
        with tempfile.TemporaryDirectory() as directory:
            root=self.case(directory);r=json.loads((root/"record.json").read_text());r["surviving_coverage"]["surviving"]["values"][0]["queries"]=[];(root/"record.json").write_text(json.dumps(r))
            result=reuse.summarize(root);self.assertEqual(result["status"],"unsupported");self.assertIn("actual affine demand query mismatch",result["reasons"][0])

    def test_cfg_and_post_analysis_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.case(directory);cfg=json.loads((root/"emitted-config.json").read_text());cfg["extra"]=0;(root/"emitted-config.json").write_text(json.dumps(cfg))
            self.assertIn("full-reemitted-cfg-mismatch",reuse.summarize(root)["reasons"])
        with tempfile.TemporaryDirectory() as directory:
            root=self.case(directory);original=reuse.analyze_packet
            def mutate(*args,**kwargs):
                out=original(*args,**kwargs);(root/"kernel.cubin").write_bytes(b"changed");return out
            with mock.patch.object(reuse,"analyze_packet",side_effect=mutate):result=reuse.summarize(root)
            self.assertIn("input-changed-during-analysis:kernel.cubin",result["reasons"])


if __name__ == "__main__":unittest.main()
