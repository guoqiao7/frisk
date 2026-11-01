import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from resource_prediction import (compiler_exact_resources, parse_compiler_resource_usage,
                                 ResourceEvidenceError, confirm_driver_resources)

TEXT="Resource usage:\n Function other:\n REG:7 STACK:0 SHARED:99 LOCAL:0\n Function frisk_execution:\n REG:168 STACK:0 SHARED:8192 LOCAL:0\n"


class ResourceFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.image=self.root/"image.cubin";self.image.write_bytes(b"\x7fELFunit-fixture")
        self.tool=self.root/"fake-cuobjdump";self.calls=self.root/"calls"
        self.tool.write_text("#!"+sys.executable+"\nfrom pathlib import Path\np=Path("+repr(str(self.calls))+")\np.write_text(p.read_text()+'x' if p.exists() else 'x')\nprint("+repr(TEXT)+")\n")
        self.tool.chmod(0o700);self.cache=self.root/"cache"
    def tearDown(self):self.temp.cleanup()
    def query(self,**kwargs):
        return compiler_exact_resources(self.image,tool_path=self.tool,cache_dir=self.cache,**kwargs)
    def test_exact_entry_not_first_function(self):
        row=parse_compiler_resource_usage(TEXT)
        self.assertEqual(row["registers_per_thread"],168);self.assertEqual(row["raw_shared_bytes"],8192)
        self.assertEqual(row["local_bytes_per_thread"],0);self.assertIsNone(row["spill_count"])
    def test_incomplete_duplicate_negative_float_and_overflow_rejected(self):
        for bad in (TEXT.replace("REG:168","REG:168 REG:80"),TEXT+" Function frisk_execution:\n REG:168 STACK:0 SHARED:0 LOCAL:0\n",
                    TEXT.replace("REG:168","REG:-1"),TEXT.replace("REG:168","REG:1.5"),
                    TEXT.replace("REG:168","REG:True"),TEXT.replace("REG:168","REG:"+str(2**70)),
                    TEXT.replace("REG:168",""),TEXT.replace("frisk_execution","different")):
            with self.subTest(bad=bad),self.assertRaises(ResourceEvidenceError):parse_compiler_resource_usage(bad)
    def test_stack_local_not_silently_called_zero_or_spill(self):
        row=parse_compiler_resource_usage(TEXT.replace("STACK:0 SHARED:8192 LOCAL:0","STACK:48 SHARED:8192 LOCAL:16"))
        self.assertTrue(row["needs_driver_local_query"]);self.assertIsNone(row["local_bytes_per_thread"])
        self.assertEqual(row["stack_bytes_per_thread"],48);self.assertEqual(row["raw_local_bytes"],16)
    def test_cache_hit_and_image_change(self):
        first=self.query();second=self.query()
        self.assertFalse(first["cache_hit"]);self.assertTrue(second["cache_hit"]);self.assertEqual(self.calls.read_text(),"x")
        self.image.write_bytes(self.image.read_bytes()+b"changed")
        third=self.query();self.assertFalse(third["cache_hit"]);self.assertNotEqual(first["cache_key"],third["cache_key"])
        self.assertEqual(self.calls.read_text(),"xx")
    def test_actual_tool_change_invalidates(self):
        first=self.query();self.tool.write_text(self.tool.read_text()+"# new build\n")
        second=self.query();self.assertFalse(second["cache_hit"]);self.assertNotEqual(first["cache_key"],second["cache_key"])
    def test_expected_image_identity_guard(self):
        with self.assertRaises(ResourceEvidenceError):self.query(expected_cubin_sha256="0"*64)
        self.assertFalse(self.calls.exists())
    def test_corrupt_or_plausibly_edited_cache_is_fatal(self):
        row=self.query();path=self.cache/(row["cache_key"]+".json");original=path.read_text()
        path.write_text("{")
        with self.assertRaises(ResourceEvidenceError):self.query()
        value=json.loads(original);value["payload"]["resources"]["registers_per_thread"]=64
        path.write_text(json.dumps(value))
        with self.assertRaises(ResourceEvidenceError):self.query()
        value=json.loads(original);value["payload"]["resources"]["raw_local_bytes"]=False
        value["payload_sha256"]=hashlib.sha256(json.dumps(value["payload"],sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
        path.write_text(json.dumps(value))
        with self.assertRaises(ResourceEvidenceError):self.query()
    def test_native_tool_failure_is_not_zero_resources(self):
        self.tool.write_text("#!"+sys.executable+"\nraise SystemExit(7)\n")
        with self.assertRaises(ResourceEvidenceError):self.query()
    def test_unknown_or_malformed_entry_rejected(self):
        with self.assertRaises(ResourceEvidenceError):self.query(entry="missing")
        for name in ("", "frisk execution", "x; exit", "x\ny"):
            with self.subTest(name=name),self.assertRaises(ValueError):self.query(entry=name)

    def test_query_rejects_original_image_changed_by_parallel_writer(self):
        self.tool.write_text("#!"+sys.executable+"\nfrom pathlib import Path\np=Path("+repr(str(self.image))+")\np.write_bytes(p.read_bytes()+b'changed')\nprint("+repr(TEXT)+")\n")
        with self.assertRaises(ResourceEvidenceError):self.query()
        self.assertFalse(list(self.cache.glob("*.json")))
    def test_query_rejects_snapshot_modified_by_tool(self):
        self.tool.write_text("#!"+sys.executable+"\nimport sys\nfrom pathlib import Path\nPath(sys.argv[-1]).write_bytes(b'bad')\nprint("+repr(TEXT)+")\n")
        with self.assertRaises(ResourceEvidenceError):self.query()
    def test_query_rejects_tool_change(self):
        self.tool.write_text("#!"+sys.executable+"\nimport sys\nfrom pathlib import Path\np=Path(sys.argv[0])\np.write_text(p.read_text()+'# changed\\n')\nprint("+repr(TEXT)+")\n")
        with self.assertRaises(ResourceEvidenceError):self.query()
    def test_cache_hit_rechecks_image_before_return(self):
        from unittest.mock import patch
        self.query()
        import resource_prediction as module
        original=module._check_resources
        def changed(value):
            original(value);self.image.write_bytes(self.image.read_bytes()+b'changed')
        with patch.object(module,"_check_resources",side_effect=changed),self.assertRaises(ResourceEvidenceError):self.query()
    def confirmation_fixture(self,shared=8192,raw=9216):
        sha=hashlib.sha256(self.image.read_bytes()).hexdigest()
        resolved=dict(kind="resolved-compiler-resources",registers_per_thread=168,shared_bytes=shared,
                      raw_native_shared_bytes=raw,local_bytes=0,
                      compiler_exact=dict(identity=dict(cubin_sha256=sha,entry="frisk_execution")))
        driver=dict(registers_per_thread=168,shared_bytes=shared,local_bytes=0,
                    device_limits=dict(reserved_shared_bytes_per_block=1024))
        return resolved,driver,sha
    def test_driver_confirmation_requires_loaded_image_and_entry(self):
        resolved,driver,sha=self.confirmation_fixture()
        for kwargs in ({},{"loaded_cubin_sha256":"0"*64,"loaded_entry":"frisk_execution"},
                       {"loaded_cubin_sha256":sha,"loaded_entry":"other"}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ResourceEvidenceError):confirm_driver_resources(resolved,driver,**kwargs)
        self.assertEqual(confirm_driver_resources(resolved,driver,loaded_cubin_sha256=sha,loaded_entry="frisk_execution")["status"],"confirmed")
    def test_raw_shared_reserve_and_zero_case(self):
        for shared,raw in ((8192,9216),(0,0)):
            resolved,driver,sha=self.confirmation_fixture(shared,raw)
            row=confirm_driver_resources(resolved,driver,loaded_cubin_sha256=sha,loaded_entry="frisk_execution")
            self.assertEqual(row["shared_bytes"],shared)
        resolved,driver,sha=self.confirmation_fixture(8192,8192)
        with self.assertRaises(ResourceEvidenceError):confirm_driver_resources(resolved,driver,loaded_cubin_sha256=sha,loaded_entry="frisk_execution")
    def test_nonzero_local_is_confirmed_by_driver_not_guessed(self):
        resolved,driver,sha=self.confirmation_fixture();resolved["local_bytes"]=None;driver["local_bytes"]=256
        row=confirm_driver_resources(resolved,driver,loaded_cubin_sha256=sha,loaded_entry="frisk_execution")
        self.assertEqual(row["local_bytes"],256);self.assertIsNone(row["spill_count"])


if __name__=="__main__":unittest.main()
