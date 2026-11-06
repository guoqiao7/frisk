import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import integrity_session as module


class IntegritySessionTest(unittest.TestCase):
    def test_first_and_exit_read_contents_while_repeats_reuse(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'input';path.write_bytes(b'payload')
            expected=hashlib.sha256(b'payload').hexdigest()
            with patch.object(module, '_hash', wraps=module._hash) as hashed:
                with module.VerificationSession() as session:
                    self.assertEqual(module.verified_sha(path,expected),expected)
                    for _ in range(5):self.assertEqual(module.verified_sha(path,expected),expected)
                    self.assertEqual(hashed.call_count,1)
                self.assertEqual(hashed.call_count,2)
                self.assertEqual(session.summary()['reused_content_checks'],5)

    def test_same_size_restored_mtime_mutation_fails(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'input';path.write_bytes(b'first');old=path.stat()
            with self.assertRaisesRegex(ValueError,'changed'):
                with module.VerificationSession():
                    module.verified_sha(path)
                    path.write_bytes(b'other');os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
                    module.verified_sha(path)
            self.assertIsNone(module.active())

    def test_exit_rehashes_even_with_unchanged_witness(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'input';path.write_bytes(b'input')
            with self.assertRaisesRegex(ValueError,'content changed'):
                with patch.object(module, '_hash', side_effect=['a'*64,'b'*64]):
                    with module.VerificationSession():module.verified_sha(path)

    def test_force_preserves_full_publication_check(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'input';path.write_bytes(b'input')
            with patch.object(module, '_hash', wraps=module._hash) as hashed:
                with module.VerificationSession():
                    module.verified_sha(path);module.verified_sha(path,force=True)
                    self.assertEqual(hashed.call_count,2)
                self.assertEqual(hashed.call_count,3)

    def test_nested_closed_and_changed_expected_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'input';path.write_bytes(b'input')
            with module.VerificationSession() as session:
                module.verified_sha(path)
                with self.assertRaises(ValueError):module.verified_sha(path,'f'*64)
                with self.assertRaises(ValueError):
                    with module.VerificationSession():pass
            with self.assertRaises(ValueError):session.sha(path)

    def test_symlink_and_replaced_inode_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'input';path.write_bytes(b'input');alias=root/'alias';alias.symlink_to(path)
            with self.assertRaises(ValueError):module.verified_sha(alias)
            with self.assertRaisesRegex(ValueError,'changed'):
                with module.VerificationSession():
                    module.verified_sha(path)
                    replacement=root/'replacement';replacement.write_bytes(b'input');replacement.replace(path)
                    module.verified_sha(path)

    def test_unenrolled_mutable_output_keeps_plain_sha_behavior(self):
        import support_closure as closure
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'output';path.write_bytes(b'one')
            with module.VerificationSession() as session:
                one=closure.sha(path);path.write_bytes(b'two');two=closure.sha(path)
                self.assertNotEqual(one,two);self.assertEqual(session.files,{})

    def test_source_identity_and_inventory_remain_exact(self):
        import cost_calibration as calibration
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'benchmark').mkdir();(root/'benchmark/a.py').write_text('a=1\n')
            with patch.object(calibration,'ROOT',root):
                before=calibration.source_identity()
                with self.assertRaisesRegex(ValueError,'inventory changed'):
                    with module.VerificationSession():
                        self.assertEqual(calibration.source_identity(),before)
                        (root/'benchmark/b.py').write_text('b=2\n')
                        calibration.source_identity()

    def test_native_forced_source_tool_and_provider_checks_are_not_cached(self):
        import native_compile_cache as native
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'payload.py';path.write_text('immutable')
            expected=module.verified_sha(path)
            with patch.object(native,'ROOT',Path(name)),patch.object(native,'_production_payload_paths',return_value=[str(path)]):
                with module.VerificationSession() as session:
                    native._source_check(dict(source_manifest={'payload.py':expected},source_digest=native._digest({'payload.py':expected})))
                    native._source_check(dict(source_manifest={'payload.py':expected},source_digest=native._digest({'payload.py':expected})))
                    self.assertEqual(session.content_hashes,1)
                    native._source_check(dict(source_manifest={'payload.py':expected},source_digest=native._digest({'payload.py':expected})),force=True)
                    self.assertEqual(session.content_hashes,2)
                    native._tool_check(dict(hashes={str(path):expected}),force=True)
                    self.assertEqual(session.content_hashes,3)
                    first=native._production_provider_identity()
                    self.assertEqual(session.content_hashes,3)
                    self.assertEqual(native._production_provider_identity(force=True),first)
                    self.assertEqual(session.content_hashes,4)
                self.assertEqual(session.content_hashes,5)

    def test_native_provider_payload_addition_invalidates_scope(self):
        import native_compile_cache as native
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'a.py').write_text('a=1')
            with patch.object(native,'_production_payload_paths',side_effect=lambda:sorted(str(p) for p in root.glob('*.py'))):
                with self.assertRaisesRegex(ValueError,'inventory changed'):
                    with module.VerificationSession():
                        native._production_provider_identity()
                        (root/'b.py').write_text('b=1')
                        native._production_provider_identity()


if __name__ == '__main__':unittest.main()
