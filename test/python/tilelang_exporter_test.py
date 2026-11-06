"""Exporter trust-boundary tests; no TileLang installation required."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / "tools/layout-reference/export_tilelang_layouts.py"


class TileLangExporterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("tilelang_exporter", SCRIPT)
        cls.exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.exporter)

    def test_revision_rejected_before_import(self):
        with self.assertRaisesRegex(ValueError, "unsupported TileLang revision"):
            self.exporter.verify_checkout(Path("/does/not/exist"), "HEAD")

    def test_not_a_git_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                self.exporter.verify_checkout(Path(temporary), self.exporter.REVISION)

    def test_module_origin_must_be_inside_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(self.exporter.require_inside(root / "tilelang/__init__.py", root),
                             str(root / "tilelang/__init__.py"))
            with self.assertRaisesRegex(ValueError, "outside pinned"):
                self.exporter.require_inside(root.parent / "site-packages/tilelang/__init__.py", root)

    def test_module_origin_resolves_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "escape").symlink_to("/usr")
            with self.assertRaisesRegex(ValueError, "outside pinned"):
                self.exporter.require_inside(root / "escape/lib/libtilelang.so", root)

    def test_exact_python_dependency_versions(self):
        expected = dict(self.exporter.PYTHON_DEPENDENCIES)
        self.exporter.verify_python_versions(expected)
        expected["z3-solver"] = "different"
        with self.assertRaisesRegex(ValueError, "Python dependency revision"):
            self.exporter.verify_python_versions(expected)

    def test_failed_export_does_not_overwrite_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "corpus.json"
            output.write_text("existing audited corpus\n")
            process = subprocess.run([sys.executable, str(SCRIPT), "--tilelang", temporary,
                                      "--revision", "wrong", "--output", str(output)],
                                     capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("unsupported TileLang revision", process.stderr)
            self.assertEqual(output.read_text(), "existing audited corpus\n")

    def test_canonical_json_stable_and_no_nan(self):
        encode = self.exporter.canonical_json
        self.assertEqual(encode({"b": [1, 2], "a": 3}), encode({"a": 3, "b": [1, 2]}))
        with self.assertRaises(ValueError):
            encode({"bad": float("nan")})
        self.assertEqual(json.loads(encode({"a": 3})), {"a": 3})

    def test_static_corpus_tracks_exporter_source(self):
        corpus = SCRIPT.parents[2] / "test/Dialect/Frisk/layout/Inputs/tilelang_layout_corpus.json"
        data = json.loads(corpus.read_text())
        self.assertEqual(data["exporter_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
        self.assertEqual(data["tilelang_commit"], self.exporter.REVISION)
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(len(data["cases"]), 23)

    def run_parent_with_child_artifacts(self, output, audit, after_child=None):
        # Only the native dependency boundary is replaced: exercise the real
        # parent, paths, writes, staging and renames against a temporary FS.
        def child(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_text(
                '{"schema_version":2,"cases":[{"name":"new"}]}\n')
            Path(command[command.index("--audit") + 1]).write_text('{"audit":"new"}\n')
            if after_child:
                after_child()
        argv = [str(SCRIPT), "--tilelang", str(output.parent), "--revision",
                self.exporter.REVISION, "--output", str(output), "--audit", str(audit)]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(self.exporter, "verify_checkout", return_value={}), \
                mock.patch.object(self.exporter.subprocess, "run", side_effect=child):
            self.exporter.main()

    def test_publication_failure_preserves_previous_corpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "corpus.json"
            locked = root / "locked"
            locked.mkdir()
            locked.chmod(0o500)
            try:
                for audit in (root / "missing/audit.json", root, locked / "audit.json"):
                    with self.subTest(audit=audit):
                        if audit.parent == locked and os.access(locked, os.W_OK):
                            self.skipTest("process can write a mode-0500 directory")
                        output.write_text("previous audited corpus\n")
                        with self.assertRaises((OSError, ValueError)):
                            self.run_parent_with_child_artifacts(output, audit)
                        self.assertEqual(output.read_text(), "previous audited corpus\n")
            finally:
                locked.chmod(0o700)

    def test_publication_rejects_destination_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "corpus.json"
            output.write_text("previous audited corpus\n")
            symlink = root / "audit-symlink.json"
            symlink.symlink_to(output)
            hardlink = root / "audit-hardlink.json"
            os.link(output, hardlink)
            parent_alias = root / "alias"
            parent_alias.symlink_to(root, target_is_directory=True)
            for audit in (output, symlink, hardlink, parent_alias / "corpus.json"):
                with self.subTest(audit=audit):
                    output.write_text("previous audited corpus\n")
                    with self.assertRaisesRegex(ValueError, "distinct"):
                        self.run_parent_with_child_artifacts(output, audit)
                    self.assertEqual(output.read_text(), "previous audited corpus\n")

    def test_destination_failure_after_child_preserves_corpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, audit = root / "corpus.json", root / "audit.json"
            output.write_text("previous audited corpus\n")
            with self.assertRaises((OSError, ValueError)):
                self.run_parent_with_child_artifacts(output, audit, after_child=audit.mkdir)
            self.assertEqual(output.read_text(), "previous audited corpus\n")

    def test_successful_publication_keeps_both_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, audit = root / "corpus.json", root / "audit.json"
            output.write_text("previous audited corpus\n")
            self.run_parent_with_child_artifacts(output, audit)
            self.assertEqual(json.loads(output.read_text())["schema_version"], 2)
            self.assertEqual(json.loads(audit.read_text()), {"audit": "new"})


if __name__ == "__main__":
    unittest.main()
