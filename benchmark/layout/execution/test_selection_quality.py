import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import selection_quality as driver
import selection_quality_fixture as fixture
import support_closure as closure

class QualityDriverTest(unittest.TestCase):
    def test_execute_binds_explicit_registered_provider(self):
        with patch.object(closure,'execute',return_value={'passed':True}) as call:
            self.assertEqual(driver.execute(Path('/tmp/case'),True,True,'memcheck'),{'passed':True})
        call.assert_called_once_with(Path('/tmp/case'),load_only=True,safety=True,safety_tool='memcheck',numerical_fixture=fixture)
    def test_nonzero_worker_is_fatal(self):
        with patch.object(closure,'invoke',return_value={'exit_code':3}):
            with self.assertRaises(RuntimeError):driver.invoke(['worker'],Path('/tmp/log'))
    def test_empty_matrix_cannot_be_measured(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);driver.write(root/'manifest.json',{'status':'validated','rows':[]})
            with patch.object(driver,'verify_protocol',return_value=fixture.protocol()):
                with self.assertRaisesRegex(ValueError,'matrix incomplete'):driver.measurement_groups(root)
    def test_partial_validation_cannot_be_measured(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);driver.write(root/'manifest.json',{'status':'failed','rows':[]})
            with patch.object(driver,'verify_protocol',return_value=fixture.protocol()):
                with self.assertRaisesRegex(ValueError,'all rows'):driver.measurement_groups(root)
    def test_original_default_provider_remains_unchanged(self):
        self.assertIsNone(inspect.signature(closure.execute).parameters['numerical_fixture'].default)
    def test_json_nan_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):driver.write(Path(temp)/'bad.json',{'x':float('nan')})
if __name__=='__main__':unittest.main()
