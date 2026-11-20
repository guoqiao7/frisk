import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import predicate_cost_recalibration as calibration


class PredicateCalibrationTests(unittest.TestCase):
    def fixture(self,directory):
        directory=Path(directory)
        for name in ['execution.json','kernel.ttgir','kernel.cubin']:(directory/name).write_bytes(name.encode())
        banks=dict(contract='frisk-emitted-affine-shared-bank-v1',implementation_sha256={'old':'source'},
            execution_sha256=calibration.sha(directory/'execution.json'),emitted_ttgir_sha256=calibration.sha(directory/'kernel.ttgir'),
            native_cubin_sha256=calibration.sha(directory/'kernel.cubin'),shared_read_bank_excess=10,shared_write_bank_excess=20)
        old=dict(features=dict(shared_load_bytes=16,shared_store_bytes=32),symbolic_banks=banks)
        new=dict(status='available',features=dict(shared_load_bytes=16,shared_store_bytes=32))
        bridge=dict(contract='shared-only-proof',old_bank_implementation_sha256={'old':'source'},new_bank_implementation_sha256={'new':'source'})
        path=directory/'bridge.json';path.write_text('{}')
        return old,new,bridge,path
    def test_reattachment_changes_only_certified_shared_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            old,new,bridge,path=self.fixture(temp);saved=copy.deepcopy(old)
            result=calibration.reattach_known_shared(new,old,temp,bridge,path)
            self.assertEqual(result['symbolic_banks']['shared_read_bank_excess'],10)
            self.assertEqual(result['symbolic_banks']['shared_write_bank_excess'],20)
            self.assertFalse(result['symbolic_banks']['shared_semantics_bridge']['full_generated_image_equality_claim'])
            self.assertFalse(result['symbolic_banks']['shared_semantics_bridge']['numbers_changed'])
            self.assertEqual(old,saved)
    def test_native_shared_width_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            old,new,bridge,path=self.fixture(temp);new['features']['shared_load_bytes']=15
            with self.assertRaises(ValueError):calibration.reattach_known_shared(new,old,temp,bridge,path)
    def test_wrong_old_shared_code_domain_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            old,new,bridge,path=self.fixture(temp);bridge['old_bank_implementation_sha256']={'other':'source'}
            with self.assertRaises(ValueError):calibration.reattach_known_shared(new,old,temp,bridge,path)
    def test_changed_original_image_is_rejected(self):
        for name in ['execution.json','kernel.ttgir','kernel.cubin']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                old,new,bridge,path=self.fixture(temp);(Path(temp)/name).write_bytes(b'changed')
                with self.assertRaises(ValueError):calibration.reattach_known_shared(new,old,temp,bridge,path)
    def test_unsupported_native_cannot_get_shared_score(self):
        with tempfile.TemporaryDirectory() as temp:
            old,new,bridge,path=self.fixture(temp);new['status']='unsupported'
            with self.assertRaises(ValueError):calibration.reattach_known_shared(new,old,temp,bridge,path)
    def test_original_training_files_are_bound_to_recorded_index_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);labels=root/'labels.json';labels.write_text('[1]')
            completion=root/'completion.json';completion.write_text('{"labels":"old"}')
            index=root/'index.json';calibration.write(index,dict(repository_root=str(root),entries=[dict(path=p.name,sha256=calibration.sha(p)) for p in (labels,completion)]))
            with mock.patch.object(calibration,'ROOT',root),mock.patch.object(calibration,'INDICES',{index:calibration.sha(index)}):
                entries=calibration._trusted_entries();calibration._check_sealed_source(labels,entries)
                # Recomputing a self-consistent completion cannot change the seal.
                labels.write_text('[2]');completion.write_text('{"labels":"new"}')
                with self.assertRaises(ValueError):calibration._check_sealed_source(labels,entries)
                with self.assertRaises(ValueError):calibration._check_sealed_source(completion,entries)
    def test_dependencies_detect_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input';path.write_bytes(b'original');deps=calibration._dependencies([path])
            calibration._verify_dependencies(deps);path.write_bytes(b'changed')
            with self.assertRaises(ValueError):calibration._verify_dependencies(deps)
    def test_future_holdout_collection_is_not_an_input(self):
        with self.assertRaises(ValueError):calibration.refresh_collection('fresh-holdout',calibration.BASE/'unused-test-output',Path('/unread'))
    def test_new_profile_pair_requires_same_receipt_and_nonlocal_coefficients(self):
        paths={'native':{'profile':'local-envelope'},'old_native':{'profile':'native-envelope'}}
        envelopes={'local-envelope':dict(kind='local',receipt_path='same',receipt_sha256='a'),
                   'native-envelope':dict(kind='native',receipt_path='same',receipt_sha256='a')}
        weights={name:1 for name in calibration.FEATURES};maxima={name:10 for name in calibration.FEATURES}
        fixed=dict(features=list(calibration.FEATURES),coefficients=dict(weights,local_transfer_bytes=0),training_maxima=dict(maxima,local_transfer_bytes=0),analysis_contract={'domain':'new'})
        calibrated=dict(features=list(calibration.FEATURES),coefficients=dict(weights,local_transfer_bytes=123),training_maxima=maxima,analysis_contract={'domain':'new'})
        def load(path):return (calibrated if str(path)=='local-envelope' else fixed,{'/bound':'hash'})
        with mock.patch.object(calibration,'read',side_effect=lambda p:envelopes[str(p)]),mock.patch.object(calibration,'_load',side_effect=load):
            self.assertEqual(calibration.validate_configuration(paths)['local_weight'],123)
            calibrated['coefficients']['native_instructions']=2
            with self.assertRaises(ValueError):calibration.validate_configuration(paths)
            calibrated['coefficients']['native_instructions']=1
            envelopes['native-envelope']['receipt_path']='other'
            with self.assertRaises(ValueError):calibration.validate_configuration(paths)
    def test_old_sealed_roots_cannot_be_output(self):
        for directory in calibration.SOURCES.values():
            with self.assertRaises(ValueError):calibration.fresh(directory/'must-not-create')

if __name__=='__main__':unittest.main()
