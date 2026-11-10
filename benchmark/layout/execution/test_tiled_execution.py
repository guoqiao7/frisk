import copy
import hashlib
import unittest
from unittest.mock import patch
from tiled_execution import _lift_text, _contract, validate_tiled_launch


def config(count=3, strong=False):
    return dict(global_alignments=[128,128,128,1] if strong else [1,1,1,1],
        root_order=[3,4,5,6], tma_plans=[],
        tiled_execution=dict(version='independent-tile-cta-v1',tile_count=count,
            grid=[count,1,1],threads=128,root_bytes=[2048,2048,2048,4096],
            root_stride_bytes=[2176,2176,2176,4224],descriptor_stride_bytes=128,
            single_tile_tt_sha256='a'*64,
            semantics='independent fixed-shape tiles; no cross-CTA communication'))


def text(descriptors=0):
    body='{\\0A.reg .u64 p;\\0Aadd.u64 p, $1, 0;\\0Ast.global.u32 [$4], $0;\\0A}'
    return 'tt.func public @frisk_execution() { %r = tt.elementwise_inline_asm "'+body+'" {constraints = "=r'+',l'*(4+descriptors)+'"} }'


class TiledExecutionTests(unittest.TestCase):
    def test_cta_coordinate_and_distinct_root_strides(self):
        result=_lift_text(text(), [64,128,256,512],0)
        self.assertIn('mov.u32 frisk_tile_cta, %ctaid.x;',result)
        for operand,stride in enumerate([64,128,256,512],1):
            self.assertIn(f'mul.wide.u32 frisk_tile_base{operand}, frisk_tile_cta, {stride};',result)
            self.assertIn(f'add.u64 frisk_tile_base{operand}, ${operand}, frisk_tile_base{operand};',result)
        self.assertIn('st.global.u32 [frisk_tile_base4], $0;',result)
        self.assertIn('\\0A.reg .u64 p;',result)
    def test_descriptor_operands_are_separate_128_byte_arrays(self):
        source=text(7).replace('add.u64 p, $1, 0;', 'add.u64 p, $11, 0;')
        result=_lift_text(source,[64]*4,7)
        self.assertIn('mul.wide.u32 frisk_tile_base11, frisk_tile_cta, 128;',result)
        self.assertIn('add.u64 p, frisk_tile_base11, 0;',result)
        self.assertNotIn('frisk_tile_base11'+chr(0),result)
    def test_duplicate_or_unknown_envelopes_fail_closed(self):
        for value in [text()+text(),text().replace('=r,l,l,l,l','=r,r,l,l,l'),text().replace('$4','$8'),text().replace('\\0A}', '}'),text().replace('.reg .u64 p;','mov.u32 p, %ctaid.x;')]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):_lift_text(value,[64]*4,0)
    def test_repeated_lift_rejected(self):
        with self.assertRaises(ValueError):_lift_text(_lift_text(text(),[64]*4,0),[64]*4,0)
    def test_exact_contract(self):self.assertEqual(_contract(config())['grid'],[3,1,1])
    def test_allocation_requirement_includes_last_tile(self):
        info=config(); got=validate_tiled_launch(info,[0x10000,0x20000,0x30000,0x40000],[6400,6400,6400,12544],grid=[3,1,1])
        self.assertEqual(got['required_bytes'],[6400,6400,6400,12544])
    def test_short_last_tile_rejected(self):
        with self.assertRaises(ValueError):validate_tiled_launch(config(),[0x10000,0x20000,0x30000,0x40000],[6399,6400,6400,12544],grid=[3,1,1])
    def test_whole_output_range_cannot_alias_later_input_tile(self):
        with self.assertRaises(ValueError):validate_tiled_launch(config(),[0x10000,0x20000,0x30000,0x11800],[6400,6400,6400,12544],grid=[3,1,1])
    def test_read_only_input_alias_is_allowed(self):
        validate_tiled_launch(config(),[0x10000,0x10000,0x10000,0x40000],[6400,6400,6400,12544],grid=[3,1,1])
    def test_pointer_range_overflow_rejected(self):
        with self.assertRaises(ValueError):validate_tiled_launch(config(),[0x10000,0x20000,0x30000,(1<<64)-1024],[6400,6400,6400,12544],grid=[3,1,1])
    def test_strides_preserve_every_cta_strong_alignment(self):
        info=config(strong=True);info['tiled_execution']['root_stride_bytes'][0]=2050
        with self.assertRaises(ValueError):_contract(info)
    def test_span_overlap_rejected(self):
        info=config();info['tiled_execution']['root_stride_bytes'][3]=4095
        with self.assertRaises(ValueError):_contract(info)
    def test_grid_must_equal_compiled_contract(self):
        with self.assertRaises(ValueError):validate_tiled_launch(config(),[0x10000,0x20000,0x30000,0x40000],[20000]*4,grid=[2,1,1])
    def test_bool_and_noninteger_dimensions_are_rejected(self):
        for value in [True,0,-1,1.0,1<<31]:
            info=config();info['tiled_execution']['tile_count']=value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):_contract(info)
    def test_boolean_grid_coordinate_rejected(self):
        info=config(1)
        with self.assertRaises(ValueError):validate_tiled_launch(info,[0x10000,0x20000,0x30000,0x40000],[20000]*4,grid=[True,1,1])
        info['tiled_execution']['grid']=[True,1,1]
        with self.assertRaises(ValueError):_contract(info)
    def test_unknown_contract_fields_rejected(self):
        info=config();info['tiled_execution']['untrusted']=True
        with self.assertRaises(ValueError):_contract(info)
    def test_descriptors_stride_is_fixed(self):
        info=config();info['tiled_execution']['descriptor_stride_bytes']=64
        with self.assertRaises(ValueError):_contract(info)
    def test_malformed_source_identity_rejected(self):
        info=config();info['tiled_execution']['single_tile_tt_sha256']='no'
        with self.assertRaises(ValueError):_contract(info)

class RegressionEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.registration=dict(runtime_environment=dict(CUDA_VISIBLE_DEVICES=None,CUDA_MODULE_LOADING='LAZY',CUDA_DEVICE_ORDER=None,LD_LIBRARY_PATH=None),tools=dict(sanitizer_runtime=dict(directory='/frozen/sanitizer')))
    def test_cvd_unset_differs_from_empty_string(self):
        from tiled_execution_regression import _verify_environment
        environment=dict(self.registration['runtime_environment']);environment['CUDA_VISIBLE_DEVICES']=''
        with patch('tiled_execution_regression._runtime_environment',return_value=environment):
            with self.assertRaises(ValueError):_verify_environment(self.registration,None)
    def test_prior_execution_files_require_recorded_seal(self):
        import tempfile
        from pathlib import Path
        from tiled_execution_regression import _sealed_prior_file,sha
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'prior.json';path.write_text('original');entries={str(path.resolve()):sha(path)}
            _sealed_prior_file(path,entries);path.write_text('self-consistent new manifest')
            with self.assertRaises(ValueError):_sealed_prior_file(path,entries)
    def test_complete_config_comparison_preserves_json_numeric_types(self):
        from tiled_execution_regression import digest
        self.assertEqual(digest({'root':{1:4}}),digest({'root':{'1':4}}))
        self.assertNotEqual(digest({'x':1}),digest({'x':1.0}))
    def test_only_exact_sanitizer_prefix_is_allowed(self):
        from tiled_execution_regression import _verify_environment
        environment=dict(self.registration['runtime_environment']);environment['LD_LIBRARY_PATH']='/frozen/sanitizer'
        with patch('tiled_execution_regression._runtime_environment',return_value=environment):_verify_environment(self.registration,'memcheck')
        for path in ['/other','/frozen/sanitizer:','/frozen/sanitizer:/other']:
            environment['LD_LIBRARY_PATH']=path
            with patch('tiled_execution_regression._runtime_environment',return_value=environment):
                with self.assertRaises(ValueError):_verify_environment(self.registration,'memcheck')

if __name__=='__main__':unittest.main()
