import unittest,ast
from pathlib import Path
import numpy as np
class NumericalCasesTest(unittest.TestCase):
 def test_actual_runner_cancellation_is_along_reduction_axis(self):
  source=Path(__file__).with_name('run_execution.py').read_text();namespace={'np':np};exec(source[source.index('cases=[]'):source.index('records=[]')],namespace);name,q,k,v=next(case for case in namespace['cases'] if case[0]=='cancelV')
  self.assertTrue(np.all(v.astype(np.float32).sum(axis=0)==0),'V must cancel along K, independently per output column')
  self.assertTrue(np.all(q==0) and np.all(k==0),'uniform P must make cancellation exact')
  function=next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='reference');exec(compile(ast.Module(body=[function],type_ignores=[]),'reference','exec'),namespace);self.assertTrue(np.array_equal(namespace['reference'](q,k,v),np.zeros((64,16),np.float32)))
if __name__=='__main__':unittest.main()
