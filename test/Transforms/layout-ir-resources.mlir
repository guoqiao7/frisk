// RUN: frisk-opt %s '-frisk-layout-pipeline=stats-file=%t.off.json' -o %t.off.mlir
// RUN: frisk-opt %s '-frisk-layout-pipeline=ir-diagnostics=true stats-file=%t.on.json' -o %t.on.mlir
// RUN: diff %t.off.mlir %t.on.mlir
// RUN: python3 -c "import json,sys; off,on=[json.load(open(p)) for p in sys.argv[1:]]; assert off['actual_ir_diagnostics'] is None; assert on['status']=='committed'; d=on['actual_ir_diagnostics']; assert d['arithmetic']['status']=='available'; r=d['resources']; assert r['status']=='unavailable'; s=r['scopes'][0]; assert s['status']=='unavailable'; assert all(s[k] is None for k in ('logical_live_carrier_slots_per_thread','logical_live_carrier_bytes_per_thread','logical_live_shared_root_bytes','declared_shared_root_bytes')); assert s['unavailable_operations'][0]['opcode']=='scf.for'; assert s['unavailable_operations'][0]['reason']=='unsupported-control-flow'" %t.off.json %t.on.json

func.func @control(%x: f32, %condition: i1) -> f32 {
  %lo = arith.constant 0 : index
  %hi = arith.constant 3 : index
  %step = arith.constant 1 : index
  %result = scf.for %i = %lo to %hi step %step iter_args(%v = %x) -> (f32) {
    %a = arith.mulf %v, %x : f32
    %next = scf.if %condition -> f32 {
      %b = arith.addf %a, %x : f32
      scf.yield %b : f32
    } else {
      scf.yield %a : f32
    }
    scf.yield %next : f32
  }
  return %result : f32
}
