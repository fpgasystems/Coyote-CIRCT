// REQUIRES: cocotb, iverilog

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=projection-selection --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0


!T = tuple<i64, i64, i64, i64, i64, i64, i64, i64>

module {
  func.func @top(%in: !stream.stream<!T>) -> !stream.stream<i64> {
   %mapOut = stream.map(%in) : (!stream.stream<!T>) -> !stream.stream<i64> {
    ^bb0(%val : !T):
      %e:8 = stream.unpack %val : !T
      %cond = arith.cmpi ne, %e#0, %e#1 : i64
      cf.cond_br %cond, ^bb1, ^bb2
    ^bb1: 
      %t0 = arith.addi %e#0, %e#1 : i64
      %t1 = arith.addi %e#2, %e#3 : i64
      %t2 = arith.addi %e#4, %e#5 : i64
      %t3 = arith.addi %e#6, %e#7 : i64
      %t4 = arith.addi %t0, %t1 : i64
      %t5 = arith.addi %t2, %t3 : i64
      %t6 = arith.addi %t4, %t5 : i64
      cf.br ^bb3(%t6: i64)
    ^bb2: 
      %c = arith.constant 3 : i64
      %r = arith.shrui %e#0, %c : i64
      cf.br ^bb3(%r: i64)
    ^bb3(%res: i64): 
      stream.yield %res: i64
    }
    return %mapOut : !stream.stream<i64>
  }
}
