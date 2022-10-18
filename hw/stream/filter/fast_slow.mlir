// REQUIRES: cocotb, iverilog

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=projection-selection --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

!T = tuple<i1, i64, i64, i64, i64, i64, i64, i64, i64>
!Tout = tuple<i64, i64>

module {
  func.func @top(%in: !stream.stream<!T>) -> !stream.stream<!Tout> {

    // Split
    %data1, %where1 = stream.split(%in) : (!stream.stream<!T>) -> (!stream.stream<!T>, !stream.stream<!T>) {
    ^0(%val : !T):
      stream.yield %val, %val : !T, !T
    }
    
    // Map (where clause)
    %where2 = stream.map(%where1) : (!stream.stream<!T>) -> !stream.stream<i1> {
      ^bb0(%val : !T):
        %e:9 = stream.unpack %val : !T
        %c0 = arith.constant 1 : i1
        %cond = arith.xori %c0, %e#0 : i1
        cf.cond_br %cond, ^bb1, ^bb2
      
      // Long path
      ^bb1: 
        %t0 = arith.muli %e#1, %e#2 : i64
        %t1 = arith.muli %t0,  %e#3 : i64
        %t2 = arith.muli %t1,  %e#4 : i64
        %t3 = arith.muli %t2,  %e#5 : i64
        %t4 = arith.muli %t3,  %e#6 : i64
        %t5 = arith.muli %t4,  %e#7 : i64
        %t6 = arith.muli %t5,  %e#8 : i64
        //%t2 = arith.muli %t1,  %e#5 : i64
        //%t3 = arith.addi %e#6, %e#7 : i64
        //%t4 = arith.muli %t3,  %e#8 : i64
        //%t5 = arith.addi %t2,  %t4 : i64
        //%c1 = arith.constant 3 : i64
        //%t6 = arith.muli %t5,  %c1 : i64
        %c2 = arith.constant 2000 : i64
        %r0 = arith.cmpi slt, %t6, %c2 : i64
        cf.br ^bb3(%r0: i1)

      // Short path
      ^bb2: 
        %c3 = arith.constant 10 : i64
        %r1 = arith.cmpi slt, %e#1, %c3 : i64
        cf.br ^bb3(%r1: i1)
      
      // Comp result
      ^bb3(%res: i1): 
        stream.yield %res: i1
    }

    // Combine
    %data2 = stream.combine(%data1, %where2) : (!stream.stream<!T>, !stream.stream<i1>) -> !stream.stream<tuple<!T, i1>> {
    ^0(%val : !T, %pres: i1):
      %out = stream.pack %val, %pres : tuple<!T, i1>
      stream.yield %out : tuple<!T, i1>
    }

    // Filter
    %data3 = stream.filter(%data2) : (!stream.stream<tuple<!T, i1>>) -> !stream.stream<tuple<!T, i1>> {
    ^0(%val : tuple<!T, i1>):
      %out:2 = stream.unpack %val : tuple<!T, i1>
      %c1 = arith.constant 1 : i1
      %cond = arith.xori %c1, %out#1 : i1
      stream.yield %cond : i1
    }

    // Map
    %outData = stream.map(%data3) : (!stream.stream<tuple<!T, i1>>) -> !stream.stream<!Tout> {
    ^0(%val : tuple<!T, i1>):
      %e:2 = stream.unpack %val : tuple<!T, i1>
      %f:9 = stream.unpack %e#0 : !T
      %out = stream.pack %f#1, %f#2 : !Tout
      stream.yield %out : !Tout
    }

    return %outData : !stream.stream<!Tout>
  }

}