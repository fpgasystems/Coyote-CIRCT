!T = tuple<i64, i64, i64, i64, i64, i64, i64, i64>
!T_p = tuple<i64, i64>

module {
  func.func @top_s(%in: !stream.stream<!T>) -> (!stream.stream<i64>) {
   
    // Map 1
    %mapOut1 = stream.map(%in) : (!stream.stream<!T>) -> !stream.stream<!T_p> {
    ^0(%val : !T):
      %e:8 = stream.unpack %val : !T
      %res = stream.pack %e#3, %e#6 : !T_p
      stream.yield %res : !T_p
    }

    // Filter
    %filOut = stream.filter(%mapOut1) : (!stream.stream<!T_p>) -> !stream.stream<!T_p> {
    ^0(%val : !T_p):
      %e:2 = stream.unpack %val : !T_p
      %c0 = arith.constant 100 : i64
      %cond = arith.cmpi sle, %e#0, %c0 : i64
      stream.yield %cond : i1
    }

    // Map 2
    %mapOut2 = stream.map(%filOut) : (!stream.stream<!T_p>) -> !stream.stream<i64> {
    ^0(%val : !T_p):
      %e:2 = stream.unpack %val : !T_p
      stream.yield %e#1 : i64
    }

    // Output 
    return %mapOut2 : !stream.stream<i64> 
  }
}
