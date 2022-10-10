!T = tuple<i64, i64, i64, i64, i64, i64, i64, i64>
!T_p = tuple<i64, i64, i64>
!T_f = tuple<i64, i64>

module {
  func.func @top_r(%in: !stream.stream<!T>, %join_in: !stream.stream<i1>, %hash_in: !stream.stream<i1>) -> 
    (!stream.stream<!T_f>, !stream.stream<i64>, !stream.stream<i64>) {
   
    // Map 1
    %mapOut1 = stream.map(%in) : (!stream.stream<!T>) -> !stream.stream<!T_p> {
    ^0(%val : !T):
      %e:8 = stream.unpack %val : !T
      %f0 = arith.addi %e#4, %e#5 : i64
      %res = stream.pack %e#0, %e#3, %f0 : !T_p
      stream.yield %res : !T_p
    }

    // Filter
    %filOut = stream.filter(%mapOut1) : (!stream.stream<!T_p>) -> !stream.stream<!T_p> {
    ^0(%val : !T_p):
      %e:3 = stream.unpack %val : !T_p
      %c0 = arith.constant 20000 : i64
      %cond = arith.cmpi sle, %e#2, %c0 : i64
      stream.yield %cond : i1
    }

    // Map 2
    %mapOut2 = stream.map(%filOut) : (!stream.stream<!T_p>) -> !stream.stream<!T_f> {
    ^0(%val : !T_p):
      %e:3 = stream.unpack %val : !T_p
      %res = stream.pack %e#0, %e#1 : !T_f
      stream.yield %res : !T_f
    }

    // Join
    %data_j1, %join_out = stream.split(%mapOut2) : (!stream.stream<!T_f>) -> (!stream.stream<!T_f>, !stream.stream<i64>) {
    ^0(%val : !T_f):
      %e:2 = stream.unpack %val : !T_f
      stream.yield %val, %e#0 : !T_f, i64
    }

    %comb_j1 = stream.combine(%data_j1, %join_in) : (!stream.stream<!T_f>, !stream.stream<i1>) -> !stream.stream<tuple<!T_f, i1>> {
    ^0(%val : !T_f, %join: i1):
      %res = stream.pack %val, %join : tuple<!T_f, i1>
      stream.yield %res : tuple<!T_f, i1>
    }

    %data_j2 = stream.filter(%comb_j1) : (!stream.stream<tuple<!T_f, i1>>) -> !stream.stream<tuple<!T_f, i1>> {
    ^0(%val : tuple<!T_f, i1>):
      %e:2 = stream.unpack %val : tuple<!T_f, i1>
      %c1 = arith.constant 1 : i1
      %cond = arith.xori %c1, %e#1 : i1
      stream.yield %cond : i1
    }

    %joinOut = stream.map(%data_j2) : (!stream.stream<tuple<!T_f, i1>>) -> !stream.stream<!T_f> {
    ^0(%val : tuple<!T_f, i1>):
      %e:2 = stream.unpack %val : tuple<!T_f, i1>
      stream.yield %e#0 : !T_f
    }

    // Distinct
    %data_h1, %hash_out = stream.split(%joinOut) : (!stream.stream<!T_f>) -> (!stream.stream<!T_f>, !stream.stream<i64>) {
    ^0(%val : !T_f):
      %e:2 = stream.unpack %val : !T_f
      stream.yield %val, %e#0 : !T_f, i64
    }

    %comb_h1 = stream.combine(%data_h1, %hash_in) : (!stream.stream<!T_f>, !stream.stream<i1>) -> !stream.stream<tuple<!T_f, i1>> {
    ^0(%val : !T_f, %hash: i1):
      %res = stream.pack %val, %hash : tuple<!T_f, i1>
      stream.yield %res : tuple<!T_f, i1>
    }

    %data_h2 = stream.filter(%comb_h1) : (!stream.stream<tuple<!T_f, i1>>) -> !stream.stream<tuple<!T_f, i1>> {
    ^0(%val : tuple<!T_f, i1>):
      %e:2 = stream.unpack %val : tuple<!T_f, i1>
      %c1 = arith.constant 1 : i1
      %cond = arith.xori %c1, %e#1 : i1
      stream.yield %cond : i1
    }

    %distData = stream.map(%data_h2) : (!stream.stream<tuple<!T_f, i1>>) -> !stream.stream<!T_f> {
    ^0(%val : tuple<!T_f, i1>):
      %e:2 = stream.unpack %val : tuple<!T_f, i1>
      stream.yield %e#0 : !T_f
    }

    // Output 
    return %distData, %join_out, %hash_out : !stream.stream<!T_f>, !stream.stream<i64>, !stream.stream<i64> 
  }
}
