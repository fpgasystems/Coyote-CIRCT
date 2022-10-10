// REQUIRES: cocotb, iverilog

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --custom-buffer-insertion --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog 

func.func @top(%in: !stream.stream<i512>, %present: !stream.stream<i1>) -> (!stream.stream<i64>, !stream.stream<i512>) {
  %data, %hashed = stream.split(%in) : (!stream.stream<i512>) -> (!stream.stream<i512>, !stream.stream<i64>) {
  ^0(%val : i512):
    %hash = arith.trunci %val : i512 to i64
    stream.yield %val, %hash : i512, i64
  }

  %comb = stream.combine(%data, %present) : (!stream.stream<i512>, !stream.stream<i1>) -> !stream.stream<tuple<i512, i1>> {
  ^0(%val : i512, %pres: i1):
    %out = stream.pack %val, %pres : tuple<i512, i1>
    stream.yield %out : tuple<i512, i1>
  }

  %distinct = stream.filter(%comb) : (!stream.stream<tuple<i512, i1>>) -> !stream.stream<tuple<i512, i1>> {
  ^0(%val : tuple<i512, i1>):
    %out:2 = stream.unpack %val : tuple<i512, i1>
    %c1 = arith.constant 1 : i1
    %cond = arith.xori %c1, %out#1 : i1
    stream.yield %cond : i1
  }

  %outData = stream.map(%distinct) : (!stream.stream<tuple<i512, i1>>) -> !stream.stream<i512> {
  ^0(%val : tuple<i512, i1>):
    %out:2 = stream.unpack %val : tuple<i512, i1>
    stream.yield %out#0 : i512
  }

  return %hashed, %outData : !stream.stream<i64>, !stream.stream<i512> 
}
