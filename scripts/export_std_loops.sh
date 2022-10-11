#!/bin/bash

# Add the binaries to the path
export PATH="$PWD/circt-stream/build/bin:$PWD/circt-stream/circt/build/bin:$PATH" 

stream-opt hw/std/base/loop_sequence.mlir --lower-std-to-handshake=disable-task-pipelining \
  --canonicalize='top-down=true region-simplify=true' \
  --handshake-materialize-forks-sinks --canonicalize \
  --custom-buffer-insertion --handshake-insert-buffers=strategy=cycles \
  --lower-handshake-to-firrtl | \
    firtool -format=mlir -o hw/std/base/gen_loop_sequence_none --split-verilog --lowering-options=disallowLocalVariables

hlstool hw/std/driver.mlir --ir-input-level=1 --dynamic-firrtl --buffering-strategy=all --ir | \
  firtool -format=mlir -o hw/std/base/gen_loop_sequence_none/ --split-verilog --lowering-options=disallowLocalVariables


stream-opt hw/std/base/loop_sequence.mlir --lower-std-to-handshake=disable-task-pipelining \
  --canonicalize='top-down=true region-simplify=true' --handshake-lock-functions \
  --handshake-materialize-forks-sinks --canonicalize \
  --custom-buffer-insertion --handshake-insert-buffers=strategy=cycles \
  --lower-handshake-to-firrtl | \
  firtool -format=mlir -o hw/std/base/gen_loop_sequence_locking/ --split-verilog --lowering-options=disallowLocalVariables

hlstool hw/std/driver.mlir --ir-input-level=1 --dynamic-firrtl --buffering-strategy=all --ir | \
  firtool -format=mlir -o  hw/std/base/gen_loop_sequence_locking/ --split-verilog --lowering-options=disallowLocalVariables


stream-opt hw/std/base/loop_sequence.mlir --lower-std-to-handshake \
  --canonicalize='top-down=true region-simplify=true' \
  --handshake-materialize-forks-sinks --canonicalize \
  --custom-buffer-insertion --handshake-insert-buffers=strategy=cycles \
  --lower-handshake-to-firrtl | \
    firtool -format=mlir -o  hw/std/base/gen_loop_sequence_pipelining --split-verilog --lowering-options=disallowLocalVariables

hlstool hw/std/driver.mlir --ir-input-level=1 --dynamic-firrtl --buffering-strategy=all --ir | \
  firtool -format=mlir -o  hw/std/base/gen_loop_sequence_pipelining/ --split-verilog --lowering-options=disallowLocalVariables
