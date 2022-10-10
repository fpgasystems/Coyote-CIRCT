#!/bin/bash

# Add the binaries to the path
export PATH="$PWD/circt-stream/build/bin:$PWD/circt-stream/circt/build/bin:$PATH" 

hlstool hw/std/base/loop_sequence.mlir --dynamic-firrtl --buffering-strategy=all --dynamic-parallelism=none --ir | \
    firtool -format=mlir -o hw/std/base/gen_loop_sequence_none --split-verilog --lowering-options=disallowLocalVariables

hlstool hw/std/driver.mlir --ir-input-level=1 --dynamic-firrtl --buffering-strategy=all --ir | \
  firtool -format=mlir -o hw/std/base/gen_loop_sequence_none/ --split-verilog --lowering-options=disallowLocalVariables


hlstool hw/std/base/loop_sequence.mlir --dynamic-firrtl --buffering-strategy=all --dynamic-parallelism=locking --ir | \
    firtool -format=mlir -o  hw/std/base/gen_loop_sequence_locking --split-verilog --lowering-options=disallowLocalVariables

hlstool hw/std/driver.mlir --ir-input-level=1 --dynamic-firrtl --buffering-strategy=all --ir | \
  firtool -format=mlir -o  hw/std/base/gen_loop_sequence_locking/ --split-verilog --lowering-options=disallowLocalVariables


hlstool hw/std/base/loop_sequence.mlir --dynamic-firrtl --buffering-strategy=all --dynamic-parallelism=pipelining --ir | \
    firtool -format=mlir -o  hw/std/base/gen_loop_sequence_pipelining --split-verilog --lowering-options=disallowLocalVariables

hlstool hw/std/driver.mlir --ir-input-level=1 --dynamic-firrtl --buffering-strategy=all --ir | \
  firtool -format=mlir -o  hw/std/base/gen_loop_sequence_pipelining/ --split-verilog --lowering-options=disallowLocalVariables
