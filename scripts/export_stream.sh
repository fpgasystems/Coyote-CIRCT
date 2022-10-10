#!/bin/bash

# Add the binaries to the path
export PATH="$PWD/circt-stream/build/bin:$PWD/circt-stream/circt/build/bin:$PATH" 

# Stream
if (($# != 3)); then
  echo "ERR: Provide a kernel type (\$1), a kernel name (\$2), and a fifo buffer size (\$3)!"
  exit 1
fi

# Output
mkdir -p hw/stream/$1/gen_$2/

# Build kernel
stream-opt hw/stream/$1/$2.mlir --convert-stream-to-handshake \
  --handshake-materialize-forks-sinks --canonicalize \
  --custom-buffer-insertion=fifobuffer-size=$3 --lower-handshake-to-firrtl | \
firtool --format=mlir --lowering-options=disallowLocalVariables --verilog -o hw/stream/$1/gen_$2/$2.sv