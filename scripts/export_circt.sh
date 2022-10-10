#!/bin/bash

# Add the binaries to the path
export PATH="$PWD/circt-stream/build/bin:$PWD/circt-stream/circt/build/bin:$PATH" 

if (($# < 1)); then
  echo "ERR: Provide a dialect (\$1)!"
  exit 1
fi

case $1 in
  "std")
    # Std
    if (($# != 5)); then
        echo "ERR: Provide a dialect (\$1), a kernel type (\$2), a kernel name (\$3), a pipelining support strategy (\$4), and a buffering strategy (\$5)!"
      exit 1
    fi

    # Output
    mkdir -p hw/$1/$2/gen
    mkdir -p hw/$1/$2/gen/$3_$4/

    if ! [[ "$4" =~ ^(none|locking|pipelining)$ ]]; then
        echo "ERR: Unknown pipelining support strategy $3"
        exit 1
    fi

    if [[ $5 != "custom" ]]; then
    hlstool hw/$1/$2/$3.mlir --dynamic-firrtl --buffering-strategy=all --dynamic-parallelism=$4 --ir |
    firtool -format=mlir -o hw/$1/$2/gen/$3_$4/ --split-verilog --lowering-options=disallowLocalVariables
    else
    case $4 in
      "none")
        stream-opt hw/$1/$2/$3.mlir --lower-std-to-handshake=disable-task-pipelining \
          --canonicalize='top-down=true region-simplify=true' \
          --handshake-materialize-forks-sinks --canonicalize \
          --custom-buffer-insertion --lower-handshake-to-firrtl | \
          firtool -format=mlir -o hw/$1/$2/gen/$3_$4/ --split-verilog --lowering-options=disallowLocalVariables
      ;;
      "locking")
        stream-opt hw/$1/$2/$3.mlir --lower-std-to-handshake=disable-task-pipelining \
          --canonicalize='top-down=true region-simplify=true' --handshake-lock-functions \
          --handshake-materialize-forks-sinks --canonicalize \
          --custom-buffer-insertion --lower-handshake-to-firrtl | \
          firtool -format=mlir -o hw/$1/$2/gen/$3_$4/ --split-verilog --lowering-options=disallowLocalVariables
      ;;
      "pipelining")
        stream-opt hw/$1/$2/$3.mlir --lower-std-to-handshake \
          --canonicalize='top-down=true region-simplify=true' \
          --handshake-materialize-forks-sinks --canonicalize \
          --custom-buffer-insertion --lower-handshake-to-firrtl | \
          firtool -format=mlir -o hw/$1/$2/gen/$3_$4/ --split-verilog --lowering-options=disallowLocalVariables
      ;;
      *)
        echo "ERR: Unknown pipelining support strategy $3"
        exit 1
        ;;
    esac
    fi
    # Build driver
    #stream-opt hw/$1/driver.mlir --custom-buffer-insertion | \
    stream-opt hw/$1/driver.mlir | \
    hlstool --ir-input-level=1 --dynamic-firrtl --buffering-strategy=all --ir | \
        firtool -format=mlir -o hw/$1/$2/gen/$3_$4/ --split-verilog --lowering-options=disallowLocalVariables
    ;;

  "stream")
    # Stream
    if (($# != 4)); then
      echo "ERR: Provide a dialect (\$1), a kernel type (\$2), a kernel name (\$3), and a fifo buffer size (\$4)!"
      exit 1
    fi

    # Output
    mkdir -p hw/$1/$2/gen
    mkdir -p hw/$1/$2/gen/$3/

    # Build kernel
    stream-opt hw/$1/$2/$3.mlir --convert-stream-to-handshake \
      --handshake-materialize-forks-sinks --canonicalize \
      --custom-buffer-insertion=fifobuffer-size=$4 --lower-handshake-to-firrtl | \
    firtool --format=mlir --lowering-options=disallowLocalVariables --verilog -o hw/$1/$2/gen/$3/$3.sv
    ;;

  *)
    echo "ERR: Dialect not supported!"
    ;;
esac
