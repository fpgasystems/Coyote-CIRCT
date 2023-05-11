#!/bin/bash

# TODO drop lld, clang, and clang++ requirements? 

SCRIPT_DIR=$(pwd)/$(dirname "$0")

cd ${SCRIPT_DIR}/../circt-stream
mkdir -p circt/build && cd circt/build
cmake ../llvm/llvm \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=.. \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
make -j

# Back to circt-stream
cd ${SCRIPT_DIR}/../circt-stream

mkdir -p build && cd build
cmake .. \
    -DLLVM_DIR=$PWD/../circt/build/lib/cmake/llvm \
    -DMLIR_DIR=$PWD/../circt/build/lib/cmake/mlir \
    -DCIRCT_DIR=$PWD/../circt/build/lib/cmake/circt \
    -DLLVM_EXTERNAL_LIT=$PWD/../circt/build/bin/llvm-lit \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
make -j

cd ..
