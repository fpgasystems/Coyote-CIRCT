#!/bin/bash

#
## Micro benchmarks (hardcoded)
#

bash scripts/export_std_nested.sh
bash scripts/export_std_loops.sh

#
## Stream circuits
#

#bash scripts/export_circt.sh map query_r 128
bash scripts/export_stream.sh filter addcomp 128
bash scripts/export_stream.sh reduction max 128
bash scripts/export_stream.sh mem distinct 128
bash scripts/export_stream.sh custom_join query_r 128


