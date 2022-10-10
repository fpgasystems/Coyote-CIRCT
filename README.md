# Coyote-CIRCT

Deploy CIRCT generated circuits effortlessly directly through Coyote.

## Prerequisites

`Vivado 2022.1`. 

## Setup

Initialize the submodules:

```bash
git submodule update --init --recursive
```

## Build `circt-stream`

To build the `circt-stream` project execute the following script:

```bash
bash scripts/build_circt_stream.sh
```

Once the project is built you can generate your own circuits created in one of the two supported dialects -> `std` (only used for benchmarks) and `stream`. Place the `.mlir` circuit description in a subdirectory within a target dialect. To generate the circuit execute the following script (Ex. a stream filter and a distinct operator):

```bash
bash scripts/export_circt.sh stream filter addcomp 128
bash scripts/export_circt.sh stream mem distinct 128
```

Check the script for the detailed list of args needed.

## Build `coyote`

#### `HW`

Run the following script to build the bitstream containing previously generated circuits (Ex. containing two previously generated circuits):
~~~~
$ sh scripts/create_hw.sh test u55c 2 stream/filter/gen_addcomp stream/mem/gen_distinct
~~~~

List of the arguments:

| $arg      | Description                                                               |
| --------- | ------------------------------------------------------------------------- |
| 1         | Name of the project dir (appended to build_hw_)                           |
| 2         | Target device (u50, u55c, u200, u250, u280, vcu118, enzian)               |
| 3         | Number of circuits integrated (each as a separate vFPGA)                  |
| 4 : 4+\$3 | Circuit directory (dialect_dir/type_dir/circuit_dir)                      |

Regular RTL circuits designed in HDL can also be included in the same manner.

#### Circuit dialects and types

Each circuit type has a separate directory under `hw/dialect/`.  Different interface to CIRCT generated circuits or a different system configuration imply a different circuit type. The following types are present for now:

| Type      | Description                                                               |
| --------- | ------------------------------------------------------------------------- |
| Map       | Base streaming computations, these do not affect length of the streams, simple data flow computation.           
| Filter    | Filtering circuits. These circuits can produce unknown number of outputs.                                  |
| Reduction | Similar to base circuits, with the internal reduction and added statistics being fired on last transfers.
| Mem       | Circuits showcasing integration with external circuits. These can be either RTL or some other form of HLS.                |
| Custom    | Custom circuits.                                            |

To add a new circuit type create a new directory under `hw/dialect/`. 

The `user_logic.sv` is the top level file for the integration of the circuits of a specific type into `coyote`. This file needs to be present and needs to be adapted for every new circuit type.

The `init_ip.tcl` instantiates any necessary ip cores.

All other files necessary for the functioning of the circuit can be placed in the same directory under `ext/`.

#### CIRCT generated code

Each exported circuit can be placed in a separate subdirectory within a circuit type. The rest should be handled by the scripts. To instantiate an exported circuit, pass the path of this subdirectory within `hw` (type/subtype) as one of the parameters to the init script.

#### `SW`

~~~~
$ sh scripts/create_sw.sh <args>
~~~~

List of the arguments:

| $arg      | Description                                                               |
| --------- | ------------------------------------------------------------------------- |
| 1         | Name of the project dir (appended to build_sw_)                           |
| 2         | Path to sources                                                           |
