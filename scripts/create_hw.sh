#!/bin/bash

#
# Args
#

if (($# != $3 + 3)) 
    then echo "ERR:  Provide the build name (\$1), target device (\$2), target number of regions (\$3), and the applications (dialect/type/circuit) (\$4)+!"
    exit 1
fi

dbg="True"
curr=0
args_arr=()
for i in ${@:4:$3} ; do 
  IFS='/' read -ra SPLT <<< "$i"
  args_arr+=( ${SPLT[0]} )
  args_arr+=( ${SPLT[1]} )
  args_arr+=( ${SPLT[2]} )
  curr=$((curr+1))
done

#
# Init HW
#

cnt=`ls -1 build_hw_$1 2>/dev/null | wc -l`
if [ $cnt != 0 ]
then 
  echo "ERR:  Directory already exists!"
  exit
fi

#
# Dirs
#

mkdir build_hw_$1
mkdir build_hw_$1/ext

curr=0
for i in ${@:4:$3} ; do 
  mkdir build_hw_$1/config_$curr 
  mkdir build_hw_$1/config_$curr/ext
  curr=$((curr+1))
done

#
# Move files
#

curr=0
for i in ${@:4:$3} ; do 
  cp -r hw/${args_arr[3*$curr+0]}//${args_arr[3*$curr+1]}/ext/*.sv build_hw_$1/ext
  cp -r hw/${args_arr[3*$curr+0]}/${args_arr[3*$curr+1]}/ext/*.svh build_hw_$1/ext
  cat hw/${args_arr[3*$curr+0]}/${args_arr[3*$curr+1]}/init_ip.tcl >> build_hw_$1/init_ip.tcl
  cp -r hw/${args_arr[3*$curr+0]}/${args_arr[3*$curr+1]}/${args_arr[3*$curr+2]}/* build_hw_$1/config_$curr/ext
  cp -r hw/${args_arr[3*$curr+0]}/${args_arr[3*$curr+1]}/user_logic.sv build_hw_$1/config_$curr/user_logic_c0_$curr.sv
  python scripts/replace.py build_hw_$1/config_$curr/user_logic_c0_$curr.sv $curr $dbg
  curr=$((curr+1))
done

#
# Config
#

cd build_hw_$1
cmake ../Coyote/hw -DFDEV_NAME=$2 -DN_REGIONS=$3 -DEN_BPSS=1 -DSHL_SCR_PATH="../scripts/package.tcl"

#
# Create project and compile
#

make shell
make compile
