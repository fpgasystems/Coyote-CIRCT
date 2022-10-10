#!/bin/bash

#
# Args
#
if (($# != 2))
    then echo "ERR:  Provide the build name (\$1) and the path to the build sources (\$2)!"
    exit 1
fi

mkdir build_sw_$1
cd build_sw_$1
/usr/bin/cmake ../sw -DTARGET_PRJ=../$2
make
