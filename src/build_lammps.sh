#!/bin/sh

export OMPI_CXX=clang++

make package-update
make -j 4 $1
