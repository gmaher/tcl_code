#!/bin/bash

#Note must be executed from withing directory
#Args:
#1: output directory


echo "Input arguments: $1"

SV="/home/marsdenlab/projects/JSV/sv_build/mysim"
TCL="/home/marsdenlab/projects/tcl_code/tcl/run_models.tcl"
OUT="/home/marsdenlab/projects/tcl_code/python/output/9/vtk"
cd $3

cd $OUT
rm $OUT/*

for g in ../groups/*; do
  echo "$g"
  $SV $TCL $g &
  sleep 70
done

#
#$1 $2
