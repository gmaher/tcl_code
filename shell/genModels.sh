#!/bin/bash

#Note must be executed from withing directory
#Args:
#1: output directory


echo "Input arguments: $1"

SV="/home/marsdenlab/projects/SV3Dev/SimVascular/Build/SimVascular-build/mysim /home/marsdenlab/projects/SV3Dev/SimVascular/Build/SimVascular-build/Tcl/SimVascular_2.0/simvascular_startup.tcl"
TCL="/home/marsdenlab/projects/tcl_code/tcl/run_models.tcl"
OUT="/home/marsdenlab/projects/old_tcl_code/tcl_code/python/output/9/vtk"
cd $3

cd $OUT
rm $OUT/*

for g in ../groups/*; do
  echo "$g"
  $SV $TCL $g -tcl
  sleep 100
done

#
#$1 $2
