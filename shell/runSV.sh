#!/bin/bash

#Note must be executed from withing directory

cd ../I2INet/raw

../../JSV/sv_build/mysim /home/marsdenlab/projects/tcl_code/tcl/run.tcl

cd ../../tcl_code

python vtkGroupsToCSV.py ../I2INet/raw
