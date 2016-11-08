#!/bin/bash

#Note must be executed from withing directory
#Args:
#1: path to simvascular binary
#2: path to run.tcl
#3: output directory
    #NOTE: output directory must contain images.txt, paths.txt, groups.txt edges.txt
    #as generated by make_svfile_lists.sh

echo "Input arguments: $1 $2 $3"
echo "Generating raw data using simvascular"
cd $3
# paste images.txt paths.txt groups.txt edge.txt | while read I P G E; do
#   echo "$I $P $G $E"
#   ($1 $2 $I $P $G)
#   sleep 30
# done

#
$1 $2
