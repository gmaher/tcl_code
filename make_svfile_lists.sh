#!/bin/bash

#this file takes in a directory and searches for a set of predefined file patterns

echo "searching "$1

find $1 -name "*-cm.mha" | sort > images.txt

find $1 -name "*groups*" -type d | sort > groups.txt

find $1 -name "*.paths" | sort > paths.txt

find $1 -name "*-cm_E48.mha" | sort > edge48.txt

find $1 -name "*-cm_E96.mha" | sort > edge96.txt

#count lines to make sure no duplicates
echo "Number images"
find $1 -name "*-cm.mha" | sort | wc -l

echo "number of groups"
find $1 -name "*groups*" -type d | wc -l 

echo "number of paths"
find $1 -name "*.paths" | wc -l

echo "number of 48 edges"
find $1 -name "*-cm_E48.mha" | wc -l

echo "number of 96 edges"
find $1 -name "*-cm_E96.mha" | wc -l
