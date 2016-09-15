#!/bin/bash

echo "submitting scripts in "$1

for file in $1*.sh; do
	echo "submitting: "$file
	sbatch $file

done