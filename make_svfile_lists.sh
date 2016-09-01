#!/bin/bash

#args:
# 1 - vascular data directory
# 2 - Edge code to looks for
echo "searching ${1} for edge codes ${2}"
#this file takes in a directory and searches for a set of predefined file patterns
touch images.txt groups.txt paths.txt edge48.txt edge96.txt
rm images.txt groups.txt paths.txt edge48.txt edge96.txt
touch images.txt groups.txt paths.txt edge48.txt edge96.txt

for dir in $1*; do
	num_images="$(find $dir -name "*OSMSC*-cm.mha" | wc -l)"

	num_groups="$(find $dir -name "*groups-cm" -type d | wc -l)"

	num_paths="$(find $dir -name "*.paths" | wc -l)"

	num_edge="$(find $dir -name "*-cm_${2}.mha" | wc -l)"

	echo $dir
	echo $num_images $num_groups $num_paths $num_edge

	if ([ "$num_images" == "1" ] && [ "$num_groups" == "1" ]  &&
	 [ "$num_paths" == "1" ] && [ "$num_edge" == "1" ]);
		then
			find $dir -name "*OSMSC*-cm.mha" >> images.txt
			find $dir -name "*groups-cm" -type d >> groups.txt
			find $dir -name "*.paths" >> paths.txt
			find $dir -name "*-cm_${2}.mha" >> edge.txt
		fi

done