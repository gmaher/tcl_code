#!/bin/bash

#args:
# 1 - vascular data directory
# 2 - Edge code to looks for
# 3 - Output directory
if [ $# -lt 3 ]
	then
		echo "Not enough arguments supplied, need vascular data dir, edge code, output dir"
		exit 1
fi

echo "searching ${1} for edge codes ${2}, outputting to ${3}"
#this file takes in a directory and searches for a set of predefined file patterns
touch $3/images.txt $3/groups.txt $3/paths.txt $3/edge48.txt $3/edge96.txt
rm $3/images.txt $3/groups.txt $3/paths.txt $3/edge48.txt $3/edge96.txt
touch $3/images.txt $3/groups.txt $3/paths.txt $3/edge48.txt $3/edge96.txt

for dir in $1*; do
	num_images="$(find $dir -name "*OSMSC*-cm.mha" | wc -l)"

	num_groups="$(find $dir -name "*groups-cm" -type d | wc -l)"

	num_paths="$(find $dir -name "*.paths" | wc -l)"

	num_edge="$(find $dir -name "*-cm_${2}.mha" | wc -l)"

	echo $dir
	echo $num_images $num_groups $num_paths $num_edge

	# if ([ "$num_images" == "1" ] && [ "$num_groups" -ge "1" ]  &&
	#  [ "$num_paths" -ge "1" ] && [ "$num_edge" == "1" ]);
	 if ([ "$num_images" == "1" ] && [ "$num_groups" == "1" ]  &&
	  [ "$num_paths" == "1" ] && [ "$num_edge" == "1" ]);
		then
			find $dir -name "*OSMSC*-cm.mha" >> $3/images.txt
			find $dir -name "*groups-cm" -type d -print -quit >> $3/groups.txt
			find $dir -name "*.paths" -print -quit >> $3/paths.txt
			find $dir -name "*-cm_${2}.mha" >> $3/edge.txt
		fi

done
