#!/bin/bash

#this file takes in a directory and searches for a set of predefined file patterns

for dir in $1*; do
	num_images="$(find $dir -name "*OSMSC*-cm.mha" | wc -l)"

	num_groups="$(find $dir -name "*groups-cm" -type d | wc -l)"

	num_paths="$(find $dir -name "*.paths" | wc -l)"

	num_edge_48="$(find $1 -name "*-cm_E48.mha" | wc -l)"

	num_edge_96="$(find $1 -name "*-cm_E96.mha" | wc -l)"

	echo "$num_images, $num_groups, $num_paths, $num_edge_48, $num_edge_96"

	if ([ "$num_images" == "1" ] && [ "$num_groups" == "1" ]  &&
	 [ "$num_paths" == "1" ] && [ "$num_edge_48" == "1" ] && [ "$num_edge_96" == "1" ]);
		then
			echo "$num_images, $num_groups, $num_paths, $num_edge_48, $num_edge_96"
		fi

done