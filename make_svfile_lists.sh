#!/bin/bash

#this file takes in a directory and searches for a set of predefined file patterns
touch images.txt groups.txt paths.txt edge48.txt edge96.txt
rm images.txt groups.txt paths.txt edge48.txt edge96.txt
touch images.txt groups.txt paths.txt edge48.txt edge96.txt

for dir in $1*; do
	num_images="$(find $dir -name "*OSMSC*-cm.mha" | wc -l)"

	num_groups="$(find $dir -name "*groups-cm" -type d | wc -l)"

	num_paths="$(find $dir -name "*.paths" | wc -l)"

	num_edge_48="$(find $dir -name "*-cm_E48.mha" | wc -l)"
	#num_edge_48=1

	num_edge_96="$(find $dir -name "*-cm_E96.mha" | wc -l)"

	if ([ "$num_images" == "1" ] && [ "$num_groups" == "1" ]  &&
	 [ "$num_paths" == "1" ] && [ "$num_edge_48" == "1" ] && [ "$num_edge_96" == "1" ]);
		then
			#echo "$num_images, $num_groups, $num_paths, $num_edge_48, $num_edge_96"
			find $dir -name "*OSMSC*-cm.mha" >> images.txt
			find $dir -name "*groups-cm" -type d >> groups.txt
			find $dir -name "*.paths" >> paths.txt
			find $dir -name "*-cm_E48.mha" >> edge48.txt
			find $dir -name "*-cm_E96.mha" >> edge96.txt

		fi

done