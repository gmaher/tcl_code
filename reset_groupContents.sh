#!/bin/bash

for file in $(find $1 -name "*group_contents.tcl"); do
	echo $file
	grep -v "_edge48*" $file > temp && mv temp $file
	grep -v "_edge96*" $file > temp && mv temp $file
	grep -v "_image*" $file > temp && mv temp $file
done