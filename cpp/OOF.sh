#!/bin/bash


EXE=/home/marsdenlab/libraries/ITK-TubularGeodesics/build/itkMultiScaleTubularityMeasureImageFilter
sigma_min=0.2
sigma_max=2.0
num_scales=8
paste test.txt output_images.txt | while read I O; do
  echo $I
  $EXE $I $O 0 0 0 0 0 0 0 0 $sigma_min $sigma_max $num_scales $sigma_min 1 0

done
