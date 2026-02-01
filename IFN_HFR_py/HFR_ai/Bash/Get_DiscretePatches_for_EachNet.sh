#!/bin/bash

InPath=$1
OutPath=$2
sub=$3
Iter=$4
hemis='lh rh'
nets='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19'
mkdir $OutPath/$sub


for hemi in $hemis
do
for net in $nets
do
# Preprocessing,smooth with 1
mri_surf2surf --srcsubject fsaverage4 --sval ${InPath}/${sub}/Iter_$Iter/Network_${net}_${hemi}.mgh --trgsubject fsaverage4 --tval $OutPath/${sub}/Network_${net}_sm1_${hemi}.mgh --hemi $hemi --nsmooth-in 1

# Get the discrete patches
mri_surfcluster --in ${OutPath}/${sub}/Network_${net}_sm1_${hemi}.mgh --subject fsaverage4 --hemi $hemi --thmin 0.01  --ocn ${OutPath}/${sub}/Network_${net}_sm1_Patch_${hemi}.mgh

rm ${OutPath}/${sub}/Network_${net}_sm1_${hemi}.mgh
done
done
