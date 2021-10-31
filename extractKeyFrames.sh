#!/bin/bash

recorded_video_folder="/home/vagrant/steganalysis/jitsi2/training-data/videos/"
output_location="/home/vagrant/steganalysis/training-data/videos/"


declare -i a=1
for i in "$recorded_video_folder"clean/*; do
	echo $i
	ffmpeg -i "$i" -vf select="eq(pict_type\,PICT_TYPE_I)" -vsync 2 "$output_location""clean/clean$a.264"
	a=$a+1
done

a=1
for i in "$recorded_video_folder"stego/*; do
	ffmpeg -i "$i" -vf select="eq(pict_type\,PICT_TYPE_I)" -vsync 2 "$output_location""stego/stego$a.264"
	a=$a+1
done
