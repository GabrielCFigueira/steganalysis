#!/bin/bash

recorded_video_folder="/home/vagrant/Downloads"
output_location="/home/vagrant/steganalysis/whereby4/training-data/videos/"

declare -i a=1
for i in "$recorded_video_folder"/whereby-regular*; do
	echo $i
	ffmpeg -i "$i" -fflags +genpts -r 24 "$output_location""clean/clean$a.264"
	a=$a+1
done

a=1
for i in "$recorded_video_folder"/whereby-stegozoa*; do
	ffmpeg -i "$i" -fflags +genpts -r 24 "$output_location""stego/stego$a.264"
	a=$a+1
done
