#!/bin/bash

recorded_video_folder="/home/vagrant/Dropbox/Apps/Jitsi Meet/Recordings/"
output_location="/home/vagrant/steganalysis/"

declare -a conditions
conditions=("bw_250" "bw_750" "bw_1500" "loss_2" "loss_5" "loss_10")

declare -i a=1
for i in "$recorded_video_folder"*reg*; do
	echo $i
	location=$output_location
	for s in "${conditions[@]}"; do
		echo $s
		if [[ $i =~ $s ]]; then
			location="$location$s/training-data/videos/clean/clean$a.264"
		fi
	done
	ffmpeg -i "$i" -vcodec libx264 "$location"
	a=$a+1
done

a=1
for i in "$recorded_video_folder"*stego*; do
	location=$output_location
	for s in "${conditions[@]}"; do
		if [[ $i == *$s* ]]; then
			location="$location$s/training-data/videos/stego/stego$a.264"
		fi
	done
	ffmpeg -i "$i" -vcodec libx264 "$location"
	a=$a+1
done
