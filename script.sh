#!/bin/bash

recorded_video_folder="/home/vagrant/Dropbox/Apps/Jitsi Meet/Recordings/"
output_location="/home/vagrant/steganalyse/training-data/videos/"


declare -i a=1
for i in "$recorded_video_folder"*reg*; do
	echo $i
	ffmpeg -i "$i" -vcodec libx264 "$output_location""clean/clean$a.264"
	a=$a+1
done

a=1
for i in "$recorded_video_folder"*stego*; do
	ffmpeg -i "$i" -vcodec libx264 "$output_location""stego/stego$a.264"
	a=$a+1
done
