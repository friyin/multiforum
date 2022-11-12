#!/bin/sh

ffmpeg -framerate 60 -pattern_type glob -i 'output/2022-11/StableFun/*.png' -c:v libx264 -pix_fmt yuv420p video/friyo_test1.mp4
