#!/usr/bin/env bash
#$1 - input video
#$2 - output video
#$3 - output height
#$4 - output width
ffmpeg -i $1 -vf scale=$4x$3 $2
