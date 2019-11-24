#!/usr/bin/env bash
#$1 - input image
#$2 - output image
#$3 - output height
#$4 - output width
convert-im6 $1 -resample $4x$3 -resize $4x$3 $2
