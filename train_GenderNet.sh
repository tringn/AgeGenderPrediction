#!/bin/sh
mkdir models/GenderNet -p

/home/trinn/workspace/caffe/build/tools/caffe train \
-solver prototxt/GenderNet_solver.prototxt \
-gpu 0

