#!/bin/sh
mkdir models/AgetNet/ -p

/home/trinn/workspace/caffe/build/tools/caffe train \
-solver prototxt/AgeNet_solver.prototxt \
-gpu 0 

