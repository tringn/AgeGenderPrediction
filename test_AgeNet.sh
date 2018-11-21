#!/bin/sh

latest=$(ls -t models/AgeNet/caffenet_age_train*.caffemodel | head -n 1)
if test -z $latest; then
        exit 1
fi

# /home/aiteam/workspace/caffe/build/tools/caffe test \
# -weights $latest \
# -model prototxt/AgeNet_train_test.prototxt \
# -iterations 13

python3 utils/Caffe_Convnet_ConfuxionMatrix.py --proto prototxt/AgeNet_deploy.prototxt --model $latest --lmdb data/lmdb/age_test_lmdb/ --mean data/mean/mean.binaryproto --net AgeNet
