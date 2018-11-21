#!/bin/sh

latest=$(ls -t models/GenderNet/caffenet_gender_train*.caffemodel | head -n 1)
if test -z $latest; then
        exit 1
fi

# /home/aiteam/workspace/caffe/build/tools/caffe test \
# -weights $latest \
# -model prototxt/GenderNet_train_test.prototxt \
# -iterations 13

python3 utils/Caffe_Convnet_ConfuxionMatrix.py --proto prototxt/GenderNet_deploy.prototxt --model $latest --lmdb data/lmdb/gender_test_lmdb/ --mean data/mean/mean.binaryproto --net GenderNet
