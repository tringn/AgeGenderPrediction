TOOLS=/home/trinn/workspace/caffe/build/tools # PATH TO caffe/build/tools
DATA=data/lmdb/age_train_lmdb
OUT=data/mean
rm -rf $OUT
mkdir -p $OUT

$TOOLS/compute_image_mean $DATA $OUT/mean.binaryproto


