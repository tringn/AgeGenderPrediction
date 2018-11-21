TOOLS=/home/trinn/workspace/caffe/build/tools #PATH TO caffe/build/tools
DATA=./
DEF_FILES=data/DEF
OUT=data/lmdb
rm -rf $OUT
mkdir -p $OUT

# Set RESIZE=true to resize the images to 227x227. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=227
  RESIZE_WIDTH=227
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

echo "Creating Age train lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA \
    $DEF_FILES/age_train.txt \
    $OUT/age_train_lmdb 
    
echo "Creating Age test lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA \
    $DEF_FILES/age_test.txt \
    $OUT/age_test_lmdb 

echo "Creating Gender train lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA \
    $DEF_FILES/gender_train.txt \
    $OUT/gender_train_lmdb 
    
echo "Creating Gender test lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA \
    $DEF_FILES/gender_test.txt \
    $OUT/gender_test_lmdb 
