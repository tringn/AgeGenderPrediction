#!/bin/sh

if [ ! -d "./graph" ]; then
   mkdir -p ./graph; 
fi

AGENET_MODEL=$(ls -t models/AgeNet/caffenet_age_train*.caffemodel | head -n 1)
AGENET_DEPLOY=prototxt/AgeNet_deploy.prototxt

mvNCCompile -s 12 -w $AGENET_MODEL $AGENET_DEPLOY -o graph/AgeNet.graph

GENDERNET_MODEL=$(ls -t models/GenderNet/caffenet_gender_train*.caffemodel | head -n 1)
GENDERNET_DEPLOY=prototxt/GenderNet_deploy.prototxt

mvNCCompile -s 12 -w $GENDERNET_MODEL $GENDERNET_DEPLOY -o graph/GenderNet.graph

# Convert mean file
python3 utils/convert_mean.py
