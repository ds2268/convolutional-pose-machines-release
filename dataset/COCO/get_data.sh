#!/usr/bin/env bash

echo Getting COCO train2014 datatset ...
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip

echo Getting COCO val2014 dataset ...
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip

echo Getting keypoint annotation
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip

echo Unzipping train2014
unzip -qq train2014.zip

echo Unzipping val2014
unzip -qq val2014.zip

echo Unzipping keypoint annotations
unzip person_keypoints_trainval2014.zip

rm -f train2014.zip
rm -f val2014.zip
rm -f person_keypoints_trainval2014.zip
