#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH

mkdir -p datasets/full_teeth media/images/teeth media/images/single_tooth results models

git clone https://github.com/ultralytics/yolov5.git notebooks/yolov5/
