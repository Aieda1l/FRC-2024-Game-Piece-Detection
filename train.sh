#!/bin/bash

# yolo detect train \
#      data=data/data.yaml \
#      model=ultralytics/cfg/models/v10/yolov10n.yaml \
#      epochs=250 \
#      batch=64 \
#      imgsz=640 \
#      device=0

yolo task=detect \
     mode=train \
     epochs=250 \
     batch=96 \
     plots=True \
     model=pretrained/yolov10n.pt \
     data=data/data.yaml \
     device=0

# Wait for user input before closing (similar to 'pause' in batch script)
read -p "Press any key to continue..."

