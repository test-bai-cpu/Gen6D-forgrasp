#!/bin/bash

# python predict.py --cfg configs/gen6d_pretrain.yaml \
#                   --database custom/mouse_processed \
#                   --video data/custom/video/mouse-test.mp4 \
#                   --resolution 960 \
#                   --transpose \
#                   --output data/custom/mouse_processed/test \
#                   --ffmpeg ffmpeg




python predict_image.py --cfg configs/gen6d_pretrain.yaml \
                  --database custom/mouse_processed \
                  --image data/custom/image/test.jpg \
                  --output data/custom/mouse_processed/test \
                  --ffmpeg ffmpeg