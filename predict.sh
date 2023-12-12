#!/bin/bash

# python predict.py --cfg configs/gen6d_pretrain.yaml \
#                   --database custom/mouse_processed \
#                   --video data/custom/video/mouse-test.mp4 \
#                   --resolution 960 \
#                   --transpose \
#                   --output data/custom/mouse_processed/test \
#                   --ffmpeg ffmpeg




# python predict_image.py --cfg configs/gen6d_pretrain.yaml \
#                   --database custom/mouse_processed \
#                   --image data/custom/image/test.jpg \
#                   --output data/custom/mouse_processed/test \
#                   --ffmpeg ffmpeg


python predict_image.py --cfg configs/gen6d_pretrain.yaml \
                  --database custom/square_b \
                  --image data/custom/image/Dec12/color_image_0.png \
                  --output data/custom/square_b/test \
                  --ffmpeg ffmpeg