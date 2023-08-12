# Visually Assisted Self-supervised Audio Speaker Localization and Tracking

### Pre-request

python 3.6, pytorch 1.7

### Dataset

AV16.3 dataset: https://zenodo.org/record/4449274#.YrQ6v-yZPJ8


### Feature Extraction


DSFD: https://github.com/Tencent/FaceDetection-DSFD

pytorch-segmentation: https://github.com/yassouali/pytorch-segmentation

calculating gccphat: https://github.com/smartcameras/AV3T/tree/master/gcf


### Data Preparation

make sure to obtain the segmentation results for every image

`python gccphat.py`


### Training and Evaluation

`python train.py`