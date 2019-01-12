# coding:utf-8

# yolo
# TRAIN_INPUT_SIZES = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
TRAIN_INPUT_SIZES = [416]
TEST_INPUT_SIZE = 416  # width=height
STRIDES = [8, 16, 32]
IOU_LOSS_THRESH = 0.5

# train
BATCH_SIZE = 32
LEARN_RATE_INIT = 1e-3
MAX_LEARN_RATE_DECAY_TIME = 2
MAX_WAVE_TIME = 2
MAX_PERIODS = 50
ANCHORS = [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],            # Anchors for small obj
           [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],    # Anchors for medium obj
           [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] # Anchors for big obj
#ANCHORS = [[[3.0, 4.75], [6.375, 11.0], [9.0, 23.75]],
#           [[7.75, 18.25], [8.25, 6.0625], [11.0625, 11.4375]],
#           [[7.0625, 10.1875], [10.46875, 5.9375], [11.46875, 11.25]]]

ANCHOR_PER_SCALE = 3
MOVING_AVE_DECAY = 0.9995
SAVE_ITER = 1
MAX_BBOX_PER_SCALE = 50

# test
SCORE_THRESHOLD = 0.2    # The threshold of the probability of the classes
IOU_THRESHOLD = 0.5     # The threshold of the IOU when implement NMS

# compute environment
GPU = '0'

# name and path
DATASET_PATH = '/home/xzh/doc/code/python_code/data/VOC'
ANNOT_DIR_PATH = 'data'
WEIGHTS_DIR = 'weights'
WEIGHTS_FILE = 'yolo_coco_initial.ckpt'
LOG_DIR = 'log'
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

