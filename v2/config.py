# coding:utf-8

# yolo
TRAIN_INPUT_SIZES = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
TEST_INPUT_SIZE = 544
STRIDES = [8, 16, 32]
IOU_LOSS_THRESH = 0.5

# train
BATCH_SIZE = 6
LEARN_RATE_INIT = 1e-4
LEARN_RATE_END = 1e-6
WARMUP_PERIODS = 2
MAX_PERIODS = 50

GT_PER_GRID = 3
MOVING_AVE_DECAY = 0.9995

# test
SCORE_THRESHOLD = 0.01    # The threshold of the probability of the classes
IOU_THRESHOLD = 0.45     # The threshold of the IOU when implement NMS

# name and path
DATASET_PATH = '/home/xzh/doc/code/python_code/data/VOC'
PROJECT_PATH = '/home/xzh/doc/code/python_code/Stronger-yolo/v2'
ANNOT_DIR_PATH = 'data'
WEIGHTS_DIR = 'weights'
WEIGHTS_INIT = 'darknet2tf/saved_model/darknet53.ckpt'
LOG_DIR = 'log'
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

