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
PERIODS_FOR_STEP0 = 20
MAX_PERIODS = 30
ANCHORS = [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],            # Anchors for small obj
           [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],    # Anchors for medium obj
           [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] # Anchors for big obj

ANCHOR_PER_SCALE = 3
MOVING_AVE_DECAY = 0.9995
MAX_BBOX_PER_SCALE = 150

# test
MULTI_TEST = False
FLIP_TEST = False
SCORE_THRESHOLD = 0.01    # The threshold of the probability of the classes
IOU_THRESHOLD = 0.45     # The threshold of the IOU when implement NMS

# compute environment
GPU = '1'

# name and path
DATASET_PATH = '/home/wz/doc/code/python_code/data/VOC'
ANNOT_DIR_PATH = 'data'
WEIGHTS_DIR = 'weights'
WEIGHTS_FILE = 'voc_fine_tune_initial.ckpt'
LOG_DIR = 'log'
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

