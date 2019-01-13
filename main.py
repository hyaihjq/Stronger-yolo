# coding: utf-8

import tensorflow as tf
import numpy as np


def valid_scale_filter(bboxes, valid_scale):
    print bboxes
    bboxes_scale = np.sqrt(np.multiply.reduce(bboxes[:, 2:4] - bboxes[:, 0:2], axis=-1))
    print bboxes_scale
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
    print scale_mask
    bboxes = bboxes[scale_mask]
    return bboxes

if __name__ == '__main__':
    bboxes = np.array([[0, 0, 29, 29], [10, 10, 35, 45]])
    valid_scale = np.array([0, 30])
    bboxes = valid_scale_filter(bboxes, valid_scale)
    print bboxes