import numpy as np
import os
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import torch.nn.init

import tensorflow as tf



class GDL(nn.Module):
    def __init__(self):
        super(GDL, self).__init__()
        self.alpha = 1

    def forward(self, sr, hr):
        pos = tf.constant(np.identity(1), dtype=tf.float32)
        neg = -1 * pos
        filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
        filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
        strides = [1, 1, 1, 1]  # stride of (1, 1)
        padding = 'SAME'

        gen_dx = tf.abs(tf.nn.conv2d(gen_CT.cpu().numpy(), filter_x, strides, padding=padding))
        gen_dy = tf.abs(tf.nn.conv2d(gen_CT.cpu().numpy(), filter_y, strides, padding=padding))
        gt_dx = tf.abs(tf.nn.conv2d(gt_CT.cpu().numpy(), filter_x, strides, padding=padding))
        gt_dy = tf.abs(tf.nn.conv2d(gt_CT.cpu().numpy(), filter_y, strides, padding=padding))

        grad_diff_x = tf.abs(gt_dx - gen_dx)
        grad_diff_y = tf.abs(gt_dy - gen_dy)

        gdl=tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha))

        return gdl





    