import numpy as np
import os
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import torch.nn.init

from keras import backend as K



class GDL(nn.Module):
    def __init__(self):
        super(GDL, self).__init__()
        self.alpha = 1

    def forward(self, sr, hr):
        # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
        y_true = K.batch_flatten(hr)
        y_pred = K.batch_flatten(hr)
        Y_true = K.reshape(y_true, (-1, ) + img_shape).cpu().numpy()
        Y_pred = K.reshape(y_pred, (-1, ) + img_shape).cpu().numpy()
        t1 = K.pow(K.abs(Y_true[:, :, 1:, :] - Y_true[:, :, :-1, :]) -
                   K.abs(Y_pred[:, :, 1:, :] - Y_pred[:, :, :-1, :]), alpha).cpu().numpy()
        t2 = K.pow(K.abs(Y_true[:, :, :, :-1] - Y_true[:, :, :, 1:]) -
                   K.abs(Y_pred[:, :, :, :-1] - Y_pred[:, :, :, 1:]), alpha).cpu().numpy()
        out = K.mean(K.batch_flatten(t1 + t2), -1).cpu().numpy()
        
        return out




    