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
        
        t1 = K.pow(torch.abs(Y_true[:, :, :, 1:, :] - Y_true[:, :, :, :-1, :]) -
                   K.abs(Y_pred[:, :, :, 1:, :] - Y_pred[:, :, :, :-1, :]), alpha)
        t2 = K.pow(torch.abs(Y_true[:, :, :, :, :-1] - Y_true[:, :, :, :, 1:]) -
                   K.abs(Y_pred[:, :, :, :, :-1] - Y_pred[:, :, :, :, 1:]), alpha)

        return gdl





    