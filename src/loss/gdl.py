import torch
import torch.nn as nn
import numpy as np


class GDL(nn.Module):

    def __init__(self):
        super(GDL, self).__init__()
        self.alpha = 2

    def forward(self, Y_true, Y_pred):

        Y_trueR = torch.from_numpy(np.flip(Y_true.detach().cpu().numpy(), 2).copy()).cuda()
        Y_predR = torch.from_numpy(np.flip(Y_pred.detach().cpu().numpy(), 2).copy()).cuda()

        t1 = torch.pow(torch.abs(Y_true - Y_trueR) - torch.abs(Y_pred - Y_predR), self.alpha)

        Y_trueR = torch.from_numpy(np.flip(Y_true.detach().cpu().numpy(), 3).copy()).cuda()
        Y_predR = torch.from_numpy(np.flip(Y_pred.detach().cpu().numpy(), 3).copy()).cuda()

        t2 = torch.pow(torch.abs(Y_trueR - Y_true) - torch.abs(Y_predR - Y_pred), self.alpha)

        loss = torch.mean((t1 + t2).reshape(-1), -1)
        
        return loss 

