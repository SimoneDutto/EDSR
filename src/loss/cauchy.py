import torch
import torch.nn as nn

class Cauchy(nn.Module):

    def __init__(self):
        super(Cauchy, self).__init__()
        self.c = 1.0

    def forward(self, X, Y):
        r = torch.add(X, -Y)
        ra = torch.abs(r)
        error = 0.5 * (self.c **2) * torch.log(1 + (ra/self.c)**2)
        
        loss = torch.sum(error)
        return loss 