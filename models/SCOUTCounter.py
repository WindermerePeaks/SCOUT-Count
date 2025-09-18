import torch
import torch.nn as nn
import torch.nn.functional as F


class SCOUTCount(nn.Module):
    def __init__(self, model_name):
        super(SCOUTCount, self).__init__()
        if model_name == "SCOUTCount":
            from .network import network as ccnet
        else:
            raise ValueError('Network ValueError.')

        self.CCN = ccnet()

        print("Model {} init success".format(model_name))
    
    def forward(self, img):
        pass

    def test_forward(self, img):                               
        density_map = self.CCN(img)
        return density_map

