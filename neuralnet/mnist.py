#!/usr/bin/env python3
import torch 
import torch.nn as nn
import torch.nn.functional as F
from .calibration_layers import Platt, PlattDiag, Temperature
#from .generic_layers import make_fc_layers, make_conv_layers
        
#TODO use generic-layers and pre-train new model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class LeNetCal(LeNet):
    def __init__(self, scaling, out_size):
        super().__init__()
        if 'matrix' in scaling:
            self.calibration = Platt()
        elif 'diag' in scaling:
            self.calibration = PlattDiag()
        else:
            self.calibration = Temperature()

    def forward(self, x):
        x = self.logits(x)
        x = self.calibration(x)
        return F.log_softmax(x, dim=1)
    
    def logits(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def load_from_lenet(self, m:LeNet):
        self.conv1.load_state_dict(m.conv1.state_dict())
        self.conv2.load_state_dict(m.conv2.state_dict())
        self.fc1.load_state_dict(m.fc1.state_dict())
        self.fc2.load_state_dict(m.fc2.state_dict())

        
def load_conv_digit_net(model_path, calibrated=False):
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('debug load_conv_digit_net calibrated: ', calibrated)
    # if calibrated:
    #     nnet = LeNetCal([method for method in ['matrix', 'diag', 'temp'] if method in model_path][0], out_size=10)
    # else:
    #     return LeNet()
    nnet = LeNet()
    incompat_keys = nnet.load_state_dict(state_dict, strict=False)
    print(incompat_keys)
    return nnet

 