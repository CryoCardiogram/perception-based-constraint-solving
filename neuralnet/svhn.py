# https://github.com/aaron-xichen/pytorch-playground/blob/master/svhn/model.py
import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
root_path = Path(__file__).parent.parent.resolve()

class SVHN(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(SVHN, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SVHNNet(nn.Module):
    def __init__(self, in_size=32):
        super(SVHNNet, self).__init__()
        n_channel = in_size
        cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
        layers = make_layers(cfg, batch_norm=True)
        # https://github.com/aaron-xichen/pytorch-playground/blob/master/svhn/model.py
        self.encoder = SVHN(layers, n_channel=8*n_channel, num_classes=10)

    def forward(self, x):
        x = self.encoder(x)
        return F.log_softmax(x, dim=1)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU(), nn.Dropout(0.3)]
            else:
                layers += [conv2d, nn.ReLU(), nn.Dropout(0.3)]
            in_channels = out_channels
    return nn.Sequential(*layers)


def load_conv_svhn(model_path=os.path.join(root_path, 'weights/svhn_pre.pt')):
    n_channel = 32
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    weights = torch.load(model_path, map_location=lambda storage,loc:storage)
    pretrained = SVHNNet(n_channel)
    pretrained.load_state_dict(weights, assign=True)
    
    return pretrained.eval()