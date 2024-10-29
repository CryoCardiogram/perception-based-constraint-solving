import os
import torchvision.models as models
from .svhn import load_conv_svhn
from .mnist import load_conv_digit_net
import torch
from torch import nn
import pathlib

root_path = pathlib.Path(__file__).parent.parent.resolve()


def resnet_feature_extract_layers(frozen=True):
    resnet50 = models.resnet50(pretrained=True)
    if frozen:
        for p in resnet50.parameters():
            p.requires_grad_(False)
    layers = list(resnet50.children())[:-1]
    layers += [nn.Flatten()]
    return layers


def resnet18_feature_extract_layers(frozen=True):
    resnet18 = models.resnet18(pretrained=True)
    if frozen:
        for p in resnet18.parameters():
            p.requires_grad_(False)
    layers = list(resnet18.children())[:-1]
    layers += [nn.Flatten()]
    return layers


def alexnet_feature_extract_layers(frozen=True):
    try:
        anet = models.AlexNet()
        state_dict = torch.load(
            os.path.join(root_path, "weights/alexnet.pth"),
            map_location=lambda storage, loc: storage,
        )
        anet.load_state_dict(state_dict)
    except Exception as e:
        print(e)
        anet = models.alexnet(pretrained=True)
    if frozen:
        for p in anet.parameters():
            p.requires_grad_(False)
    layers = list(anet.children())[:-1]
    layers += [nn.Flatten()]
    return layers


def svhn_feature_extract_layers(frozen=True):
    svhnnet = load_conv_svhn().encoder
    if frozen:
        for p in svhnnet.parameters(()):
            p.requires_grad_(False)
    layers = list(svhnnet.children())[:-1]  # remove top fc
    layers += [nn.Flatten()]
    return layers


def mnist_feature_extract_layers(frozen=True, reset=False):
    pretrain_weights_path = os.path.join(
        root_path, "weights/plattcalibrated-matrix_mnist_e15_lr1e-05_tl0.01.pt"
    )
    lenet = load_conv_digit_net(pretrain_weights_path, calibrated=False)
    if frozen:
        for p in lenet.parameters():
            p.requires_grad_(False)
    layers = [
        lenet.conv1,
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        lenet.conv2,
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(),
    ]
    if reset:
        lenet.conv1.reset_parameters()
        lenet.conv2.reset_parameters()
    return layers


def simclr_feature_extract_layers(frozen=True):
    try:
        state_dict = torch.load(
            os.path.join(root_path, "weights/rn50_simclr.pt"),
            map_location=lambda storage, loc: storage,
        )
    except:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt",
            model_dir=os.path.join(root_path, "weights"),
            map_location=lambda storage, loc: storage,
        )["state_dict"]
        torch.save(state_dict, "weights/rn50_simclr.pt")
    # retrieve keys in dict above beginning with 'encoder'
    # must strip the leading 'encoder' away to load weights correctly though..
    weights = {
        ".".join(k.split(".")[1:]): v for k, v in state_dict.items() if "encoder" in k
    }
    simclr_rn50 = models.resnet50(False)
    simclr_rn50.load_state_dict(weights)
    if frozen:
        for p in simclr_rn50.parameters():
            p.requires_grad_(False)
    layers = list(simclr_rn50.children())[:-1]
    layers += [nn.Flatten()]
    return layers
