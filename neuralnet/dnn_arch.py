"""
This file defines deep neural network (DNN) architectures used as perception layers in the hybrid PBCS system.

DNNs can operate at single-item level (cell) or at multi-item level (whole image), which is reflected by n.

The output of all forward passes is a dict, which can contain the following keys:

    - predictions: (n x C) probability score tensor for each class
    - logits: (n x C) logits score tensor before pass through softmax
    - styles: (n) [O,1] score tensor for each font

"""

import numpy as np
import torch
from torch import nn

from .generic_layers import (
    make_conv_layers,
    make_fc_layers,
    make_fc_layers_global_pooling,
)


class CellNet(nn.Module):
    def __init__(self, feat_ext: nn.Module, classifier: nn.Module):
        super(CellNet, self).__init__()
        self.feat_extract = feat_ext
        self.classifier = classifier

    def forward(self, x):
        h = self.feat_extract(x)
        return {
            "predictions": self.classifier(h),
            "embeddings": h,
        }


class CellStyleNet(CellNet):
    """
    Multi-head network for Sudoku.
    `classifier` module predicts number (or blank)
    `style_clf` module classifies between printed and handwritten.


    """

    def __init__(
        self,
        feat_ext: nn.Module,
        classifier: nn.Module,
        style_clf: nn.Module = nn.Identity(),
    ) -> None:
        """

        Args:
            feat_ext (nn.Module): backbone cnn
            classifier (nn.Module): digit classifier
            style_clf (nn.Module, optional): font style classifier. Defaults to nn.Identity() to disable style classification.
        """
        super().__init__(feat_ext, classifier)
        self.style_clf = style_clf

    def forward(self, x):
        h = self.feat_extract(x)
        return {
            "predictions": self.classifier(h),
            # if no style_clf, this contains feature maps
            "styles": self.style_clf(h),
            "embeddings": h,
        }


class PlattCellNet(CellNet):
    """
    Platt calibration wrapper

    `calibration` layer learns a mapping from logits to calibrated input for the softmax
    """

    def __init__(
        self,
        feat_ext: nn.Module,
        classifier: nn.Module,
        calibration_layer: nn.Module,
        style_clf: nn.Module = nn.Identity(),
        style_calibration: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__(feat_ext, classifier)
        self.calibration_digit = calibration_layer
        self.clf_layers = nn.Sequential(*list(self.classifier.children())[:-1])
        self.style_clf = style_clf
        self.style_clf_layers = nn.Sequential(*list(self.style_clf.children())[:-1])
        # let's calibrate style_clf with temp scaling by default
        self.calibration_style = style_calibration
        # need a calibration attribute for backward compat
        self.calibration = nn.ModuleDict(
            {"predictions": self.calibration_digit, "styles": self.calibration_style}
        )

    def logits(self, x):
        """
        Remove softmax
        """
        h = self.feat_extract(x)
        return {
            "predictions": self.clf_layers(h),
            "styles": self.style_clf_layers(h),
            "embeddings": h,
        }

    def forward(self, x):
        out_logits = self.logits(x)
        h1 = out_logits["predictions"]
        h2 = out_logits["styles"]
        output_dict = {
            "logits": out_logits,
            "predictions": nn.Softmax(-1)(self.calibration_digit(h1)),
            "styles": nn.Sigmoid()(self.calibration_style(h2)),
        }
        # hcal = self.calibration(h)
        # output_dict.update({'predictions': self.clf_layers[-1](hcal)})
        return output_dict


class SharedPatchNet(nn.Module):
    """Wrapper for cell-level DNN architectures used over a single item, in a batch."""

    def __init__(self, cell_net: CellNet) -> None:
        super().__init__()
        self.cell_net = cell_net

    def forward(self, x):
        """Expect input of the form B x O x F_1 (x F2, ...),
        where O is the size of the optimization problem, B is the batch size, F_i are remaining dimensions for feature vectors.
        """
        B, O = x.shape[:2]
        # single output over every item in every batch
        out = self.cell_net(x.flatten(0, 1))
        # convert to B x (O * K)
        output_dict = {k: v.view(B, -1) for k, v in out.items()}
        return output_dict


class SharedPatchNetCal(SharedPatchNet):
    def __init__(self, cell_net: PlattCellNet) -> None:
        super().__init__(cell_net)

    @property
    def calibration(self):
        return self.cell_net.calibration

    def logits(self, x):
        B = x.shape[0]
        P_S = x.shape[1]
        out_logits = self.cell_net.logits(x.flatten(0, 1))
        for n, h_tensor in out_logits.items():
            out_logits[n] = h_tensor.view(B, P_S, -1)
        return out_logits

    def forward(self, x):
        logits_output = self.logits(x)
        h = logits_output["predictions"]
        B = h.shape[0]

        if any(torch.isnan(p).any() for p in self.parameters()):
            print("nans in the weights!")
            print(
                "following layers are impacted:",
                [n for n, p in self.named_parameters() if torch.isnan(p).any()],
            )
        output_dict = {
            "logits": logits_output,  # B x (0*K)
            "predictions": self.calibration["predictions"](h).softmax(-1),
        }
        if "styles" in logits_output:
            output_dict["styles"] = self.calibration["styles"](
                logits_output["styles"]
            ).sigmoid()

        return output_dict


class Print(nn.Module):
    # debug
    def __init__(self, txt=""):
        super(Print, self).__init__()
        self.txt = txt

    def forward(self, x):
        print(f"{self.txt}{x.shape}")
        print("requires grad", x.requires_grad)
        return x


class MultiItemNet(nn.Module):
    """Wrapper for cell-level DNN architecutre used jointly over multiple items."""

    def __init__(self, cellnet: CellNet):
        raise NotImplementedError


class WholeImageCNN(nn.Module):
    """Generalized 5-layers CNN, similar to https://github.com/Kyubyong/sudoku or
    modified from SudokuNet used in NeurASP.
    """

    def __init__(self, grid_shape=(9, 9), n_classes=10) -> None:
        super().__init__()
        self.grid_shape = grid_shape
        self.n_classes = n_classes
        whole_image_backbone_config = [
            (32, 4, 2, 0),
            (64, 3, 2, 0),
            (128, 3, 2, 0),
            (256, 2, 1, 0),
            (512, 2, 1, 0),
        ]
        conv_layers = make_conv_layers(
            whole_image_backbone_config,
            in_channels=1,
            p=0.25,
            pool_ks=3,
            pool_str=3,
            batch_norm=True,
        )
        out_layers = make_fc_layers_global_pooling(
            in_dim=512, out_shape=grid_shape, num_classes=n_classes
        )
        # out_layers += [nn.Flatten(1)]
        # backbone
        self.feat_extract = nn.Sequential(*conv_layers)
        # classifier
        self.classifier = nn.Sequential(*out_layers, nn.Softmax(-1))

    def forward(self, x):
        h = self.feat_extract(x)
        return {"predictions": self.classifier(h)}


def get_neurasp_arch():
    import torch.nn.functional as F

    class Sudoku_Net(nn.Module):
        def __init__(self):
            super(Sudoku_Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2)
            self.conv1_bn = nn.BatchNorm2d(32)
            self.dropout1 = nn.Dropout(p=0.25)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
            self.conv2_bn = nn.BatchNorm2d(64)
            self.dropout2 = nn.Dropout(p=0.25)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
            self.conv3_bn = nn.BatchNorm2d(128)
            self.dropout3 = nn.Dropout(p=0.25)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=1)
            self.conv4_bn = nn.BatchNorm2d(256)
            self.dropout4 = nn.Dropout(p=0.25)
            self.conv5 = nn.Conv2d(256, 512, kernel_size=2, stride=1)
            self.conv5_bn = nn.BatchNorm2d(512)
            self.dropout5 = nn.Dropout(p=0.25)

            self.maxpool = nn.MaxPool2d(3)
            self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((9, 9))

            self.conv1x1_1 = nn.Conv2d(in_channels=512, out_channels=10, kernel_size=1)
            self.conv1x1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
            self.conv1x1_3 = nn.Conv2d(in_channels=512, out_channels=10, kernel_size=1)

            self.fc1 = nn.Linear(41472, 81 * 10)
            self.dropout5 = nn.Dropout(p=0.25)

        def forward(self, x):
            x = self.dropout1(self.conv1_bn(self.conv1(x)))
            x = F.relu(x)
            x = self.dropout2(self.conv2_bn(self.conv2(x)))
            x = F.relu(x)
            x = self.dropout3(self.conv3_bn(self.conv3(x)))
            x = F.relu(x)
            x = self.dropout4(self.conv4_bn(self.conv4(x)))
            x = F.relu(x)
            x = self.dropout5(self.conv5_bn(self.conv5(x)))
            x = F.relu(x)
            x = self.maxpool(x)
            x = self.conv1x1_1(x)
            x = nn.Softmax(1)(x)
            batch_size = len(x)
            x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, 810)
            x = x.view(batch_size, 81, 10)
            return x

    return Sudoku_Net()


class FullImageCNN(nn.Module):
    def __init__(self, backbone_layers: list, output_layers: list) -> None:
        super().__init__()
        self.feat_extract = nn.Sequential(*backbone_layers)
        self.classifier = nn.Sequential(*output_layers)
        self.softmax = nn.Softmax(-1)

    def logits(self, x):
        h = self.feat_extract(x)
        return {"logits": self.classifier(h)}

    def forward(self, x):
        out = self.logits(x)
        out["predictions"] = self.softmax(out["logits"])
        return out
