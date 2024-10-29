import torch
import torch.nn as nn
import math
from .generic_layers import make_fc_layers
from einops.layers.torch import Rearrange
from einops import rearrange
from enum import Enum


class Temperature(nn.Module):
    """
    A decorator for the `Temperature layer` taking as input logits of the model
    """

    def __init__(self):
        super(Temperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        # logits = self.model(inpt)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature

    def name(self):
        return "Temperature scaling"


class Platt(nn.Module):
    """
    Multi-class scaling layer (Matrix Scaling)
    """

    def __init__(self, out_size=10):
        super(Platt, self).__init__()
        self.fc = nn.Linear(out_size, out_size)
        for name, param in self.named_parameters():
            if "conv" in name and "weight" in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2.0 / n))
            elif "norm" in name and "weight" in name:
                param.data.fill_(1)
            elif "norm" in name and "bias" in name:
                param.data.fill_(0)
            elif "classifier" in name and "bias" in name:
                param.data.fill_(0)

    def forward(self, logits):
        return self.fc(logits)

    def name(self):
        return "Matrix scaling"


class PlattDiag(nn.Module):
    def __init__(self, out_size=10):
        super(PlattDiag, self).__init__()
        weights = nn.init.xavier_uniform_(torch.diag_embed(torch.ones(out_size))).diag()
        self.W = nn.Parameter(weights)
        self.b = nn.Parameter(torch.ones(out_size))

    def forward(self, logits):
        return logits.matmul(torch.diag_embed(self.W)) + self.b

    def name(self):
        return "Vector scaling"


def create_output_layer(
    output_layer_type: Enum,
    output_layer_config: list,
    n_items: int,
    enable_batch_norm: bool,
    num_pred_classes: int,
):
    """Helper function to create output layer on top of backbone.
    The output layer takes unormalized logits scores for each class as input, and learns to transform it before passing it to a softmax function.

    The resulting layer assumes input in the format `B` x (`n_item` x `num_class`)

    Args:
        output_layer_type (Enum): hyperparams. Should contain the type of output layer (in ´output_layer_type`)
        output_layer_config (list): hidden layers cfg within the output layer. Input and output size
            are pre-determined depending on the selected architecture.
        n_items (int): number of problem items (81 cells in visual sudoku for example)
        enable_batch_norm (bool): flag to enable/disable batch normalization layer
        num_pred_classes (int): number of labels

    Returns:
        torch.nn.Module: A trainable output layer, with a softmax activation
    """
    if output_layer_type is None:
        return nn.Sequential(
            Rearrange(
                "batch (n_item n_class) -> (batch n_item) n_class",
                n_class=num_pred_classes,
            ),
            nn.Softmax(-1),
            Rearrange("(batch n_item) n_class -> batch n_item n_class", n_item=n_items),
        )
    if output_layer_type.value == "CORR":
        return nn.Sequential(
            *make_fc_layers(
                output_layer_config + [n_items * num_pred_classes],
                in_dim=n_items * num_pred_classes,
                bn=enable_batch_norm,
                p=0.2,
                out=nn.Identity,
            ),
            Rearrange(
                "b (n_items n_pred_class) -> b n_items n_pred_class",
                n_items=n_items,
                n_pred_class=num_pred_classes,
            ),
            nn.Softmax(-1),
        )
    elif output_layer_type.value == "SHARED":
        return nn.Sequential(
            Rearrange(
                "batch (n_item n_class) -> (batch n_item) n_class",
                n_class=num_pred_classes,
            ),
            *make_fc_layers(
                output_layer_config + [num_pred_classes],
                in_dim=num_pred_classes,
                p=0.2,
                bn=enable_batch_norm,
                out=nn.Identity,
            ),
            Rearrange(
                "(b n_items) n_pred_class -> b n_items n_pred_class", n_items=n_items
            ),
            nn.Softmax(-1),
        )
    elif output_layer_type.value == "PATCH":

        class PatchSpecificOutputNN(nn.Module):
            def __init__(self):
                super(PatchSpecificOutputNN, self).__init__()
                self.out_layers = nn.ModuleList(
                    [
                        nn.Sequential(
                            *make_fc_layers(
                                output_layer_config + [num_pred_classes],
                                in_dim=num_pred_classes,
                                p=0.2,
                                out=nn.Identity,
                            )
                        )
                        for _ in range(n_items)
                    ]
                )

            def forward(self, x):
                h = rearrange(x, "b (n k) -> n b k", n=n_items)
                return rearrange(
                    [self.out_layers[i](inpu) for i, inpu in enumerate(h)],
                    "n b k -> b n k",
                    n=n_items,
                )

        return nn.Sequential(PatchSpecificOutputNN(), nn.Softmax(-1))
    else:
        # rearrange and then softmax

        raise NotImplementedError
