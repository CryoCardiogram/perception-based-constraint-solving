#!/usr/bin/env python3
# Adapted from https://github.com/gpleiss/temperature_scaling
# from DFL.methods.spo import SPO
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics.classification.auroc import AUROC
from tqdm.auto import tqdm, trange


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=softmaxes.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def __logit_labels(valid_loader, model):
    logits_list = []
    labels_list = []

    for inpt, target in valid_loader:
        label = target["label"]
        # inpt = inpt#.cuda()
        # print("into fc2 calibr: ", inpt.shape)
        logits_tmp = model.logits(inpt)
        logits_tmp = logits_tmp.detach()
        logits_list.append(logits_tmp)
        labels_list.append(label)
    yield torch.cat(logits_list)  # .cuda()
    yield torch.cat(labels_list)  # .cuda()


def set_temperature(valid_loader, model, tlr):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    # self.cuda()
    nll_criterion = nn.CrossEntropyLoss()  # .cuda()
    ece_criterion = _ECELoss()  # .cuda()
    # First: collect all the logits and labels for the validation set
    logits, labels = __logit_labels(valid_loader, model)

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    yield before_temperature_nll
    yield before_temperature_ece
    # print('Current temperature: %.3f' % model.calibration.temperature.item())
    print(
        "Before temperature - NLL: %.3f, ECE: %.3f"
        % (before_temperature_nll, before_temperature_ece)
    )

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([model.calibration.parameters()], lr=tlr, max_iter=500)
    # if optname == 'adam':
    #    optimizer = optim.Adam([model.calibration.temperature], lr=tlr)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(model.calibration(logits), labels)
        loss.backward()
        return loss

    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(model.calibration(logits), labels).item()
    after_temperature_ece = ece_criterion(model.calibration(logits), labels).item()
    yield after_temperature_nll
    yield after_temperature_ece
    # print('Optimal temperature: %.3f' % model.calibration.temperature.item())
    print(
        "After temperature - NLL: %.3f, ECE: %.3f"
        % (after_temperature_nll, after_temperature_ece)
    )


def scaling(valid_loader, model, tlr, do_calibrate=True):
    model.eval()
    model.calibration.train()
    nll_criterion = nn.CrossEntropyLoss()
    ece_criterion = _ECELoss()
    logits, labels = __logit_labels(valid_loader, model)

    flat_shape = labels.reshape(-1).shape

    before_nll = nll_criterion(
        model.calibration(logits).reshape(*flat_shape, -1), labels.reshape(flat_shape)
    )
    before_ece = ece_criterion(
        model.calibration(logits).reshape(*flat_shape, -1), labels.reshape(flat_shape)
    )
    # print('Before - NLL: %.3f, ECE: %.3f' % (before_nll, before_ece))

    # yield before_nll.item()
    # yield before_ece.item()

    if do_calibrate:
        optimizer = optim.LBFGS(model.calibration.parameters(), lr=tlr, max_iter=200)

        # optimizer = optim.SGD(model.calibration.parameters(), lr=tlr, weight_decay=0.001)
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(model.calibration(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        # for e in range(500):
        #     optimizer.zero_grad()
        #     loss = nll_criterion(model.calibration(logits).reshape(*flat_shape,-1), labels.reshape(flat_shape))
        #     loss.backward()
        #     optimizer.step()

    after_nll = nll_criterion(
        model.calibration(logits).reshape(*flat_shape, -1), labels.reshape(flat_shape)
    )
    after_ece = ece_criterion(
        model.calibration(logits).reshape(*flat_shape, -1), labels.reshape(flat_shape)
    )
    # yield after_nll.item()
    # yield after_ece.item()
    # print('After - NLL: %.3f, ECE: %.3f' % (after_nll, after_ece))
