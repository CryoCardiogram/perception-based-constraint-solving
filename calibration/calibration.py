#!/usr/bin/env python3
# Adapted from https://github.com/gpleiss/temperature_scaling
import torch
from torch import nn, optim
from torch.nn import functional as F
# from DFL.methods.spo import SPO
import numpy as np 
from tqdm.auto import tqdm, trange
from torchmetrics.classification.auroc import AUROC



    


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
        label = target['label']
        #inpt = inpt#.cuda()
        #print("into fc2 calibr: ", inpt.shape)
        logits_tmp = model.logits(inpt)
        logits_tmp = logits_tmp.detach()
        logits_list.append(logits_tmp)
        labels_list.append(label)
    yield torch.cat(logits_list)#.cuda()
    yield torch.cat(labels_list)#.cuda()
    

def set_temperature(valid_loader, model, tlr):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    #self.cuda()
    nll_criterion = nn.CrossEntropyLoss()#.cuda()
    ece_criterion = _ECELoss()#.cuda()
    # First: collect all the logits and labels for the validation set
    logits, labels = __logit_labels(valid_loader, model)

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    yield before_temperature_nll
    yield before_temperature_ece
    #print('Current temperature: %.3f' % model.calibration.temperature.item())
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([model.calibration.parameters()], lr=tlr, max_iter=500)
    #if optname == 'adam':
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
    #print('Optimal temperature: %.3f' % model.calibration.temperature.item())
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

def scaling(valid_loader, model, tlr, do_calibrate=True):
    model.eval()
    model.calibration.train()
    nll_criterion = nn.CrossEntropyLoss()
    ece_criterion = _ECELoss()
    logits, labels = __logit_labels(valid_loader, model)

    flat_shape = labels.reshape(-1).shape

    before_nll = nll_criterion(model.calibration(logits).reshape(*flat_shape, -1), labels.reshape(flat_shape))
    before_ece = ece_criterion(model.calibration(logits).reshape(*flat_shape, -1), labels.reshape(flat_shape))
    print('Before - NLL: %.3f, ECE: %.3f' % (before_nll, before_ece))
    
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

    after_nll = nll_criterion(model.calibration(logits).reshape(*flat_shape,-1), labels.reshape(flat_shape))
    after_ece = ece_criterion(model.calibration(logits).reshape(*flat_shape, -1), labels.reshape(flat_shape))
    # yield after_nll.item()
    # yield after_ece.item()
    print('After - NLL: %.3f, ECE: %.3f' % (after_nll, after_ece))
    
def spo_subgradient(y_hat, y_i, sol_true, solver):
    #assert y_hat.shape == y_i.shape, f'shape mismatch {y_hat.shape} vs {y_i.shape}'
    sol_spo = torch.from_numpy(onehot(solver((2*y_hat.view(-1) - y_i.view(-1)).view(9,9,-1).detach().numpy())[1]))
    print(np.argmax(sol_spo, -1))
    return 2*( sol_true - sol_spo) * -1 # maximization
    
def regret_score(y,y_pred, sol_trues, solver):
    total_loss =0 
    for i in range(len(y)):
        sol_true = sol_trues[i]
        sol_hat = onehot(solver(y_pred[i].reshape(9,9,-1).detach().numpy())[1])
        print(sol_hat.shape)
        print(sol_true.shape)
        print(y[i].shape)
        total_loss +=  ((sol_true.reshape(-1) - sol_hat.reshape(-1)).dot(y[i].reshape(-1)))
    return total_loss/len(y)

def onehot(grid):
    n = len(grid)
    one_hot_sol = np.zeros((n**2,10))
    one_hot_sol[np.arange(n**2),grid.reshape(-1)] = 1
    return one_hot_sol.reshape(n,n,10)

def get_solve_from_torch(solver):
    def solve_from_torch(torch_y):
        return torch.from_numpy(
            onehot(
                solver(torch_y.detach().numpy().reshape(9,9,-1))[1] 
            )
        ).flatten()
    return solve_from_torch

class SPO(torch.autograd.Function):
    """
        Implementation of SPO [1].
        [1] Elmachtoub, A. N., & Grigas, P. (2021). Smart “predict, then optimize”. Management Science.
        """
    @staticmethod
    def forward(ctx, input, y_true, sol_true, solver, is_minimization):
        """
        The forward pass computes and stores the solution for the SPO-perturbed cost vector (for the backward
        pass), and then computes and returns the relative regret of the predicted cost vector.

        :param ctx: the context object
        :param input: the predicted cost vector
        :param y_true: The true cost vector
        :param sol_true: The true solution
        :param optimization_problem: An OptimizationProblem object representing the parameterized optimization problem
        to solve
        :return: the regret of the predicted cost vector with respect to the ground-truth cost vector
        """
        mm = 1 if is_minimization else -1
        sol_hat = solver(input)
        print('sol hat: ', torch.argmax(sol_hat.view(9,9,-1), dim=-1))
        sol_spo = solver(2 * input - y_true)
        ctx.save_for_backward(sol_spo, sol_true, sol_hat, torch.tensor(mm))
        return mm * (sol_hat - sol_true).dot(y_true)

    @staticmethod
    def backward(ctx, grad_output):
        sol_spo, sol_true, sol_hat, mm = ctx.saved_tensors
        return int(mm) * 2 * (sol_true - sol_spo), None, None, None, None


def dfl_scaling(valid_loader, model, lr, solver, epochs=10, do_calibrate=True):
    # All solutions in one hot encoding as well
    model.eval()
    model.calibration.train()
    spo = SPO.apply
    auroc = AUROC(num_classes=10)

    optimizer = optim.AdamW(model.calibration.parameters(), lr=lr, weight_decay=0.001)
    first = True 

    #TODO precompute sols 
    torch_solver = get_solve_from_torch(solver)

    for _ in range(epochs):
        for batch in valid_loader:
            optimizer.zero_grad()
            x, y = batch
            y_hat = model(x).squeeze()

            sol_trues = []
            ys = []
            grads = []
            
            y_hats = []
            loss = 0
            for i in trange(len(y)):
                y_hat_i = y_hat[i].flatten()
                # Need one-hot encoding of y
                y_i = torch.from_numpy(onehot(y[i].reshape(9,9,))).flatten()
                print('y true: ', np.argmax(y_i.reshape(9,9,-1), -1))
                print('y pred: ', np.argmax(y_hat_i.detach().numpy().reshape(9,9,-1), -1))
                # no need to call solver, we ignored empty cells so y_i = sol_i
                sol_true_i = y_i#torch.from_numpy(onehot(solver(y_i)[1]))
                sol_trues.append(sol_true_i)
                ys.append(y_i)
                y_hats.append(y_hat_i)
                # grads  = spo_subgradient(y_hat_i, y_i, sol_true_i, solver)
                # grads.append(subgrad)
                loss += spo(y_hat_i, y_i, y_i, torch_solver, False)
            loss /= len(y)
            print('regret: ', loss)
            # y_hat.retain_grad()
            # y_hat.backward(gradient=torch.stack(grads,0).view(y_hat.shape))
            loss.backward()
            optimizer.step()

            # curve
            auc = auroc(y_hat.reshape(-1, y_hat.shape[-1]), y.int().reshape(-1))
            print('AUC: ', auc)
            # regret_loss = regret_score(ys, y_hats, ys, torch_solver)
            # print('regret: ', regret_loss)
    
    return None, None, None, None


