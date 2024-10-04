import torch
import numpy as np
from torchmetrics import AUROC, Accuracy, ConfusionMatrix


def get_cell_scorer(metric_cls, name):
    @torch.no_grad()
    def metric_scores(dnn_output: dict, num_classes,
                      target: dict, cs_output: dict = None):
        get_score = metric_cls(
            num_classes=num_classes,
            task='multiclass' if num_classes > 2 else 'binary',
            average='weighted').to(dnn_output['predictions'].device)
        # B x (9*9)
        target_shape = target['label'].flatten(-1).shape
        # before solve
        cell_metric = get_score(dnn_output['predictions'].detach().reshape(
            *target_shape), target['label'].reshape(*target_shape))
        res = {
            f'cell_{name}': cell_metric,
        }
        # after solve
        if cs_output is not None:
            cell_metric_solve = get_score(
                cs_output["perception"].reshape(
                    *target_shape),
                target['label'].reshape(*target_shape))

            res[f'cell_{name}_solve'] = cell_metric_solve

        return res

    return metric_scores



@torch.no_grad()
def grid_accuracy_scores(dnn_output: dict, num_classes,
                         target: dict, cs_output: dict = None):
    grid_acc = Accuracy(
        task='multiclass',
        num_classes=num_classes,
        average='micro'  # should sum over all cells?
    )
    # B x (9*9)
    target_shape = target['label'].flatten(-1).shape
    # B x (9*9) x K to B x (9*9)
    preds = dnn_output['predictions'].detach().reshape(*target_shape, -1).argmax(-1).float()

    grid_acc_val = torch.eq(preds, target['label'].reshape(*target_shape)).all(-1).float()
    res = {
        'grid_accuracy': grid_acc_val.mean()
    }
    if cs_output is not None:
        # only compare on non-blank?
        # pred_solve = np.argmax(cs_output['perception_onehot'].reshape(*target_shape, -1), -1)
        # grid_acc_val_solve = np.equal(pred_solve, target['label'].reshape(*target_shape).numpy()).all().sum()
        # directly compare full solutions
        grid_acc_val_solve = torch.eq(cs_output['solution'].reshape(*target_shape).long(), target['cs_output']['solution'].reshape(*target_shape).long()).all(-1).float()
        res['grid_accuracy_solve'] = grid_acc_val_solve.mean()
    return res

@torch.no_grad()
def cell_accuracy_scores(dnn_output: dict, num_classes,
                         target: dict, cs_output: dict = None):

    target_shape = target['label'].flatten(-1).shape
    preds = dnn_output['predictions'].detach().reshape(*target_shape, -1).argmax(-1).float()
    accuracy = Accuracy(
        task='multiclass',
        num_classes=num_classes
    ).to(dnn_output['predictions'].device)
    # cell_acc_val = torch.eq(preds, target['label'].reshape(*target_shape)).sum(-1) / preds.flatten(-1).shape[-1]
    cell_acc_val = accuracy(preds, target['label'].reshape(*target_shape))
    res = {
        'cell_accuracy': cell_acc_val
    }
    if cs_output is not None:
        
        # compare full solutions
        # cell_acc_solve = torch.eq(cs_output['solution'].reshape(-1).long(), target['cs_output']['solution'].reshape(-1).long()).sum(-1) / preds.flatten(-1).shape[-1]
        cell_acc_solve = accuracy(cs_output['solution'].reshape(*target_shape).long(),  target['cs_output']['solution'].reshape(*target_shape).long())
        res['cell_accuracy_solve'] = cell_acc_solve

        # perception_acc = torch.eq(cs_output['perception'].reshape(*target_shape), target['label'].reshape(*target_shape)).sum(-1) / preds.flatten(-1).shape[-1]
        perception_acc = accuracy(cs_output['perception'].reshape(*target_shape), target['label'].reshape(*target_shape))
        res['perception_accuracy'] = perception_acc
    return res
