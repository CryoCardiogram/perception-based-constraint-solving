import sys
import pathlib
root_path = pathlib.Path(__file__).parent.parent.resolve().__str__()
if root_path not in sys.path:
    sys.path.insert(0, root_path)
import torch.nn as nn
import torch
from constraint_solver.constraint_solver import ConstraintSolver
from pbr_module import PBR
from typing import Iterable
from collections import Counter 
import torch.nn.functional as F 


class PBCSweightedCE(PBR):
    def __init__(self, dnn: nn.Module, constraint_solver: ConstraintSolver, splitter:nn.Module=nn.Identity(),
                 lr=0.01, num_pred_classes=10, puzzle_shape=(9, 9), hparams=dict(), train_labels_weights: Iterable = None, ):
        super().__init__(dnn, constraint_solver, splitter, lr, num_pred_classes, puzzle_shape, hparams)

        self.use_weights = train_labels_weights is not None
        self.weighted_binary_ce_criterion = nn.BCELoss(weight=train_labels_weights)

    def training_step(self, batch, batch_idx):
        x, target = batch
        x = x.view(-1, *x.shape[-3:])
        loss = 0
        # for i in range(len(x)):
        dnn_output, cs_output = self(x)
        loss_dict = self.compute_loss(dnn_output, cs_output, target)
        loss += torch.stack([v for v in loss_dict.values()]).mean()  # TODO hyperparams
        return loss

    def compute_loss(self, dnn_output: dict, cs_output, target) -> dict:
        B = len(dnn_output['predictions'])
        target_dim = torch.tensor([B, self.n_items], device=self.device)

        # compute batch weights for each class
        counter = Counter(target['label'])
        classes_counts = torch.tensor([counter.get(k, 1) for k in range(self.num_pred_classes)], dtype=torch.float, device=self.device)
        train_class_weights = 1 - (classes_counts / classes_counts.sum())
        weighted_bce = nn.BCELoss(weight=train_class_weights if self.use_weights else None)
        cell_value_loss = weighted_bce(
            dnn_output['predictions'].view(*target_dim, -1),torch.eye(self.num_pred_classes, device=self.device)[target['label'].view(*target_dim)].float()
        )

        return {
            'cell_value_ce_loss': cell_value_loss
        }
