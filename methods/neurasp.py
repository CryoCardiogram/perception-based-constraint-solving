import sys
import pathlib

root_path = pathlib.Path(__file__).parent.parent.resolve().__str__()
if root_path not in sys.path:
    sys.path.insert(0, root_path)
import torch.nn as nn
import torch
from constraint_solver.sudoku_solver import SudokuSolverNeurASP
from pbr_module import PBR
from typing import Iterable
from collections import Counter
from neurasp import NeurASP

from lightning.pytorch.utilities import grad_norm


class NeurASPModule(PBR):
    def __init__(
        self,
        dnn: nn.Module,
        constraint_solver: SudokuSolverNeurASP,
        splitter: nn.Module,
        lr=1e-2,
        num_pred_classes=10,
        puzzle_shape=(9, 9),
        hparams=dict(),
    ):
        super().__init__(
            dnn,
            constraint_solver,
            splitter,
            lr=lr,
            num_pred_classes=num_pred_classes,
            puzzle_shape=puzzle_shape,
            hparams=hparams,
        )

        self.neurasp = NeurASP(
            constraint_solver.dprogram + constraint_solver.neural_rule,
            {"identify": self.dnn},
            {"identify": self.configure_optimizers()},
        )
        self.solver_name = "ASP"
        self.constraint_solver.neurasp_obj = self.neurasp

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.dnn, norm_type=2)
        self.log_dict(norms)

    def training_step(self, batch, batch_idx):
        x, target = batch
        x = x.view(-1, *x.shape[-3:])
        B = len(x)
        patches = self.splitter(x)
        dataList = []
        obsList = []
        tmp = (patches, {"identify": target["label"]})
        dataList.append({"img": tmp})
        obsList.append("")
        # TODO: probably only works with grid CNN
        self.neurasp.learn(
            dataList=dataList,
            obsList=obsList,
            alpha=0,
            epoch=1,
            lossFunc="cross",
            method="nn_prediction",
            bar=False,
        )
