import pathlib
from abc import abstractmethod

import lightning as pl
import torch
from cpmpy.model import ExitStatus
from torch import nn

from constraint_solver.constraint_solver import ConstraintSolver
from evaluation import accuracy_scores, grid_accuracy_scores
from utee import timeit


class PBR(pl.LightningModule):
    """Abstract Class for Perception-based Reasoning Module"""

    def __init__(
        self,
        dnn: nn.Module,
        constraint_solver: ConstraintSolver,
        splitter: nn.Module = nn.Identity(),
        lr=1e-2,
        num_pred_classes=10,
        puzzle_shape=(9, 9),
        hparams=dict(),
    ):
        super().__init__()
        self.dnn = dnn
        self.constraint_solver = constraint_solver
        self.splitter = splitter
        self.lr = lr
        self.num_pred_classes = num_pred_classes
        self.puzzle_shape = torch.tensor(puzzle_shape)
        self.n_items = torch.cumprod(self.puzzle_shape, -1)[-1]
        self.solver_name = str(constraint_solver.__class__)
        self.save_hyperparameters("lr", "num_pred_classes", hparams)

    def forward(self, x, solve=False):
        # input of shape B x H x W x C
        x = x.view(-1, *x.shape[-3:])
        B = len(x)
        patches = self.splitter(x)
        l = []
        dnn_output = timeit(self.dnn.forward, l)(patches)
        dnn_output["pred-time"] = torch.tensor(l, device=self.device)
        cs_output = None
        if solve:
            cs_output = self.constraint_solver.solve_from_torch(
                dnn_output["predictions"].reshape(B, *self.puzzle_shape, -1),
                self.device,
            )
        return dnn_output, cs_output

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def compute_loss(self, dnn_output: dict, cs_output, target) -> dict:
        pass

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, testing=False)

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, testing=True)

    def _shared_eval(
        self,
        batch,
        batch_idx,
        log=True,
        testing=False,
        pre_dnn_output=None,
        pre_cs_output=None,
        num_classes=None,
    ):
        x, target = batch
        x = x.view(-1, *x.shape[-3:])
        dnn_output, cs_output = pre_dnn_output, pre_cs_output
        if dnn_output is None:
            dnn_output, cs_output = self(x, solve=testing)
        eval_output = dict()
        str_eval_type = "test" if testing else "val"
        eval_output[f"{str_eval_type}_pred_time"] = dnn_output["pred-time"]
        if testing or cs_output is not None:
            eval_output[f"{str_eval_type}_solve_time"] = cs_output["runtime"]

        for scorer in [accuracy_scores, grid_accuracy_scores]:
            stats = scorer(
                dnn_output,
                self.num_pred_classes if num_classes is None else num_classes,
                target,
                cs_output,
            )

            for k, v in stats.items():
                new_k = f"{str_eval_type}_{k}"
                eval_output[new_k] = v

        if testing:
            # handle unsat differently
            eval_output["unsolved"] = (
                (cs_output["status"] == ExitStatus.UNSATISFIABLE.value).float().sum()
            )
            if self.logger is not None:
                torch.save(
                    dnn_output,
                    pathlib.Path(
                        self.logger.log_dir,
                        f'test_{batch_idx}_solved_{eval_output["test_grid_accuracy_solve"]}.pt',
                    ),
                )
                torch.save(
                    target, pathlib.Path(self.logger.log_dir, f"target_{batch_idx}.pt")
                )
        for k, v in eval_output.items():
            if log:
                self.log(
                    k,
                    v,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                )
        # self.log_dict(eval_output)
        return eval_output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.dnn.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        pass
