import sys
import pathlib

root_path = pathlib.Path(__file__).parent.parent.resolve().__str__()
if root_path not in sys.path:
    sys.path.insert(0, root_path)
import torch.nn as nn
import torch
from constraint_solver.constraint_solver import ConstraintSolver
from pbr_module import PBR
from neuralnet import get_calibration_wrapped_model, get_shared_calibration_wrapper
from typing import Iterable
from evaluation.calibration import evaluate_slim
from calibration import scaling
import torch.nn.functional as F
from calibration.visualization import ReliabilityDiagram, ConfidenceHistogram
import numpy as np
from utee.img_processing import debug_save_cellimgs
from utee import timeit
from neuralnet import SharedPatchNet
from custom_types import CalibrationType, BackboneArch, IS_MULTI


class Calibration(PBR):
    """
    Training then Calibration on validation set
    """

    def __init__(
        self,
        dnn: nn.Module,
        constraint_solver: ConstraintSolver,
        splitter: nn.Module = nn.Identity(),
        lr=0.01,
        num_pred_classes=10,
        puzzle_shape=...,
        hparams=...,
        calib_dl=None,
    ):
        super().__init__(
            dnn,
            constraint_solver,
            splitter,
            lr,
            num_pred_classes,
            puzzle_shape,
            hparams,
        )
        self.nll_criterion = nn.CrossEntropyLoss()  # required because input are logits
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.bce_criterion = nn.BCELoss()
        self.calib_dl = calib_dl
        self.calibrated = False
        self.calibration_method = hparams.get("calibration", None)
        self.puzzle_shape = torch.tensor(self.puzzle_shape)

    def on_train_end(self) -> None:
        print("debug methods calibration on train end...")
        self.dnn = get_calibration_wrapped_model(
            self.calibration_method, self.dnn, self.num_pred_classes
        )
        if self.calibration_method is not None:
            _ = scaling(self.calib_dl, self.dnn, 1e-3)
            self.calibrated = True

    def training_step(self, batch, batch_idx):
        self.dnn.eval()
        self.dnn.calibration.train()
        x, target = batch
        loss = 0
        dnn_output = self.dnn(x)
        loss_dict = self.compute_loss(dnn_output, None, target)
        loss += torch.stack([v for v in loss_dict.values()]).mean()
        return loss

    def forward(self, x, solve=False):
        # B x C x H x W
        x = x.view(-1, *x.shape[-3:])
        B = len(x)
        l = []
        patches = self.splitter(x)
        dnn_output = timeit(self.dnn.forward, l)(patches)
        dnn_output["pred-time"] = torch.Tensor(l)

        n_item = torch.cumprod(self.puzzle_shape, -1)[-1]
        if self.calibrated and self.calibration_method != CalibrationType.DIRICHLET:
            for head in self.dnn.calibration.keys():
                dnn_output[head] = F.softmax(
                    self.dnn.calibration[head](
                        dnn_output["logits"][head].reshape(B * n_item, -1)
                    ),
                    dim=1,
                )
        elif self.calibrated and self.calibration_method == CalibrationType.DIRICHLET:
            pass

        elif "predictions" not in dnn_output:
            for head in self.dnn.calibration.keys():
                dnn_output[head] = F.softmax(
                    dnn_output["logits"][head].reshape(B * n_item, -1), dim=1
                )
        cs_output = None
        if solve:
            cs_output = self.constraint_solver.solve_from_torch(
                dnn_output["predictions"].reshape(B, *self.puzzle_shape, -1),
                self.device,
            )
        return dnn_output, cs_output

    def compute_loss(self, dnn_output: dict, cs_output, target) -> dict:
        target_dim = target["label"].flatten().shape
        if "logits" in dnn_output:
            cell_value_loss = self.nll_criterion(
                dnn_output["logits"]["predictions"].view(*target_dim, -1),
                target["label"].flatten(),
            )
        else:
            cell_value_loss = self.bce_criterion(
                dnn_output["predictions"].view(*target_dim, -1),
                torch.eye(self.num_pred_classes)[target["label"].flatten()],
            )

        return {"cell_value_cross_entropy": cell_value_loss}

    def on_test_end(self) -> None:
        assert hasattr(self, "outputs")
        self.outputs = np.asarray(self.outputs)  # N x 81 x 10
        self.labels = np.asarray(self.labels)  # N x 81
        # path
        path = pathlib.Path(self.logger.log_dir)
        self.labels = self.labels.flatten()

        self.outputs = self.outputs.reshape(-1, self.outputs.shape[-1])
        _, reldia = ReliabilityDiagram().plot(self.outputs, self.labels, logits=False)
        reldia.savefig(path / f"rel_dia_label.png")
        _, confhist = ConfidenceHistogram().plot(
            self.outputs, self.labels, logits=False
        )
        confhist.savefig(path / f"conf_hist.png")

        for l in range(self.num_pred_classes):
            idx = self.labels == l
            # print('debug reldiag for class: ', l)
            # print('debug idx.all() == False?', (idx ==False).all())
            if (~idx).all():
                continue
            self.outputs = self.outputs.reshape(-1, self.outputs.shape[-1])
            _, reldia = ReliabilityDiagram().plot(
                self.outputs[idx], self.labels[idx], logits=False
            )
            reldia.savefig(path / f"rel_dia_label{l}.png")
            _, confhist = ConfidenceHistogram().plot(
                self.outputs[idx], self.labels[idx], logits=False
            )
            confhist.savefig(path / f"conf_hist{l}.png")

    def _shared_eval(self, batch, batch_idx, log=True, testing=False):
        x, target = batch
        x = x.view(-1, *x.shape[-3:])
        dnn_output, cs_output = self(x, solve=testing)
        eval_output = super()._shared_eval(
            batch,
            batch_idx,
            log=False,
            testing=testing,
            pre_dnn_output=dnn_output,
            pre_cs_output=cs_output,
            num_classes=self.num_pred_classes,
        )
        # calibration metrics
        cali_output = dict()
        str_eval_type = "test" if testing else "val"
        # dnn_output
        # if testing:
        n_items = torch.cumprod(self.puzzle_shape, -1)[-1]
        _, ece, ece_cw, mce, _, brier = evaluate_slim(
            dnn_output["predictions"].cpu().numpy().reshape(n_items, -1),
            target["label"].cpu().numpy().flatten(),
        )
        nll = self.compute_loss(dnn_output, None, target)
        cali_output["ece"] = ece
        cali_output["ece_cw"] = ece_cw
        cali_output["mce"] = mce
        cali_output["loss"] = nll["cell_value_cross_entropy"]
        cali_output["brier"] = brier
        for k, v in cali_output.items():
            new_k = f"{str_eval_type}_{k}"
            eval_output[new_k] = v

        if testing:
            if not hasattr(self, "outputs"):
                self.outputs = []
                self.labels = []
            self.outputs.append(
                dnn_output["predictions"].detach().cpu().numpy().reshape(n_items, -1)
            )
            self.labels.append(target["label"].cpu().numpy())

            target_shape = target["label"].flatten().shape
            labels_reduce = (
                target["label"].reshape(*self.puzzle_shape, -1).reshape(*target_shape)
            )
            pred_solve = torch.argmax(
                dnn_output["predictions"].reshape(*target_shape, -1), -1
            )
            gridacc = (
                (cs_output["perception"].reshape(*target_shape) == labels_reduce)
                .all()
                .sum()
            )
            if gridacc < 1:
                debug_save_cellimgs(
                    x, pathlib.Path(self.logger.log_dir, f"not-solved_{batch_idx}")
                )
                torch.save(
                    pred_solve,
                    pathlib.Path(self.logger.log_dir, f"not-solved_{batch_idx}.pt"),
                )

        for k, v in eval_output.items():
            if k.split("_")[0] in cali_output:
                self.log(k, v, prog_bar=True, on_step=False, on_epoch=True)
            else:
                self.log(k, v, prog_bar=False, on_step=False, on_epoch=True)

        return eval_output


class CalibrationOnly(Calibration):
    """
    Assume a pre-trained dnn. Only performs calibration step on validation set.
    """

    def __init__(
        self,
        dnn: nn.Module,
        constraint_solver: ConstraintSolver,
        splitter: nn.Module = nn.Identity(),
        lr=0.01,
        num_pred_classes=10,
        puzzle_shape=...,
        hparams=...,
        calib_dl=None,
    ):
        super().__init__(
            dnn,
            constraint_solver,
            splitter,
            lr,
            num_pred_classes,
            puzzle_shape,
            hparams,
            calib_dl,
        )
        # force eval routine to behave correctly
        self.calibrated = True
        for n, p in self.dnn.named_parameters():
            if "calibration" not in n:
                if hparams.get("freeze_level", 1) >= 1:
                    p.requires_grad = False
        ## assuming dnn is a sharedpatchnet
        # self.dnn = get_calibration_wrapped_model(self.calibration_method, self.dnn, self.num_pred_classes)
        assert isinstance(self.dnn, SharedPatchNet)

        self.dnn = get_shared_calibration_wrapper(
            self.calibration_method, self.dnn, self.num_pred_classes
        )

        self._debug_saved = False

    def on_train_end(self) -> None:
        # override call to scaling
        pass

    def training_step(self, batch, batch_idx):
        # self.dnn.eval()
        # self.dnn.calibration.train()
        x, target = batch
        x = x.view(-1, *x.shape[-3:])
        dnn_output, _ = self(x, False)
        target_shape = target["label"].flatten().shape

        loss = self.nll_criterion(
            self.dnn.calibration["predictions"](
                dnn_output["logits"]["predictions"].reshape(*target_shape, -1)
            ),
            target["label"].reshape(target_shape),
        )
        if self.hparams_initial["backbone"] in IS_MULTI:
            # build target
            label_style = (target["label"] >= 10).type(torch.float)
            # add loss term for style clf head
            loss += self.bce_logits(
                self.dnn.calibration["styles"](
                    dnn_output["logits"]["styles"].reshape(*target_shape, -1)
                ),
                label_style.reshape(*target_shape, -1),
            )
        return loss

    def configure_optimizers(self):
        print("debug config opt: ", self.calibration_method)
        if self.calibration_method == CalibrationType.TEMPERATURE:
            optimizer = torch.optim.LBFGS(
                self.dnn.calibration.parameters(), lr=self.lr, max_iter=500
            )
        elif self.calibration_method == CalibrationType.NONE:
            optimizer = torch.optim.SGD(self.dnn.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.SGD(self.dnn.calibration.parameters(), lr=self.lr)
        return optimizer
