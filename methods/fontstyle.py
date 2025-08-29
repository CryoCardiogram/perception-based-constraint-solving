import pathlib
import sys

root_path = pathlib.Path(__file__).parent.parent.resolve().__str__()
if root_path not in sys.path:
    sys.path.insert(0, root_path)
import logging
from collections import Counter
from typing import Iterable

import torch
import torch.nn as nn
from torchmetrics import F1Score

from constraint_solver.constraint_solver import ConstraintSolver
from evaluation import accuracy_scores, get_cell_scorer
from methods.pbcs import PBCSweightedCE
from neuralnet.dnn_arch import CellStyleNet
from pbr_module import PBR
from utee import timeit
from utee.img_processing import debug_save_cellimgs

logger = logging.getLogger(__file__)


def reduce_to_ten_labels(labels: torch.Tensor):
    reduced_labels = torch.zeros(*labels.shape, device=labels.device).long()
    reduced_labels[labels < 10] = labels[labels < 10]
    reduced_labels[labels >= 10] = labels[labels >= 10] - 9
    return reduced_labels


def reduce_to_two(labels: torch.Tensor):
    reduced_labels = torch.zeros(*labels.shape, device=labels.device).long()
    reduced_labels[labels < 10] = 0
    reduced_labels[labels >= 10] = 1
    return reduced_labels


class FontStyle(PBCSweightedCE):
    def __init__(
        self,
        dnn: nn.Module,
        constraint_solver: ConstraintSolver,
        splitter: nn.Module = ...,
        lr=0.01,
        num_pred_classes=10,
        puzzle_shape=...,
        hparams=...,
        train_labels_weights: Iterable = None,
    ):
        super().__init__(
            dnn,
            constraint_solver,
            splitter,
            lr,
            num_pred_classes,
            puzzle_shape,
            hparams,
            train_labels_weights,
        )
        self.font_f1_score = F1Score("binary", average="weighted")

    def forward(self, x, solve=False):
        # return super().forward(x, solve)
        dnn_output, cs_output = super().forward(x, False)
        n_items = torch.cumprod(self.puzzle_shape, -1)[-1]
        B = len(x)
        # B x O x 19
        probs = dnn_output["predictions"].reshape(B, n_items, -1)
        # separate digit value and font clf
        # B X O x 10
        reduced_digit_probs = torch.zeros(*probs.shape[:-1], 10).to(self.device)
        reduced_digit_probs[:, :, 0] = probs[:, :, 0]
        reduced_digit_probs[:, :, 1:] = probs[:, :, 1:10] + probs[:, :, 10:]
        probs_hw = torch.zeros(*probs.shape[:-1])
        probs_hw = probs[:, :, 10:].sum(-1)
        dnn_output["predictions_digit"] = reduced_digit_probs.detach()
        dnn_output["predictions_style"] = probs_hw.detach()

        if solve:
            pred_key = (
                "predictions_merged"
                if "predictions_merged" in dnn_output
                else "predictions"
            )
            cs_output = self.constraint_solver.solve_from_torch(
                dnn_output[pred_key].reshape(B, n_items, -1), self.device
            )
        return dnn_output, cs_output

    def on_test_epoch_end(self) -> None:
        self.log("test_F1-fixed", self.font_f1_score.compute())
        self.font_f1_score.reset()
        return super().on_test_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.log("val_F1-fixed", self.font_f1_score.compute())
        self.font_f1_score.reset()
        return super().on_validation_epoch_end()

    def _shared_eval(self, batch, batch_idx, log=True, testing=False):
        x, target = batch
        x = x.view(-1, *x.shape[-3:])
        B = len(x)
        dnn_output, cs_output = self(x, solve=testing)
        if "predictions_merged" in dnn_output:
            # use that one as the "prediction" for global eval
            dnn_output_global = {
                "predictions": dnn_output["predictions_merged"],
                "pred-time": dnn_output["pred-time"],
            }
            eval_output = super()._shared_eval(
                batch,
                batch_idx,
                log=False,
                testing=testing,
                pre_dnn_output=dnn_output_global,
                pre_cs_output=cs_output,
                num_classes=19,
            )
        else:
            eval_output = super()._shared_eval(
                batch,
                batch_idx,
                log=False,
                testing=testing,
                pre_dnn_output=dnn_output,
                pre_cs_output=cs_output,
            )
        str_eval_type = "test" if testing else "val"
        if not testing:
            loss_dict = self.compute_loss(dnn_output, cs_output, target)
            for k, v in loss_dict.items():
                eval_output[f"{str_eval_type}_{k}"] = v

        n_items = torch.cumprod(self.puzzle_shape, -1)[-1]

        # digit classification accuracy
        target_shape = target["label"].reshape(B, self.n_items).shape
        labels_non_reduced = target["label"].reshape(*target_shape)
        labels_reduce = reduce_to_ten_labels(
            labels_non_reduced.reshape(*self.puzzle_shape, -1)
        ).reshape(*target_shape)
        target_digit = {"label": labels_reduce}

        pred = dnn_output["predictions_digit"].reshape(B, n_items, -1)
        dnn_output_digit = {"predictions": dnn_output["predictions_digit"]}

        stats_digit = accuracy_scores(dnn_output_digit, 10, target_digit, None)

        eval_output[f"{str_eval_type}_digit_cell_accuracy"] = stats_digit[
            "cell_accuracy"
        ]
        # per-label accuracy
        for l in torch.arange(10):
            idx = labels_reduce == l
            per_cell_acc = (
                pred.argmax(-1)[idx] == labels_reduce[idx]
            ).flatten().sum() / labels_reduce[idx].numel()
            if labels_reduce[idx].numel() != 0:
                eval_output[f"{str_eval_type}_digit_cell_accuracy_{l.item()}"] = (
                    per_cell_acc
                )
            else:
                # lead to NaNs
                pass

        # manually compute solve-accuracy here:
        if testing:
            if log and eval_output[f"{str_eval_type}_grid_accuracy_solve"] < 1:
                logger.info(
                    "pred_digit",
                    torch.argmax(
                        dnn_output["predictions_digit"].reshape(*target_shape, -1), -1
                    ).reshape(B, *self.puzzle_shape),
                    "pred_style",
                    dnn_output["predictions_style"]
                    .reshape(*target_shape, -1)
                    .reshape(B, *self.puzzle_shape)
                    .cpu()
                    .numpy()
                    .round(3),
                    "label",
                    target["label"].reshape(B, *self.puzzle_shape),
                    "pred after solve",
                    cs_output["solution"].reshape(B, *self.puzzle_shape),
                    "solution",
                    target["cs_output"]["solution"].reshape(B, *self.puzzle_shape),
                    sep="\n",
                )
                if "wiped" in cs_output:
                    logger.info(
                        "wildcard?",
                        cs_output["perception"].reshape(B, *self.puzzle_shape),
                    )
                debug_save_cellimgs(
                    x, pathlib.Path(self.logger.log_dir, f"not-solved_{batch_idx}")
                )
                torch.save(
                    dnn_output,
                    pathlib.Path(self.logger.log_dir, f"not-solved_{batch_idx}.pt"),
                )
                if "incorrect" in target:
                    logger.info("incorrect(s)")
                    index_corr = target["incorrect"]
                    logger.info(
                        "original:",
                        target["cs_output"]["solution"].reshape(B, *self.puzzle_shape)[
                            index_corr[:, :, 0], index_corr[:, :, 1]
                        ],
                    )
                    logger.info(
                        "corrupted:",
                        target["label"].reshape(B, *self.puzzle_shape)[
                            index_corr[:, :, 0], index_corr[:, :, 1]
                        ],
                    )

            # per-label accuracy
            labels_reduce_solve = target["cs_output"]["solution"].flatten().long()
            for l in torch.arange(1, 10):
                idx = labels_reduce_solve == l
                # ignore nan
                if labels_reduce_solve[idx].numel() != 0:
                    eval_output[f"{str_eval_type}_cell_accuracy_solve_{l.item()}"] = (
                        cs_output["solution"].long().flatten()[idx]
                        == labels_reduce_solve[idx]
                    ).sum() / labels_reduce_solve[idx].numel()

        # font eval
        labels_reduce_font = reduce_to_two(
            target["label"].reshape(*self.puzzle_shape, -1)
        ).reshape(*target_shape)
        dnn_output_font = {
            "predictions": dnn_output["predictions_style"].reshape(B, n_items),
        }
        target_font = {"label": labels_reduce_font}
        f1_scores = get_cell_scorer(F1Score, "font_f1-score")
        # key = f'cell_font_f1-score'
        font_stats = f1_scores(dnn_output_font, 2, target_font, None)

        self.font_f1_score.update(dnn_output_font["predictions"], target_font["label"])

        for key in font_stats.keys():
            eval_output[f"{str_eval_type}_{key}"] = font_stats[key]

        for k, v in eval_output.items():
            self.log(
                k,
                v,
                prog_bar=(k == "val_cell_accuracy") or ("loss" in k) or ("font" in k),
                on_step=False,
                on_epoch=True,
            )

        return eval_output


class FontStyleMT(FontStyle):
    def forward(self, x, solve=False):
        l = []
        x = x.view(-1, *x.shape[-3:])
        patches = self.splitter(x)
        B = len(x)
        dnn_output = timeit(self.dnn.forward, l)(patches)
        dnn_output["pred-time"] = torch.tensor(l, device=self.device)
        cs_output = None
        # have to merge both predictions and styles for solvers
        pred_merged = torch.zeros(B, self.n_items, 19, device=self.device)
        pred_digits = dnn_output["predictions"].reshape(B, self.n_items, -1).detach()
        pred_is_hw = dnn_output["styles"].reshape(B, self.n_items).detach()
        # to do vectorize multiplication easily, move batch dimension in the end
        # it also only works if vector has shape B x 0_1 x 0_2 x K (with height and width)
        for b in range(B):
            pred_merged.reshape(B, *self.puzzle_shape, -1)[b, :, :, 1:10] = (
                pred_digits.reshape(B, *self.puzzle_shape, -1)[b, :, :, 1:]
                * (1 - pred_is_hw.reshape(B, *self.puzzle_shape, -1))
            )
            pred_merged.reshape(B, *self.puzzle_shape, -1)[b, :, :, 10:] = (
                pred_digits.reshape(B, *self.puzzle_shape, -1)[b, :, :, 1:]
                * (pred_is_hw.reshape(B, *self.puzzle_shape, -1))
            )
            pred_merged.reshape(B, *self.puzzle_shape, -1)[b, :, :, 0] = (
                pred_digits.reshape(B, *self.puzzle_shape, -1)[b, :, :, 0]
            )
        # separate digit value and style for eval
        dnn_output["predictions_merged"] = pred_merged
        dnn_output["predictions_digit"] = pred_digits
        # P(handwritten AND not empty)
        dnn_output["predictions_style"] = dnn_output["styles"].reshape(
            B, self.n_items
        ).detach() * (1 - pred_merged[:, :, 0].reshape(B, self.n_items))
        if solve:
            cs_output = self.constraint_solver.solve_from_torch(
                pred_merged, self.device
            )
        return dnn_output, cs_output

    def training_step(self, batch, batch_idx):
        x, target = batch
        x = x.view(-1, *x.shape[-3:])
        loss = 0
        patches = self.splitter(x)
        dnn_output = self.dnn(patches)
        loss_dict = self.compute_loss(dnn_output, None, target)
        loss += torch.stack([v for v in loss_dict.values()]).sum()  # TODO hyperparams
        return loss

    def compute_loss(self, dnn_output: dict, cs_output, target) -> dict:
        # create digit_label and font_label
        digit_target = {k: v for k, v in target.items() if "label" not in k}
        label_digit = torch.clone(target["label"])
        label_digit[target["label"] >= 10] -= 9
        label_font = (target["label"] >= 10).type(torch.float).flatten()
        digit_target["label"] = label_digit
        # use parent method for digits
        loss_dict = super().compute_loss(dnn_output, cs_output, digit_target)
        loss_dict["cell_value_ce_loss"] *= self.hparams_initial["mt_alpha"]
        # compute style loss
        counter = Counter(label_font)
        classes_counts = torch.tensor(
            [counter.get(k, 1) for k in range(2)], dtype=torch.float, device=self.device
        )
        train_class_weights = 1 - (classes_counts / classes_counts.sum())
        target_dim = target["label"].flatten().shape
        style_criterion = nn.BCELoss(
            weight=train_class_weights if self.use_weights else None
        )

        cell_style_loss = style_criterion(
            dnn_output["styles"].view(*target_dim), label_font
        )

        loss_dict["cell_font_ce_loss"] = cell_style_loss * (
            1 - self.hparams_initial["mt_alpha"]
        )
        return loss_dict

    def _shared_eval(self, batch, batch_idx, log=True, testing=False):
        x, target = batch
        dnn_output, cs_output = self(x, solve=testing)
        eval_output = dict()
        str_eval_type = "test" if testing else "val"
        if not testing:
            loss_dict = self.compute_loss(dnn_output, cs_output, target)
            for k, v in loss_dict.items():
                eval_output[f"{str_eval_type}_{k}"] = v
        eval_output[f"{str_eval_type}_pred_time"] = dnn_output["pred-time"]
        target_shape = target["label"].flatten().shape
        labels_reduce_font = reduce_to_two(
            target["label"].reshape(*self.puzzle_shape, -1)
        ).reshape(*target_shape)
        pred_font = dnn_output["styles"].reshape(*target_shape).detach()
        dnn_output_font = {
            "predictions": pred_font,
        }
        target_font = {"label": labels_reduce_font}

        self.font_f1_score.update(dnn_output_font["predictions"], target_font["label"])
        stats = {"F1-fixed": self.font_f1_score.compute()}
        stats.update(accuracy_scores(dnn_output_font, 2, target_font, None))
        for k, v in stats.items():
            new_k = f"{str_eval_type}_{k}"
            eval_output[new_k] = v

        for k, v in eval_output.items():
            self.log(
                k,
                v,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        return eval_output
