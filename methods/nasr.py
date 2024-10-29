import sys
import pathlib
from torch import nn

from constraint_solver.constraint_solver import ConstraintSolver

root_path = pathlib.Path(__file__).parent.parent.resolve().__str__()
if root_path not in sys.path:
    sys.path.insert(0, root_path)
import torch.nn as nn
import torch
from pbr_module import PBR
from nasr.transformer_sudoku import TransformerSudoku
from utee import timeit
from cpmpy.model import ExitStatus
from lightning.pytorch.utilities import grad_norm


class NASRTrainer(PBR):
    def __init__(
        self,
        dnn: nn.Module,
        constraint_solver: ConstraintSolver,
        splitter: nn.Module,
        lr=0.01,
        num_pred_classes=10,
        puzzle_shape=...,
        hparams=...,
        freeze_level=0,
        weight_decay=3e-2,
        warmup=10,
        clip_gradient_norm=0,
    ):
        """_summary_

        Args:

            freeze_level (int, optional): Freeze up to `freeze_level` layers in the NASR pipeline,
                from Perception to Mask-NN. If set to -1, then transformers are trained from scratch. Default to 0 (enable gradient update everywhere)
            weight_decay (_type_, optional): _description_. Defaults to 3e-2.
            warmup (int, optional): _description_. Defaults to 10.
            clip_gradient_norm (int, optional): _description_. Defaults to 0.
        """
        super().__init__(
            dnn,
            constraint_solver,
            splitter,
            lr,
            num_pred_classes,
            puzzle_shape,
            hparams,
        )
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.clip_gradient_norm = clip_gradient_norm
        self.tf_sudoku = TransformerSudoku(self.dnn, 81)
        if freeze_level > -1:
            print("debug nasr pretrained weights for sudoku tf")
            self.tf_sudoku.load_pretrained_models("big_kaggle")
        self.freeze_level = freeze_level

        self.nasr_layers = [
            self.tf_sudoku.perception,
            self.tf_sudoku.nn_solver,
            self.tf_sudoku.mask_nn,
        ]

        for layer in self.nasr_layers[: self.freeze_level]:
            for p in layer.parameters():
                p.requires_grad_(False)

        if self.hparams["pretrained"] is None:
            # still have to train perception module
            for p in self.tf_sudoku.perception.parameters():
                p.requires_grad_(True)

    # def on_before_optimizer_step(self, optimizer, optimizer_idx):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.tf_sudoku, norm_type=2)
    #     self.log_dict(norms)

    def forward(self, x, solve=False):
        l = []
        B = len(x)
        x = x.view(B, *x.shape[-3:])
        patches = self.splitter(x)
        dnn_output = timeit(self.tf_sudoku, l)(patches, "rl")
        dnn_output["pred-time"] = torch.tensor(l, device=self.device)
        cs_output = None

        if solve:
            #
            # cleaned_boards = torch.zeros(B, self.n_items, dnn_output['solver_nn_output'].shape[-1]+1, device=self.device )
            # cleaned_boards[:,:,1:] = dnn_output['solver_nn_output'].view(B, self.n_items,-1) * (dnn_output['mask_nn_output'].sigmoid().view(B,self.n_items, -1))
            # # cleaned_boards[:,:,0] = (1-dnn_output['mask_nn_output'].view(B, self.n_items).sigmoid().view(B, self.n_items))
            # # print(dnn_output['solver_nn_output'].argmax(-1).reshape(9,9)+1)
            # # print('vs the one found by solver?')
            # cs_output = self.constraint_solver.solve_from_torch(
            #     cleaned_boards.reshape(B,self.n_items,-1), self.device, **dnn_output
            # )

            masking_prob = dnn_output["mask_nn_output"].view(B, self.n_items).sigmoid()
            # use torch.distrib to sample from predicted prob. distribution
            b = torch.distributions.Bernoulli(masking_prob)
            sampled_mask_boards = b.sample().view(B, self.n_items)
            dnn_output["log_prob"] = b.log_prob(sampled_mask_boards)

            # model.saved_log_probs = b.log_prob(sampled_mask_boards)
            sampled_mask_boards = sampled_mask_boards.reshape(masking_prob.shape)
            # TODO: highlight that the clean board is built by adding mask_nn output as prob mass for value 0
            cleaned_boards = torch.multiply(
                dnn_output["solver_nn_output"].view(B, self.n_items, -1),
                sampled_mask_boards.view(B, self.n_items, -1),
            )
            dnn_output["cleaned_board"] = cleaned_boards
            cleaned_board_for_prob_solving = torch.zeros(
                B,
                self.n_items,
                dnn_output["solver_nn_output"].shape[-1] + 1,
                device=self.device,
            )
            cleaned_board_for_prob_solving[:, :, 1:] = cleaned_boards
            # cleaned_boards[:,:,1:] = dnn_output['solver_nn_output'].view(B, self.n_items,-1)
            # cleaned_boards[:,:,0] = sampled_mask_boards.view(B, self.n_items)
            # call solver on cleaned predicted solution (B x 81 x 9)
            # solver
            cs_output = self.constraint_solver.solve_from_torch(
                cleaned_board_for_prob_solving, self.device, **dnn_output
            )

        return dnn_output, cs_output

    def training_step(self, batch, batch_idx):
        x, target = batch
        B = len(x)
        loss = 0
        dnn_output, cs_output = self(x, True)

        # compute policy_loss
        loss_dict = self.compute_loss(dnn_output, cs_output, target)
        loss += torch.stack([v for v in loss_dict.values()]).mean()
        return loss

    def compute_loss(self, dnn_output: dict, cs_output: dict, target: dict) -> dict:
        # compute rewards for the batch
        final_solution = []
        for symbolic_solution, status, nn_solution in zip(
            cs_output["solution"], cs_output["status"], dnn_output["cleaned_board"]
        ):
            if status == ExitStatus.UNSATISFIABLE.value:
                print("debug adding nn solver solution", nn_solution.argmax(-1).float())
                final_solution.append(nn_solution.argmax(-1).float())
            else:
                final_solution.append(symbolic_solution.float().view(-1, self.n_items))
        final_solution = torch.vstack(final_solution)
        ground_truth_board = target["cs_output"]["solution"].view(final_solution.shape)
        # 10 if the whole grid match, 0 otherwise
        rewards = torch.all(final_solution == ground_truth_board, -1).float() * 10
        partial_reward = torch.eq(final_solution, ground_truth_board).sum(-1)
        rewards += partial_reward / 81
        # remove baseline for more stability
        rewards = (rewards - rewards.mean()) / (
            rewards.std() + torch.finfo(torch.float32).eps
        )
        # compute policy loss
        policy_loss = rewards @ -dnn_output["log_prob"]
        return {"policy_loss": policy_loss.sum()}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.tf_sudoku.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1 / self.warmup, total_iters=self.warmup
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
