import pathlib
import sys
import time

import numpy as np
import torch
from cpmpy.model import ExitStatus
from pyswip import Prolog

from constraint_solver.constraint_solver import ConstraintSolver
from nasr.board import Board

root_path = pathlib.Path(__file__).parent.parent.resolve()
if str(root_path) not in sys.path:
    sys.path.insert(0, root_path)


class SudokuSolverProlog(ConstraintSolver):
    """Wrapper for SWI-Prolog used in NASR [1].

    [1] Cornelio et al. Learning where and when to reason in neuro-symbolic inference. ICLR 2023.

    """

    def __init__(self, shape, labelmap) -> None:
        super().__init__(
            shape,
            labelmap,
        )
        self.prolog_instance = Prolog()
        self.prolog_instance.consult(str(root_path / "nasr" / "sudoku_prolog.pl"))

    def solve_prediction(self, y, **solve_params) -> dict:
        cleaned_board = y.argmax(-1).reshape(self.shape).astype(int)
        board_to_solve = Board(cleaned_board)

        assert (
            "solver_nn_output" in solve_params
        ), "missing solver_nn_output as extra parameter"
        assert "predictions" in solve_params, "missing predictions as extra parameter"
        start = time.perf_counter()
        try:
            solver_success = board_to_solve.solve(
                solver="prolog", prolog_instance=self.prolog_instance
            )
        except:
            solver_success = False
        res = {"runtime": time.perf_counter() - start}
        if solver_success:
            res["status"] = np.array([ExitStatus.OPTIMAL.value])
            res["solution"] = board_to_solve.board
        else:
            res["status"] = np.array([ExitStatus.UNSATISFIABLE.value])
            res["solution"] = (
                solve_params["solver_nn_output"]
                .reshape(*self.shape, -1)
                .detach()
                .numpy()
                .argmax(-1)
            )
        res["perception"] = (
            solve_params["predictions"]
            .reshape(*self.shape, -1)
            .detach()
            .numpy()
            .argmax(-1)
        )
        return res

    def solve(self, y, **solve_params):
        # one-hot encoding
        n = len(self.labelmap)
        y_one_hot = np.identity(n)[y.reshape(-1)]
        return self.solve_prediction(
            y_one_hot,
            solver_nn_output=torch.from_numpy(np.full_like(y, np.nan)),
            predictions=torch.from_numpy(np.full_like(y, np.nan)),
        )
