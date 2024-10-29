from constraint_solver.constraint_solver import ConstraintSolver, ObjectiveFun
import cpmpy as cp
import numpy as np
from cpmpy.tools import mus
from cpmpy.transformations.flatten_model import flatten_model
import sys
import pathlib
import logging
from cpmpy.model import ExitStatus
from cpmpy.solvers import CPM_ortools

root_path = pathlib.Path(__file__).parent.parent.resolve()
if str(root_path) not in sys.path:
    sys.path.insert(0, root_path)


logger = logging.getLogger(__file__)


def prob_matrix_2_log_prob_cpm(probs, precision: float):
    """Convert probability tensor to cpmarray negative log probabilities

    Args:
        probs (_type_): np.array of probabilities
        precision (_type_): control rounding to integers

    Returns:
        NDVarArray: negative log-probabilities, rounded to integer values
    """
    logprobs = -np.log(np.maximum(probs, np.finfo(np.float32).eps))
    # logprobs = -np.log(np.maximum(probs, precision))
    logprobs = np.array(logprobs / precision).astype(int)
    logprobs = cp.cpm_array(logprobs)
    return logprobs


def reduce_to_ten_classes(probs: np.ndarray) -> np.ndarray:
    """if probs is a probability tensor that considers print/written value (19 classes), reduces it to 10-class probability tensor

    Args:
        probs (np.ndarray): probability tensor (not in logspace)

    Returns:
        np.ndarray:
    """
    reduced_probs = np.zeros((*probs.shape[:-1], 10))
    # blank
    reduced_probs[:, :, 0] = probs[:, :, 0]
    # value
    for i in range(1, 10):
        reduced_probs[:, :, i] = probs[:, :, i] + probs[:, :, i + 9]
    return reduced_probs


class ObjectiveHCOP(ObjectiveFun):
    """Build an objective function s.t. the solver maximizes the joint likelihood across all cells"""

    def get_logprobs(self, y, **obj_params) -> dict:
        # convert to int with given precision

        if self.is_log:
            tmp_probs = np.exp(y)
        else:
            tmp_probs = y
        if self.is_extended:
            probs_handwritten = np.sum(tmp_probs[:, :, 10:], -1)

            tmp_probs = reduce_to_ten_classes(y)
        logprobs = prob_matrix_2_log_prob_cpm(tmp_probs, self.precision)
        res = {"value": logprobs}
        if self.is_extended:
            res["style"] = prob_matrix_2_log_prob_cpm(probs_handwritten, self.precision)
        return res


class ObjectivePrintedOnly(ObjectiveFun):
    """Build an objective function such that the solver maximizes the joint likelihood,
    only over cells with printed digit only
    """

    def __init__(
        self,
        precision: float = 0.0001,
        is_log=False,
        is_extended=False,
        weight_printed=1.0,
    ) -> None:
        """_summary_

        Args:
            precision (float, optional): control solver rounding of float to integers such as
            `int(probs*(1/precision))`. Defaults to 0.0001.
            is_log (bool, optional): Specify if neural network output are log-probabilities. Defaults to False.
            is_extended (bool, optional): specify if input is extended to account for printed/written digits. Defaults to False.
            weight_printed (float, optional): Weight (between 0 and 1) that printed/blank should have wrt handwritten digit. Defaults to 1.0.
        """
        super().__init__(precision, is_log, is_extended)
        assert (
            self.is_extended
        ), "set is_extended to True to use this objective function"
        self.weight_printed = weight_printed

    def get_logprobs(self, y, **obj_params) -> dict:
        if self.is_log:
            tmp_probs = np.exp(y)
            logprobs = prob_matrix_2_log_prob_cpm(tmp_probs, self.precision)
        else:
            tmp_probs = y

        prob_hw = tmp_probs[:, :, 10:].sum(-1)
        # have to do it before conversion to int cpm_array
        logprobs = -np.log(np.maximum(tmp_probs, np.finfo(np.float32).eps))
        logprobs = np.array(logprobs / self.precision)
        logprobs[:, :, :10] *= self.weight_printed
        logprobs[:, :, 10:] *= 1 - self.weight_printed
        logprobs = cp.cpm_array(logprobs.astype(int))

        # logprob_printed = prob_matrix_2_log_prob_cpm(probs_fontstyle, self.precision)
        return {
            "value": logprobs,
            "style": prob_matrix_2_log_prob_cpm(prob_hw, self.precision),
        }

    def build_objective(
        self, dict_logprobs: dict, identify, **obj_params
    ) -> cp.expressions.core.Operator:
        logprobs = dict_logprobs["value"]
        obj = np.sum(logprobs[idx][v] for idx, v in np.ndenumerate(identify))
        obj += np.sum(
            logprobs[idx][v + 9] for idx, v in np.ndenumerate(identify) if v > 0
        )
        return obj


class SudokuSolver(ConstraintSolver):
    def __init__(
        self,
        shape,
        labelmap,
        objective_builder: ObjectiveFun,
        topk: int = None,
        time_limit: int = None,
        corrupt: bool = False,
    ) -> None:
        """Default Hybrid Sudoku Solver of https://arxiv.org/abs/2003.11001
        implemented with CPMpy.

        Args:
            labelmap (dict): map symbol representation to integers values
            shape (Tuple[int,int]): shape of the puzzle grid, (9,9) for a regular Sudoku
            objective_builder (ObjectiveFun): Initialized objective function builder
            topk (int, optional): only use top 'k' most likely predictions. Defaults to None.
            time_limit (int, optional): solver runtime limit. Defaults to None.
        """
        super().__init__(shape, labelmap)
        self.objective_builder = objective_builder
        self.precision = objective_builder.precision
        self.topk = topk
        self.time_limit = time_limit
        self.is_extended = objective_builder.is_extended
        self.corrupt = corrupt

        # constants
        n = shape[0]
        assert n == shape[1], "Expecting square logprobs matrix"
        assert len(labelmap) == n + 1, f"need {n} symbols + empty one"
        b = np.sqrt(n).astype(int)
        assert n == b * b, "Expecting n to be a squared nr"

        # decision vars
        grid = cp.IntVar(1, n, shape=shape, name="grid")

        # perception vars
        identify = cp.IntVar(0, n, shape=shape, name="perception")

        # plain model
        m = cp.Model(
            [cp.alldifferent(row) for row in grid],
            [cp.alldifferent(col) for col in grid.T],
            [
                cp.alldifferent(grid[i : i + b, j : j + b])
                for i in range(0, n, b)
                for j in range(0, n, b)
            ],
        )

        # identify layer
        # identify element is either empty symbol or matches grid symbol
        m += [
            (identify[idx] != 0).implies(grid[idx] == identify[idx])
            for idx, _ in np.ndenumerate(grid)
        ]

        self.grid = grid
        self.identify = identify
        self.model = flatten_model(m)

    def __str__(self) -> str:
        return "HCOP"

    def get_objective(self, y, **obj_params):
        assert (
            y.shape[:2] == self.shape
        ), f"pred should have shape of the puzzle in first dimensions, namely ({self.shape})"
        dict_logprobs = self.objective_builder.get_logprobs(y, **obj_params)
        obj = self.objective_builder.build_objective(
            dict_logprobs, self.identify, **obj_params
        )
        return obj

    def _solve_step(
        self, newmodel: CPM_ortools, y: np.ndarray, time_limit: int, **solve_params
    ):
        if self.topk is not None:
            self.post_topk(newmodel, y)

        if newmodel.solve(time_limit=time_limit, num_search_workers=1):
            return {
                "runtime": np.asarray(newmodel.cpm_status.runtime, dtype=float),
                "status": np.asarray(newmodel.cpm_status.exitstatus.value, dtype=int),
                "solution": self.grid.value().astype(float),
                "perception": self.identify.value().astype(float),
            }
        else:
            idx_low_high = y.argsort()
            return {
                "runtime": np.asarray(newmodel.cpm_status.runtime, dtype=float),
                "status": np.asarray(newmodel.cpm_status.exitstatus.value, dtype=int),
                "solution": np.full_like(self.grid.value(), np.nan).astype(float),
                "perception": idx_low_high[:, :, -1].astype(float),
            }

    def solve_prediction(self, y, **solve_params) -> dict:
        # probs is n*n*n+1, last dimension probabilities
        assert (
            y.shape[:-1] == self.shape
        ), f"Shapes must match for all but last dimension, {y.shape} vs {self.shape}"

        obj = self.get_objective(y, **solve_params)
        newmodel = cp.Model(self.model.constraints, minimize=obj)
        newmodel = CPM_ortools(cpm_model=newmodel)
        res = self._solve_step(newmodel, y, self.time_limit, **solve_params)
        res["objective"] = obj.value()
        if res["objective"] is None:
            res["objective"] = np.nan
        return res

    def solve(self, y, **solve_params):
        assert (
            y.shape == self.shape
        ), f"Shapes must match in all dimension, got {y.shape} vs {self.shape}"
        tmp_givens = np.copy(y)
        if self.is_extended or self.corrupt:
            tmp_givens[tmp_givens > 9] -= 9
        newmodel = cp.Model(self.model.constraints)
        newmodel += cp.all(self.identify == cp.cpm_array(tmp_givens))
        # newmodel = CPM_ortools(newmodel)
        newmodel.solve(time_limit=self.time_limit)
        if newmodel.cpm_status.exitstatus != ExitStatus.OPTIMAL:
            print("solving: \n", y)
            print(mus(newmodel.constraints))
            print(self.identify.value())
            print(newmodel.cpm_status.exitstatus)
        return {
            "runtime": np.asarray(newmodel.cpm_status.runtime),
            "status": np.asarray(newmodel.cpm_status.exitstatus.value),
            "solution": self.grid.value(),
            "perception": self.identify.value(),
            #'solution_onehot': np.identity(len(self.labelmap))[self.grid.value().reshape(-1)]
        }

    def post_topk(self, model, preds) -> None:
        idx_low_high = preds.argsort()
        # n-k worst of last dimension
        for i in range(preds.shape[-1] - self.topk):
            model += [not cp.any(self.identify == idx_low_high[:, :, i])]


class SudokuSolverNoGood(SudokuSolver):
    def __init__(
        self,
        shape,
        labelmap,
        objective_builder: ObjectiveFun,
        topk: int = None,
        time_limit: int = None,
        max_iter=100,
        corrupt: bool = False,
        log: str = None,
    ) -> None:
        super().__init__(shape, labelmap, objective_builder, topk, time_limit, corrupt)
        self.max_iter = max_iter
        self.log = log
        self.count = 0

    def __str__(self) -> str:
        return f"HOCOP({self.max_iter})"

    def solve_prediction(self, y, **solve_params) -> dict:
        assert (
            y.shape[:-1] == self.shape
        ), f"Shapes must match for all but last dimension, {y.shape} vs {self.shape}"

        obj = self.get_objective(y, **solve_params)
        newmodel = cp.Model(self.model.constraints, minimize=obj)
        newmodel = CPM_ortools(newmodel)
        res = self.solve_till_unique(newmodel, y)
        res["objective"] = obj.value()
        return res

    def solve_till_unique(self, model: CPM_ortools, y):
        res = self._solve_step(model, y, self.time_limit)
        res["count_nogoods"] = 0
        time_left = self.time_limit - res["runtime"].item()
        while not self.is_unique(self.identify.value(), self.grid.value()):
            if self.log:
                with open(self.log, "+ta") as f:
                    f.writelines(
                        [str(self.count)] + [f"{k}: {v}\n" for k, v in res.items()]
                    )
            if res["count_nogoods"] >= self.max_iter:
                break
            # if res['status'] == ExitStatus.UNSATISFIABLE.value:
            #     break
            if time_left <= 0:
                break
            # forbid current assignment
            nogood_constraints = [
                not cp.all(self.identify == self.identify.value()),
            ]
            model += nogood_constraints
            solver_output = self._solve_step(
                model, y, np.max([time_left, np.finfo(np.float32).eps])
            )
            res.update({k: v for k, v in solver_output.items() if "runtime" not in k})
            res["runtime"] += solver_output["runtime"]
            if self.log:
                with open(self.log, "+ta") as f:
                    f.writelines(
                        [str(self.count)] + [f"{k}: {v}\n" for k, v in res.items()]
                    )
            time_left -= res["runtime"].item()
            res["count_nogoods"] += 1
        self.count += 1
        return res

    def is_unique(self, vals_identify, vals_grid):
        "check that 'identify' has no other solution than grid"
        m = CPM_ortools(self.model)
        m += [(self.identify == vals_identify)]
        m += [not cp.all((self.grid == vals_grid).flatten())]
        is_sat = m.solve(stop_after_first_solution=True)
        return not is_sat


class SudokuSolverWildcard(SudokuSolverNoGood):
    def __init__(
        self,
        shape,
        labelmap,
        objective_builder: ObjectiveFun,
        topk: int = None,
        time_limit: int = None,
        max_iter: int = 1,
        corrupt: bool = False,
        wildcard_tr: float = None,
        soft: bool = False,
        dynamic: bool = False,
        gamma_hw: float = None,
        **kwargs,
    ) -> None:
        """Hybrid Sudoku Solver with wildcard option.
        The solver can disregard predictions that do not reach the minimum confidence threshold for a given cell.

        Args:
            labelmap (dict): map symbol representation to integers values
            shape (Tuple[int,int]): shape of the puzzle grid, (9,9) for a regular Sudoku
            objective_builder (ObjectiveFun): Initialized objective function builder
            topk (int, optional): only use top 'k' most likely predictions. Defaults to None.
            time_limit (int, optional): solver runtime limit. Defaults to None.
            wildcard_tr (float, optional): Threshold for the wildcard option. Defaults to None.
            soft (bool, optional): if True, the solver does not prune low-scored values. Default to False.
            dynamic (bool, optional): if True, compute the threshold for each cell, based on P(printed) and P(handwritten). Defaults to False.
            gamma_hw (float, optional) coeff. to compute P(wrong) dynamically. Defaults to None.
        """
        super().__init__(
            shape, labelmap, objective_builder, topk, time_limit, max_iter, corrupt
        )
        self.soft = soft
        self.threshold = wildcard_tr
        self.dynamic = dynamic
        self.gamma_hw = gamma_hw
        # self.gamma_print = gamma_print
        if wildcard_tr is None:
            self.threshold = 1 / len(labelmap)
        n = shape[0]
        b = np.sqrt(n).astype(int)
        grid = cp.IntVar(1, n, shape=shape, name="grid")
        # perception vars
        # add 'wildcard' option for solver
        identify = cp.IntVar(0, n + 1, shape=shape, name="perception")
        # keep track of wiped out cells
        wiped = cp.BoolVar(shape=shape, name="wildcard")

        # plain model
        m = cp.Model(
            [cp.alldifferent(row) for row in grid],
            [cp.alldifferent(col) for col in grid.T],
            [
                cp.alldifferent(grid[i : i + b, j : j + b])
                for i in range(0, n, b)
                for j in range(0, n, b)
            ],
        )

        # perception layer
        # identify is either empty symbol, matches grid symbol or wildcard symbol
        m += [
            (cp.all([(identify[idx] > 0), (identify[idx] <= n)])).implies(
                grid[idx] == identify[idx]
            )
            for idx, _ in np.ndenumerate(grid)
        ]

        # monitor perception variable assigned to wildcard
        m += [
            (identify[idx] > n) == (wiped[idx]) for idx, _ in np.ndenumerate(identify)
        ]

        self.grid = grid
        self.identify = identify
        self.model = flatten_model(m)
        self.wiped = wiped

    def __str__(self):
        return f"wildcard({self.threshold})"

    def post_wildcard_threshold(self, model, preds):
        tmp_probs = preds
        if self.is_extended:
            tmp_probs = reduce_to_ten_classes(preds)
        to_discard = tmp_probs < self.threshold

        model += [
            (self.identify[i, j] != v)
            for (i, j, v), b in np.ndenumerate(to_discard)
            if b
        ]

    def get_objective(self, y, **obj_params):
        dict_logprobs = self.objective_builder.get_logprobs(y, **obj_params)
        if self.dynamic:
            assert (
                "style" in dict_logprobs
            ), "missing style attribute from dict of logprobs"
            # log to prob space
            prob_hw = np.exp(-(dict_logprobs["style"].astype(float) * self.precision))
            prob_wrong = self.gamma_hw * prob_hw + (1 - self.gamma_hw) * (1 - prob_hw)
            self.threshold = prob_wrong

        # modify logprobs to account for the wildcard
        log_threshold = prob_matrix_2_log_prob_cpm(
            np.array([self.threshold]), self.precision
        )
        target_shape = np.asarray(dict_logprobs["value"].shape)
        target_shape[-1] += 1
        lprob_value_and_wildcard = np.zeros(target_shape, dtype=int)
        lprob_value_and_wildcard[:, :, :-1] = dict_logprobs["value"]
        lprob_value_and_wildcard[:, :, -1] = log_threshold
        dict_logprobs["value"] = cp.cpm_array(lprob_value_and_wildcard)
        # build objective
        obj = self.objective_builder.build_objective(
            dict_logprobs, self.identify, **obj_params
        )
        return obj

    def solve_prediction(self, y, **solve_params) -> dict:
        res = super().solve_prediction(y, **solve_params)
        res["wiped"] = self.wiped.value()
        return res

    def _solve_step(
        self, newmodel: cp.Model, y: np.ndarray, time_limit: int, **solve_params
    ):
        if not self.soft:
            self.post_wildcard_threshold(newmodel, y)
        # print('debug sudoku solver threshold: post wildcard constraints')
        return super()._solve_step(newmodel, y, time_limit, **solve_params)
