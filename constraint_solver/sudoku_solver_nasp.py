import pathlib
import sys
import time

import numpy as np
from cpmpy.model import ExitStatus

from constraint_solver.constraint_solver import ConstraintSolver
from neurasp import NeurASP
from neurasp.mvpp import MVPP

root_path = pathlib.Path(__file__).parent.parent.resolve()
if str(root_path) not in sys.path:
    sys.path.insert(0, root_path)


class SudokuSolverNeurASP(ConstraintSolver):
    """Wrapper for NeurASP [1]. They demonstrate how to interface machine learning with ASP, through neural atoms.

    [1] Yang, Z., Ishay, A., & Lee, J. (2021, January). NeurASP: embracing neural networks into answer set programming. IJCAI 2020.

    """

    def __init__(self, shape, labelmap) -> None:
        super().__init__(shape, labelmap)
        self.dprogram = """
        % we assign one number at each position (R,C)
        a(R,C,N) :- identify(Pos, img, N), R=Pos/9, C=Pos\9, N!=empty.
        {a(R,C,N): N=1..9}=1 :- identify(Pos, img, empty), R=Pos/9, C=Pos\9.

        % alldifferent rows
        :- a(R,C1,N), a(R,C2,N), C1!=C2.

        % alldifferent columns
        :- a(R1,C,N), a(R2,C,N), R1!=R2.

        % alldifferent 3*3 grid
        :- a(R,C,N), a(R1,C1,N), R!=R1, C!=C1, ((R/3)*3 + C/3) = ((R1/3)*3 + C1/3).
        """
        self.neural_rule = """
        % neural rule
        nn(identify(81, img), [empty,1,2,3,4,5,6,7,8,9]).
        """
        self.neurasp_obj = NeurASP(
            self.neural_rule + self.dprogram, dict(), None, False
        )
        self.n = {"identify": len(self.labelmap)}

    def decode_output(self, stable_model: list):
        res = {}

        grid_vars = np.zeros(81)
        identify_vars = np.zeros(81)

        # required for 'eval' to work properly
        def identify(cell_id, y, val):
            identify_vars[cell_id] = val

        def a(cell_r, cell_c, val):
            grid_vars.reshape(*self.shape)[cell_r, cell_c] = val

        img = None
        empty = 0
        for atom in stable_model:
            eval(atom)

        res["solution"] = grid_vars.reshape(*self.shape)
        res["perception"] = identify_vars.reshape(*self.shape)
        return res

    def solve_prediction(self, y, **solve_params) -> dict:
        # required for compatibility
        self.neurasp_obj.nnOutputs = {"identify": {"img": y.reshape(-1)}}
        # directly jump to Step 3 of the original ´infer´ method
        mvppRules = ""
        facts = ""
        # Step 3: turn the NN outputs (from usual classification neurual networks) into a set of MVPP probabilistic rules
        for ruleIdx in range(self.neurasp_obj.mvpp["nnPrRuleNum"]):
            probs = [
                self.neurasp_obj.nnOutputs[m][t][i * self.n[m] + j]
                for (m, i, t, j) in self.neurasp_obj.mvpp["nnProb"][ruleIdx]
            ]
            if len(probs) == 1:
                mvppRules += "@{} {}; @{} {}.\n".format(
                    probs[0],
                    self.neurasp_obj.mvpp["atom"][ruleIdx][0],
                    1 - probs[0],
                    self.neurasp_obj.mvpp["atom"][ruleIdx][1],
                )
            else:
                tmp = ""
                for atomIdx, prob in enumerate(probs):
                    tmp += "@{} {}; ".format(
                        prob, self.neurasp_obj.mvpp["atom"][ruleIdx][atomIdx]
                    )
                mvppRules += tmp[:-2] + ".\n"

        # Step 3: find an optimal SM under obs
        dmvpp = MVPP(facts + mvppRules + self.dprogram)
        print("debug asp?", facts + mvppRules + self.dprogram)
        start = time.time()
        res = {}
        idx_low_high = y.argsort()
        try:
            stable_model = dmvpp.find_one_most_probable_SM_under_obs_noWC("")[0]

            res.update(self.decode_output(stable_model))
            res["status"] = np.array([ExitStatus.OPTIMAL.value])
        except:
            res["status"] = np.array([ExitStatus.UNSATISFIABLE.value])
            res["solution"] = np.full_like(np.zeros(81).reshape(*self.shape), np.nan)
            res["perception"] = idx_low_high[:, :, -1]
        res["runtime"] = time.time() - start
        return res

    def solve(self, y, **solve_params):
        # one-hot encoding
        n = len(self.labelmap)
        y_one_hot = np.identity(n)[y.reshape(-1)]
        return self.solve_prediction(y_one_hot, **solve_params)
