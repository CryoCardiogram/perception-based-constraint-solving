import torch
import numpy as np
import cpmpy as cp
from abc import abstractmethod

class ConstraintSolver:
    """ Abstract Class for modeling and solving CP problems

    """
    def __init__(self, shape, labelmap) -> None:
        """_summary_

        Args:
            shape (iterable): Puzzle board dimensions
            labelmap (iterable): map label-id -> semantic meaning
        """
        self.labelmap = labelmap
        self.shape = shape
    
    @abstractmethod
    def solve_prediction(self, y, **solve_params) -> dict:
        """
        Solve call returns a dict containing at least the following keys:
        `solution`, `perception`

        Args:
            y (np.ndarray): predicted input of the solver, should be of dimension `len(self.shape)+1`
        """
        pass
    
    @abstractmethod
    def solve(self, y, **solve_params):
        """Solve call

        Args:
            y (np.ndarray): exact input of the solver, should be of shape `self.shape`
        """
        pass

    def solve_from_torch(self, ytorch:torch.Tensor, device, **solve_params):
        # batch
        list_cs_output = [
            self.solve_prediction(y_t.reshape(*self.shape, -1), **solve_params)
            for y_t in ytorch.detach().cpu().numpy()
        ]
        # merge 
        dict_of_lists_of_tensors = {key: [torch.from_numpy(d[key]).to(device).float() if isinstance(d[key], np.ndarray) else torch.tensor([d[key]], device=device) 
                                          for d in list_cs_output] 
                                          for key in list_cs_output[0].keys()}
        
        dict_of_tensors = {key: torch.stack(tensor_list) if isinstance(tensor_list[0], torch.Tensor) else tensor_list 
                           for key, tensor_list in dict_of_lists_of_tensors.items()}
        return dict_of_tensors

class ObjectiveFun:
    """Abstract class for modeling the objective function
    """
    def __init__(self, precision:float = 1e-4, is_log=False, is_extended=False) -> None:
        """

        Args:
            precision (float, optional): required for float to int conversion. Defaults to 1e-4.
            is_log (bool, optional): set to True if NN output are in logspace. Defaults to False.
            is_extended (bool, optional): set to True if NN output encompasses fontstyle. Defaults to False.
        """
        self.precision = precision
        self.is_log = is_log 
        self.is_extended = is_extended


    @abstractmethod
    def get_logprobs(self, y, **obj_params) -> dict:
        pass
    
    def build_objective(self, dict_logprobs:dict, identify, **obj_params) -> cp.expressions.core.Operator:
        """Build the objective

        Args:
            dict_logprobs (dict): output of ´get_logprobs´
            identify (cp.Intvar): array of cp.intvar perception variables 

        Returns:
            cp.expressions.core.Operator: objective function expression to be minimized
        """
        logprobs = dict_logprobs['value']
        obj = np.sum(logprobs[idx][v] for idx, v in np.ndenumerate(identify))
        return obj
