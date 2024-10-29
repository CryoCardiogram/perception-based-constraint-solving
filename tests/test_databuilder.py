import numpy as np
from custom_types import BackboneArch, Datasets
from neuralnet import get_model
from constraint_solver.sudoku_solver import (
    SudokuSolver,
)
from constraint_solver.sudoku_solver import ObjectiveHCOP
from utee.databuilder import build_data


def test_build_data():
    conf = {
        "max_data_size": 20,
        "puzzle_shape": (9, 9),
        "backbone": BackboneArch.LENET,
        "dataset": Datasets.VSUDOKU_MNIST,
        "num_pred_class": 10,
        "batchnorm": True,
        "fc_hidden_layers": [20, 10],
    }
    cs = SudokuSolver(conf["puzzle_shape"], np.arange(10), ObjectiveHCOP())
    net, kwargs_dataset = get_model(
        conf["backbone"],
        conf["dataset"],
        conf["num_pred_class"],
        conf["puzzle_shape"],
        batchnorm=conf["batchnorm"],
        fc_hidden_layers=conf["fc_hidden_layers"],
    )
    train_dl_mnist, valid_dl_mnist, test_dl_mnist = build_data(
        conf["dataset"],
        conf["backbone"],
        cs,
        kwargs_dataset,
        conf,
    )
    assert (
        len(train_dl_mnist.dataset)
        + len(valid_dl_mnist.dataset)
        + len(test_dl_mnist.dataset)
        == 20
    )

    conf["dataset"] = Datasets.VSUDOKU_APP
    conf["backbone"] = BackboneArch.SVHN_CNN
    conf["max_data_size"] = 30
    net, kwargs_dataset = get_model(
        conf["backbone"],
        conf["dataset"],
        conf["num_pred_class"],
        conf["puzzle_shape"],
        batchnorm=conf["batchnorm"],
        fc_hidden_layers=conf["fc_hidden_layers"],
    )
    train_dl_app, valid_dl_app, test_dl_app = build_data(
        conf["dataset"],
        conf["backbone"],
        cs,
        kwargs_dataset,
        conf,
    )
    assert (
        len(train_dl_app.dataset) + len(valid_dl_app.dataset) + len(test_dl_app.dataset)
        == 30
    )
    conf["dataset"] = Datasets.VSUDOKU_NEURASP
    conf["backbone"] = BackboneArch.LENET
    conf["max_data_size"] = 25
    train_dl_asp, valid_dl_asp, test_dl_asp = build_data(
        conf["dataset"],
        conf["backbone"],
        cs,
        kwargs_dataset,
        conf,
    )
    assert (
        len(train_dl_asp.dataset) + len(valid_dl_asp.dataset) + len(test_dl_asp.dataset)
        == 25
    )
    return
