import seaborn as sns
import pandas as pd
import os
import yaml
import pathlib
import torch
from collections import OrderedDict
from typing import Iterable

root_path = pathlib.Path(__file__).parent.parent.resolve().__str__()


# hotfix


def set_method_name(row):
    name = ""
    # first part, solver: baseline, hybrid or higher-order, or related work
    if row["obfun"] == 4:
        solver = "NeurASP"
    elif row["obfun"] == 5:
        solver = "SwiPl"

    elif row["topk"] == 1:
        solver = "baseline"
    elif row["solver"] == "HOCOP(1)" or row["solver"] == "HCOP":
        solver = "hybrid"
    elif "HOCOP" in row["solver"]:
        solver = "higher-order"
    elif "wildcard" in row["solver"]:
        if row["no_good"]:
            solver = "higher-order"
        else:
            solver = "hybrid"
    else:
        solver = "baseline"
    name += solver
    # then wildcard, either wildcard-prune, wildcard-soft, wildcard-dynamic-prune, wildcard-dynamic-soft
    if row["wildcard_tr"] != "None" or row["dynamic"]:
        wildcard = "wildcard"
        if row["dynamic"]:
            wildcard += "-dynamic"
        if row["wildcard_soft"]:
            wildcard += "-soft"
        else:
            wildcard += "-prune"
        name += f" ({wildcard})"

    if row["nasr"]:
        name = f"NASR-{name}"
    return name
