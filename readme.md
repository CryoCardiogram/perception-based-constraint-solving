# Perception-based Constraint Solving for Sudoku Images

This repository provides the code to reproduce the experiments described in the paper [Perception-based Constraint Solving for Sudoku Images](https://link.springer.com/article/10.1007/s10601-024-09372-9).


## Installation

### with Poetry (recommended)

1) install [poetry](https://python-poetry.org/docs/)

2) clone this repository

3)  run `poetry install` 

### with conda and pip

1) install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

2) clone this repository

3) run `conda create -n pbcs python=3.12`

4) run `conda install potassco::clingo` to install clingo

5) install the remaining python libraries with `pip install -r readme_req.txt`

## Dataset

[Download the data](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/3SUHHR) and follow instructions to extract it within a `data/` folder. 

Check that the data is correctly extracted by running `pytest` at project root level. Tests in `test_databuilder.py` should all pass. 

## Experiments 

Use `exp_joint_inf.py` to train a neural network (NN) and run an evaluation on the test. Trained NN are stored automatically in their respective subfolder under `output/`. You can [download](https://github.com/CryoCardiogram/perception-based-constraint-solving/releases/tag/v0.1.0) weights of pretrained backbone in the release and place them under a `weights/` folder to speed-up training. 

Use `exp_calibration.py` to calibrate a pretrained model. 

Run either of those scripts with the `--help` option for a detailed explanation about their usage. 

## Citation
