import sys
import pathlib
import os

root_path = pathlib.Path(__file__).parent.parent.resolve().__str__()
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from .datawrapper import (
    CellDataset,
    ImgDataset,
    CellIndexedDataset,
    BalanceClassSampler,
)
from constraint_solver.constraint_solver import ConstraintSolver
import torch
from torch.utils.data import RandomSampler, WeightedRandomSampler
import torchvision
import pickle
from torch.utils.data import DataLoader
import lightning as pl
import numpy as np
from custom_types import Datasets, BackboneArch, ImbalancePolicy, IS_MULTI, IS_WHOLE
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from utee.img_processing import CorruptSudoku, AugmentSudoku, extract_grid
from PIL import Image
from pathlib import Path
import os
import torchvision.transforms as T
from tqdm.auto import tqdm
from torchvision.utils import make_grid, save_image
import copy
from concurrent.futures import ProcessPoolExecutor


def _load_pickle(file_to_load):
    with open(file_to_load, "rb") as fp:
        labels = pickle.load(fp)
    return labels


def __get_max_data_ind(x, y, max_data_size: int, rng: np.random.Generator):
    if max_data_size is None:
        return np.arange(len(y))
    else:
        return rng.choice(np.arange(len(y)), min(max_data_size, len(x)), replace=False)


def get_data(
    dataset: Datasets, puzzle_shape, max_data_size: int = None, fontstyle: bool = False
):
    """Fetch raw data (images, labels, solutions) for the given `dataset` type.

    Args:
        dataset (Datasets): Dataset type
        puzzle_shape (Iterable[Int]): shape of the decision problem
        max_data_size (int): number of instance to sample to build dataset. Default to None (all data used).
        fontstyle (bool): toggle to build combined labels (digit-style). Only relevant for Assistant App dataset

    Raises:
        NotImplementedError: dataset type must be defined in config.py.

    Returns:
        dict: 'images' with input features, 'labels' with ground-truth classification labels,
            ´solutions´ with true optimal solutions. `rng` random generator used to sample instances.
    """
    rng = np.random.default_rng(243)
    raw_data = dict(rng=rng)

    if dataset == Datasets.VSUDOKU_APP:
        x = np.load(os.path.join(root_path, "data/raw_data/app/data.npy")).astype(
            np.uint8
        )
        y_num = torch.tensor(
            np.load(os.path.join(root_path, f"data/raw_data/app/labels.npy")).reshape(
                -1, *puzzle_shape
            ),
            dtype=torch.long,
        )
        max_data_ind = __get_max_data_ind(x, y_num, max_data_size, rng)
        x = x[max_data_ind]
        y_num = y_num[max_data_ind]
        y = y_num

        labels2int = defaultdict(dict)
        for i in range(10):
            if i == 0:
                labels2int[0] = {0: 0, 1: 0}
            else:
                labels2int[i] = {0: i, 1: i + 9}
        y_style = torch.tensor(
            np.load(
                os.path.join(root_path, f"data/raw_data/app/labels_hw.npy")
            ).reshape(-1, *puzzle_shape),
            dtype=torch.long,
        )[max_data_ind]

        combined_labels = np.asarray(
            [
                labels2int[num][hw]
                for num, hw in zip(
                    y_num.reshape(-1).numpy(), y_style.reshape(-1).numpy()
                )
            ]
        ).reshape(y_num.shape)
        combined_labels = torch.from_numpy(combined_labels).long()
        raw_data["combined_labels"] = combined_labels
        if fontstyle:
            # print('debug databuilder: combined labels', np.where(combined_labels > 9))
            y = combined_labels
        sols = None
    elif dataset == Datasets.VSUDOKU_NEURASP:
        img_dict = _load_pickle(
            os.path.join(root_path, "data/raw_data/neurasp/image_dict_reg_100.p")
        )
        label_dict = _load_pickle(
            os.path.join(root_path, "data/raw_data/neurasp/label_dict_reg_100.p")
        )
        sorted_keys = sorted(list(img_dict.keys()))
        x = np.array(
            [Image.fromarray(img_dict[k]).resize((300, 300)) for k in sorted_keys]
        )
        y = torch.stack([torch.from_numpy(label_dict[k]) for k in sorted_keys]).long()
        sols = None
        max_data_ind = __get_max_data_ind(x, y, max_data_size, rng)
        x = x[max_data_ind]
        y = y[max_data_ind]
        # print('debug neurasp vsudoku x.shape', x.shape)

    elif dataset == Datasets.VSUDOKU_MNIST:
        vizsudoku_symbolic_dir = os.path.join("data", "raw_data", "mnist")
        assert os.path.exists(os.path.join(vizsudoku_symbolic_dir, "features.pt")), (
            "generate data set first"
        )
        X = torch.load(os.path.join(vizsudoku_symbolic_dir, "features_img.pt"))
        y = torch.load(os.path.join(vizsudoku_symbolic_dir, "features.pt")).long()
        sols_dict = None
        if os.path.exists(os.path.join(vizsudoku_symbolic_dir, "sols.pt")):
            sols_dict = torch.load(os.path.join(vizsudoku_symbolic_dir, "sols.pt"))
        max_data_ind = __get_max_data_ind(X, y, max_data_size, rng)
        X = X[max_data_ind]
        y = y[max_data_ind]
        sols = None
        if sols_dict is not None:
            sols = {k: v[max_data_ind] for k, v in sols_dict.items()}
        # add channel dim (C) and flatten 9 x 9 -> 81 (B)
        # N x B x C x H x W
        X = X.view(X.shape[0], -1, 1, X.shape[-2], X.shape[-1])
        # N x B * C
        givens = (y != 0).sum(-1).bool().reshape(y.shape[0], -1)
        # proper one-hot encoding for '0'
        ind_zero = torch.where(y.sum(-1) == 0)
        y = torch.from_numpy(np.pad(y, [(0, 0), (0, 0), (0, 0), (1, 0)]))
        y[ind_zero][:, 0] = 1
        # argmax to convert to int base labels
        y = y.argmax(-1)
        # N x B
        # y = y.view(y.shape[0], -1)

        # merge batch into one image
        X = torch.stack([make_grid(x, nrow=9) for x in X])[:, 1, :, :]
        # save_image(X[0], 'toast_data_sudoku_O.jpg')
        # print('debug-image saved')
        # unsqueeze channel dim
        x = X.unsqueeze(1)
        # convert back imgs to numpy for compatibility reasons
        to_pil = torchvision.transforms.ToPILImage()
        x = np.array([np.array(to_pil(img)).astype(np.uint8) for img in x])
        # x = np.expand_dims(x, axis=len(x.shape))
        # print("debug x.shape in mnist data", x.shape)

    else:
        raise NotImplementedError()

    raw_data["images"] = x
    raw_data["labels"] = y
    raw_data["solutions"] = sols
    return raw_data


def get_data_loaders(
    raw_data_dict: dict,
    backbone: BackboneArch,
    solver: ConstraintSolver,
    kwargs_dataset_transforms: dict,
    conf: dict,
    seed_train_val=244,
):
    x = raw_data_dict["images"]
    y = raw_data_dict["labels"]
    sols = raw_data_dict["solutions"]
    rng = raw_data_dict["rng"]
    dataset_cls = ImgDataset
    is_cell_lvl_arch = backbone not in IS_WHOLE
    if is_cell_lvl_arch:
        dataset_cls = CellDataset

    train_index, test_index = train_test_split(
        np.arange(len(x)), train_size=0.7, random_state=243
    )
    train_index, valid_index = train_test_split(
        train_index, train_size=0.8, random_state=seed_train_val
    )
    if conf.get("train_size", None) is not None:
        train_size = conf.get("train_size", len(train_index))
        train_index = rng.choice(train_index, train_size, replace=False)
    pad = conf.get("overlap_pad", 0.0)
    n_errors = conf.get("n_errors", 0)
    b_augment_sudoku = conf.get("sudoku_augment", False)
    b_corrupt_sudoku = False
    if n_errors > 0:
        assert conf["dataset"] == Datasets.VSUDOKU_APP, (
            f"Cannot build corrupted instances with dataset {conf['dataset']}"
        )
        b_corrupt_sudoku = True

    def filter_transforms(item):
        k, _ = item
        return "transform" in k

    def pop_normalize(list_transforms: list):
        for i, t in enumerate(list_transforms):
            if isinstance(t, T.Normalize):
                list_transforms.pop(i)
                return
            if isinstance(t, T.Compose):
                # find if normalize layer is in this block
                for ii, layer in enumerate(t.transforms):
                    if isinstance(layer, T.Normalize):
                        t.transforms.pop(ii)

                        # print('found/pop normalize')
                        return

    imbalance_policy = conf.get("imbalance_policy", None)
    is_stratified_sampling = (
        imbalance_policy == ImbalancePolicy.DOWNSAMPLING
        or imbalance_policy == ImbalancePolicy.UPSAMPLING
    )
    # build train/valid/test
    list_data = []

    # try to pre-compute solutions here once
    if sols is None:
        print("pre-compute SOLS...")
        solver_params = [{"params": 0} for _ in range(len(y))]
        with ProcessPoolExecutor(max_workers=conf.get("n_workers", 1)) as executor:
            sols_list = list(
                tqdm(executor.map(solver.solve, y, chunksize=100), total=len(y))
            )
        dict_of_lists_of_tensors = {
            key: [
                torch.from_numpy(d[key]).float()
                if isinstance(d[key], np.ndarray)
                else torch.tensor([d[key]])
                for d in sols_list
            ]
            for key in sols_list[0].keys()
        }

        sols = {
            key: torch.stack(tensor_list)
            if isinstance(tensor_list[0], torch.Tensor)
            else tensor_list
            for key, tensor_list in dict_of_lists_of_tensors.items()
        }
        folder_name = {
            Datasets.VSUDOKU_APP: "app",
            Datasets.VSUDOKU_MNIST: "mnist",
            Datasets.VSUDOKU_NEURASP: "neurasp",
        }
        torch.save(
            sols,
            os.path.join("data", "raw_data", folder_name[conf["dataset"]], "sols.pt"),
        )

    for i, index in enumerate([train_index, valid_index, test_index]):
        data_transforms = dict(**kwargs_dataset_transforms)
        if i > 0:
            # only retain preprocessing step, remove data augment for valid and test data
            data_transforms = {
                k: v
                for k, v in filter(filter_transforms, kwargs_dataset_transforms.items())
            }
        if b_augment_sudoku or b_corrupt_sudoku:
            assert conf["dataset"] == Datasets.VSUDOKU_APP, (
                "sudoku augment only available for Sudoku Assistant App Dataset"
            )
            assert "combined_labels" in raw_data_dict, (
                "missing combined labels to build corrupted instances"
            )
            # build corrupter
            # use sudoku assistant to build store
            combined_labels = raw_data_dict["combined_labels"]
            x_app = x[index]
            y_app = combined_labels[index]
            sols_app = None if sols is None else {k: v[index] for k, v in sols.items()}
            data_transform_corrupt = copy.deepcopy(data_transforms)
            pop_normalize(data_transform_corrupt["img_transform"].transforms)
            data_store = CellIndexedDataset(
                x_app,
                y_app,
                solver,
                shape=conf["puzzle_shape"],
                sols=sols_app,
                **data_transform_corrupt,
            )
            # avoid double normalize
            store = {
                # store with handwritten
                k: data_store.cells_img[data_store.labels == k + 9, :, :]
                for k in range(1, 10)
            }

        if i == 2:
            if b_corrupt_sudoku:
                corrupter = CorruptSudoku(
                    store,
                    num_errors=n_errors,
                    seed=seed_train_val,
                    font_clf=conf.get("fontstyle", False),
                )
                # filter out instances that don't have empty spaces
                data_transforms["corrupter"] = corrupter
        elif i == 1:
            if conf.get("corrupt_valid", False):
                corrupter = CorruptSudoku(
                    store, num_errors=n_errors, seed=seed_train_val
                )
                # filter out instances that don't have empty spaces
                data_transforms["corrupter"] = corrupter
        else:  # i == 0:
            pass

        if is_stratified_sampling and i == 0 and is_cell_lvl_arch:
            # train data: use labels to index data per-cell (enable balanced sampler)
            list_data.append(
                CellIndexedDataset(
                    x[index],
                    y[index],
                    solver,
                    shape=conf["puzzle_shape"],
                    sols=None
                    if sols is None
                    else {k: v[index] for k, v in sols.items()},
                    **data_transforms,
                )
            )
        else:
            list_data.append(
                dataset_cls(
                    x[index],
                    y[index],
                    solver,
                    shape=conf["puzzle_shape"],
                    sols=None
                    if sols is None
                    else {k: v[index] for k, v in sols.items()},
                    **data_transforms,
                )
            )

        if hasattr(list_data[-1], "overlap_pad"):  # and i == 0:
            # overlap padding (only for train set)
            list_data[-1].overlap_pad = pad

    # random or balanced sampler
    sampler = RandomSampler(list_data[0])

    def collate_subsampling(list_of_patch_target):
        """Merge subsampled cell patches together"""
        patches = []
        targets = []
        for p, t in list_of_patch_target:
            patches.append(p.unsqueeze(0))
            targets.append(t)

        x = torch.vstack(
            [
                make_grid(torch.vstack(patches[i : i + 81]), nrow=9, padding=0).mean(0)
                for i in range(0, len(patches), 81)
            ]
        )
        target = {
            "label": torch.tensor([t["label"] for t in targets]).reshape(-1, 81),
        }
        return x, target

    if is_stratified_sampling and is_cell_lvl_arch:
        sampler = BalanceClassSampler(
            list_data[0].y.reshape(-1).numpy().tolist(), imbalance_policy.value.lower()
        )

    train_data, valid_data, test_data = list_data
    # corrupt sudokus in test set

    # ensure stratified sampling still occurs in batches during training
    train_batch_size = conf.get("train_batch_size", 1)
    n_max_workers = 0  # min(conf.get('train_batch_size', 0), 8)
    if is_stratified_sampling:
        trainloader = DataLoader(
            train_data,
            batch_size=81 * train_batch_size,
            sampler=sampler,
            num_workers=n_max_workers,
            drop_last=train_batch_size is not None,
            collate_fn=collate_subsampling,
        )

    else:
        trainloader = DataLoader(
            train_data,
            batch_size=train_batch_size,
            sampler=sampler,
            num_workers=n_max_workers,
            drop_last=train_batch_size is not None,
        )

    validloader = DataLoader(valid_data, batch_size=1)
    testloader = DataLoader(test_data, batch_size=1, shuffle=False)
    return trainloader, validloader, testloader


def build_data(
    dataset: Datasets,
    backbone: BackboneArch,
    solver: ConstraintSolver,
    kwargs_dataset_transforms: dict,
    conf: dict,
    seed_train_val=244,
):
    raw_data_dict = get_data(
        dataset,
        conf["puzzle_shape"],
        conf["max_data_size"],
        conf.get("fontstyle", False),
    )
    return get_data_loaders(
        raw_data_dict, backbone, solver, kwargs_dataset_transforms, conf, seed_train_val
    )
