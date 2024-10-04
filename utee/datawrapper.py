import torch
import numpy as np
from functools import lru_cache
from typing import Iterable, Iterator, List
from torch.utils.data import Sampler
import torchvision.transforms as T
from constraint_solver.constraint_solver import ConstraintSolver
from utee.img_processing import split, CorruptSudoku, debug_save_cellimgs
import copy
from torchvision.utils import make_grid


def add_augment_before_cast_transform(transforms:List, augment:torch.nn.Module) -> List:
    has_normalize_tf = False
    for i,t in enumerate(transforms):
        if isinstance(t, T.Normalize):
            has_normalize_tf = True
            break
    new_t = copy.deepcopy(transforms)
    new_t.insert(-2 if has_normalize_tf else -1, augment)
    return new_t, i

class ImgDataset:
    """ Wrapper for whole-image dataset.

        Samples whole-image instances and corresponding solution.
    """

    def __init__(
            self,
            X,
            y: torch.Tensor,
            solver: ConstraintSolver,
            solver_params: Iterable = None,
            shape= (9,9),
            img_transform = T.Compose([]),
            img_augment = torch.nn.Identity(),
            corrupter: CorruptSudoku = None,
            **kwargs,
        ):
        """

        Args:
            X (np.ndArray): PIL image
            y (torch.Tensor): labels
            solver (ConstraintSolver): CP solver object
            solver_params (Iterable, optional): list of solver params (as dict) for each instance. Defaults to None.
            img_transform (nn.Module, optional): preprocessing to the whole image. Defaults to torch.nn.Identity().
            img_augment (nn.Module, optional): data augmentation for the whole image. Defaults to torch.nn.Identity().
            corrupter (CorrupSudoku, optional): add handwritten errors to whole image. Defaults to None.
        """
        self.X = X
        self.y = y
        try:
            tf_list, idx = add_augment_before_cast_transform(img_transform.transforms, img_augment)
            self.T = T.Compose(tf_list)
            self.tf_norm_idx = idx
        except:
            self.T = T.Compose([img_transform, img_augment])
            self.tf_norm_idx = -1

        if solver_params is None:
            solver_params = [{'params': 0} for _ in range(len(y))]
        
        if solver is not None and kwargs.get('sols', None) is None:
            self.sols = [solver.solve(yi, **p)
                for yi, p in zip(y, solver_params)
            ]
        else:
            if 'sols' in kwargs:
                self.sols = [{k:v[i] for k,v in kwargs['sols'].items()} for i in range(len(kwargs['sols']['runtime'])) ]
            else:
                self.sols = np.full_like(y, np.nan)
        self.corrupter = corrupter

    def __len__(self):
        return len(self.X)

    def _transform_img(self, img, target):
        x = img 
        # ugly insert corrupt transform at the right place
        for i, tf in enumerate(self.T.transforms[:self.tf_norm_idx]):
            x = tf(x)
        if self.corrupter is not None:
            x, target = self.corrupter(x, target)
        if self.tf_norm_idx == -1:
            return x, target 
        else: 
            x = T.Compose([*self.T.transforms[self.tf_norm_idx:]])(x)
            return x, target


    def __getitem__(self, index):
        # data augmentation
        
        target = {
            #'img': x,
            'label': self.y[index],
            'cs_output': self.sols[index]
        }
        x, target = self._transform_img(self.X[index], target)
        # if self.corrupter is not None:
        #     x, target = self.corrupter(x, target)
        img_idx = torch.zeros(*target['label'].shape)
        img_idx[:] = index
        target['img_idx'] = img_idx
        return x, target


class CellDataset:
    """ Wrapper for whole-image dataset.

        Samples cell-images instances by splitting the 
        corresponding full image. Return all cell-images 
        and their label as a batch. 
    """

    def __init__(
            self,
            X,
            y: torch.Tensor,
            solver: ConstraintSolver,
            solver_params: Iterable = None,
            shape: Iterable = (9,9),
            img_transform=T.Compose([]),
            cell_transform=T.Compose([]),
            img_augment=torch.nn.Identity(),
            cell_augment=torch.nn.Identity(),
            overlap_pad=0.0,
            corrupter: CorruptSudoku = None,
            **kwargs):
        """

        Args:
            X (np.ndArray): PIL image
            y (torch.Tensor): labels
            shape (Iterable): puzzle grid shape
            solver (ConstraintSolver): CP solver
            solver_params (Iterable, optional): list of solver params (as dict) for each instance. Defaults to None.
            img_transform (_type_, optional): preprocessing to whole image. Defaults to torch.nn.Identity().
            cell_transform (_type_, optional): preprocessing to cell image. Defaults to torch.nn.Identity().
            img_augment (_type_, optional): data augmentation to whole image. Defaults to torch.nn.Identity().
            cell_augment (_type_, optional): data augmentation to cell image. Defaults to torch.nn.Identity().
            overlap_pad (float, optional): overlap padding for cells. Defaults to 0.0.
            corrupter (CorrupSudoku, optional): add handwritten errors to whole image. Defaults to None.
        """
        # pop the final Normalize transforms 
        list_img_transforms,_ = add_augment_before_cast_transform(img_transform.transforms, img_augment)
        self.has_normalize_tf = False
        index_normalize = -1
        for i,t in enumerate(list_img_transforms):
            if isinstance(t, T.Normalize):
                self.has_normalize_tf = True
                index_normalize = i 
            if isinstance(t, T.Compose):
                # find if normalize layer is in this block
                for layer in t.transforms:
                    if isinstance(layer, T.Normalize):
                        self.has_normalize_tf = True 
                        index_normalize = i 

        normalize = list_img_transforms[index_normalize:] if self.has_normalize_tf else [torch.nn.Identity()]
        
        list_img_transforms = list_img_transforms[:index_normalize] if self.has_normalize_tf else list_img_transforms
        
        self.img_transform = T.Compose(list_img_transforms)
        self.corrupter = corrupter
        # pre solve training instances
        self.X = X#torch.stack([self.img_transform(x) for x in X])
        self.y = y
        if solver_params is None:
            solver_params = [{'params': 0} for _ in range(len(y))]
        if solver is not None and kwargs.get('sols', None) is None:
            self.sols = [solver.solve(yi, **p)
                for yi, p in zip(y, solver_params)
            ]
        else:
            if "sols" in kwargs:
                # self.sols = [{k:val for val in v} for k,v in kwargs['sols'].items()]
                self.sols = [{k:v[i] for k,v in kwargs['sols'].items()} for i in range(len(kwargs['sols']['runtime'])) ]
            else:
                self.sols = np.full_like(y, np.nan)
        # ToTensor transform done at cell lvl (convert to PIL for autoaugment first)
        self.cell_transform = T.Compose([T.ToPILImage(), cell_transform, cell_augment, T.ToTensor(), *normalize])
        # print("img transform in celldataset", self.img_transform)
        # print("cell transform in celldataset", self.cell_transform)
        self.shape = shape
        self.overlap_pad = overlap_pad

    def __len__(self):
        return len(self.X)

    def _transform_img(self, img, target):
        x = self.img_transform(img)
        if self.corrupter is not None:
            x, new_target = self.corrupter(x, target)
            return x, new_target
        return x, target

    @lru_cache(30) # use LRU to cache whole img in memory. This speeds up training considerably
    def __getitem_solo__(self, index):
        # imgs = [self.cell_transform(cellimg[0]) for cellimg in split(
        #      self.X[index].unsqueeze(0), self.shape, self.overlap_pad)]
        target = {
            'label': self.y[index].flatten(),
            'cs_output': self.sols[index]
        }
        whole_img, target = self._transform_img(self.X[index], target)
        img_idx = torch.zeros(*target['label'].shape)
        img_idx[:] = index
        target['img_idx'] = img_idx
        cell_imgs = [self.cell_transform(cellimg) for cellimg in split(
            whole_img.unsqueeze(0), self.shape, self.overlap_pad
        )]
        x = torch.stack(cell_imgs, 0)
        return x, target
    
    def __getitem__(self, index):
        x, target = self.__getitem_solo__(index)
        merged = make_grid(x, nrow=self.shape[0], padding=0)
        if x.size(1) == 1:  # single-channel images 
            merged = merged.mean(0).view(1, *merged.shape[-2:])
        #TODO same size as fullimg?
        # topil = T.ToPILImage()
        # x = self.img_transform(np.array(topil(x)))
        return merged, target


class CellIndexedDataset(CellDataset):
    """ Wrapper for whole-image dataset.


    Samples cell-image instances individually. 
    Required for `BalancedClassSampler`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gelem = self.shape[0] * self.shape[1]
         # pre-transform whole img to save up time
        self.targets = [ 
            {
                'label':self.y[i].flatten(),
                'cs_output':self.sols[i],
            } 
            for i in range(len(self.sols))
        ]
        self.imgs, self.targets=  list(zip(*[
            self._transform_img(self.X[i], self.targets[i])
            for i in range(len(self.sols))
        ]))
        self.cells_img = []
        self.labels = []
        for whole_img, target in zip(self.imgs, self.targets):
            cells = [self.cell_transform(cellimg) for cellimg in split(
                whole_img.unsqueeze(0), self.shape, self.overlap_pad
            )]
            x = torch.stack(cells, 0)
            self.cells_img.append(x)
            self.labels.append(target['label'].reshape(-1))
        self.cells_img = torch.cat(self.cells_img, 0)
        self.labels = torch.cat(self.labels, 0)

    def __len__(self):
        return len(self.X) * self.gelem

    def __getitem__(self, index):
        # img_idx = np.floor(index / (self.gelem)).astype(int)
        # cell_idx = index - img_idx * self.gelem
        cell_idx = index
        # img, target = super().__getitem__(img_idx)
        # x = img[cell_idx]#.unsqueeze(0)
        # y = target['label'][cell_idx]
        x = self.cells_img[cell_idx]
        y = self.labels[cell_idx]
        return x, {
            #'img':x,
            'label': y,
            'img_idx':index // 81
        }


class BalanceClassSampler(Sampler):
    """Abstraction over data sampler.
    Allows you to create stratified sample on unbalanced classes.
    """

    def __init__(self, labels: List[int], mode: str = "downsampling"):
        """
        Args:
            labels (List[int]): list of class label
                for each elem in the datasety
            mode (str): Strategy to balance classes.
                Must be one of [downsampling, upsampling]
        """
        super().__init__(labels)

        labels = np.array(labels)
        samples_per_class = {
            label: (labels == label).sum() for label in set(labels)
        }

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, str):
            assert mode in ["downsampling", "upsampling"]

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = (
                mode
                if isinstance(mode, int)
                else max(samples_per_class.values())
            )
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_ = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length


