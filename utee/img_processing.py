import torch.nn.functional as F
import math
from torchvision.utils import save_image, make_grid
import numpy as np 
import cpmpy as cp 
from PIL import Image
from abc import abstractmethod
from typing import Iterable, Tuple
import cv2
import torchvision.transforms as T 
import torch
from itertools import product 
from constraint_solver.sudoku_solver import reduce_to_ten_classes

def split(x, grid_shape, pad=0.0):
    '''
    split padded grid-image and iterate over resulting cells images.

    Args:
        x: (1 x C x H x W) image tensor

        grid_shape: (S_h, S_w) puzzle shape

        pad (float): overlap padding scale  (default: 0.0)
    '''
    p_h, p_w = x.shape[-2:]
    C = x.shape[1]
    B = x.shape[0]
    # cell dims
    kernel_h = p_h // grid_shape[0]
    kernel_w = p_w // grid_shape[1]
    # prepad to take the entire image
    basepad_h = p_h % grid_shape[0]
    basepad_w = p_w % grid_shape[1]

    overlap_h = math.ceil(kernel_w * pad)
    overlap_w = math.ceil(kernel_h * pad)
    new_kernel_w = kernel_w + overlap_w * 2
    new_kernel_h = kernel_h + overlap_h * 2
    pad_w = basepad_w + overlap_w * 2
    pad_h = basepad_h + overlap_h * 2

    padded_img = F.pad(x, (pad_w // 2, pad_w // 2,
                           pad_h // 2, pad_h // 2), mode='replicate')
    patches = padded_img.unfold(
        2, new_kernel_h, kernel_h).unfold(
        3, new_kernel_w, kernel_w)
    for cellimg in patches.reshape(B, C, -1, new_kernel_h, new_kernel_w).permute(
            0, 2, 1, 3, 4).view(-1, C, new_kernel_h, new_kernel_w):
        yield cellimg

class GridSplitter(torch.nn.Module):
    def __init__(self, grid_shape,cell_transforms, pad=0) -> None:
        super(GridSplitter, self).__init__()
        self.grid_shape = grid_shape
        self.cell_transforms = cell_transforms
        self.pad = pad

    def forward(self, img: torch.Tensor):
        B = img.shape[0]
        p_h, p_w = img.shape[-2:]
        C = img.shape[1]
        kernel_h = p_h // self.grid_shape[0]
        kernel_w = p_w // self.grid_shape[1]
        basepad_h = p_h % self.grid_shape[0]
        basepad_w = p_w % self.grid_shape[1]

        overlap_h = math.ceil(kernel_w * self.pad)
        overlap_w = math.ceil(kernel_h * self.pad)
        new_kernel_w = kernel_w + overlap_w * 2
        new_kernel_h = kernel_h + overlap_h * 2
        pad_w = basepad_w + overlap_w * 2
        pad_h = basepad_h + overlap_h * 2

        padded_img = F.pad(img, (pad_w // 2, pad_w // 2,
                            pad_h // 2, pad_h // 2), mode='replicate')
        patches = padded_img.unfold(
            2, new_kernel_h, kernel_h).unfold(
            3, new_kernel_w, kernel_w)
        return self.cell_transforms(patches.reshape(B, C, -1, new_kernel_h, new_kernel_w).permute(
                0, 2, 1, 3, 4).view(B, -1, C, new_kernel_h, new_kernel_w))
            

class MeshSplitter(torch.nn.Module):
    """This splitter segments the original by following the meshgrid.

        Args:
            grid_shape (Iterable): Sudoku grid shape (9,9)
            cell_transforms (torch.nn.Module): image preprocessing pipeline for cell patches.
            meshgrid (Iterable[Tuple[Point]]): List of 4 tuples of points (x, y), representing  corner coordinates of each patch. 
    """
    def __init__(self, grid_shape, cell_transforms: torch.nn.Module, meshgrid:Iterable[Iterable[Tuple[int,int]]]) -> None:
        super().__init__(grid_shape, cell_transforms)
        self.meshgrid = meshgrid

    def forward(self, img:torch.Tensor):
        # for each mesh
        for mesh in self.meshgrid:
            # find bbox
            polygon = np.array(mesh)
            x,y,w,h = cv2.boundingRect(polygon)
            # crop bbox
            yield self.cell_transforms(img.squeeze()[:, y:y+h, x:x+w]).unsqueeze(0)



def extract_grid(img:np.ndarray):
        height = img.shape[0]
        width = img.shape[1]
        #PORTRAIT IMAGE
        if (width < height):
            imageBGR_cropped = img[math.floor(height / 2 - width * 0.375):math.ceil(height / 2 + width * 0.375),
                                        math.floor(width * 0.125):math.ceil(width - width * 0.125),
                                        :]
        #LANDSCAPE IMAGE                                
        else:
               imageBGR_cropped = img[math.floor(height * 0.125):math.ceil(height - height * 0.125),
                                           math.floor(width / 2 - height * 0.375):math.ceil(width / 2 + height * 0.375), 
                                        :]
        return imageBGR_cropped

def debug_save_cellimgs(xtorch, fname='debug-grid-img', nrow=9):
    grid_img = make_grid(xtorch, nrow=nrow)
    save_image(grid_img, f'{fname}.jpg')

class FullSudokuException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class NonValidSudokuException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class CorruptSudoku:
    """_summary_

    Args:
        store (dict): maps label-id -> List[cell_image]. Sample from these to build corrupted sudoku.
        num_errors (int): amount of erroneously filled cells. Default 0. 
        seed (int): seed. Default None.
        smart (bool): Enable to use a solver to generate erroneous sudoku. Otherwise, simply add duplicate digis on same line/row/block. Defaults to True.
        font_clf (bool): Enable to specify that perception module do font classification. Default to True.
    """
    def __init__(self, store:dict, num_errors=0, seed=None, smart=True, font_clf=True) -> None:
        self.num_errors = num_errors
        self.store = store
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.boxes = None
        self.topil = T.ToPILImage()
        self.totensor = T.ToTensor()
        self.PUZZLE_SHAPE = (9,9)
        self.smart = smart
        self.font_clf = font_clf
        self.__filler_calls = 0
    
    def compute_boxes(self, img:torch.Tensor, n = 9):
        h,w = img.shape[1:] 
        d_h = h // n
        d_w = w // n 
        boxes = []
        grid = product(range(0, h-h%d_h, d_h), range(0, w-w%d_w, d_w))
        for i, j in grid: 
            # xxyy format
            box = (j, j+d_w, i, i+d_h)
            boxes.append(box)
        self.boxes = np.array(boxes).reshape(*self.PUZZLE_SHAPE, -1)


    def make_sudoku_unsat(self, givens):
        givens = np.array(givens)
        dim,_ = givens.shape

        cells = cp.intvar(1,9, givens.shape, name="cells")
        idxes = [(i,j) for (i,j),v  in np.ndenumerate(givens) if v == 0]
        if len(idxes) == 0:
            raise FullSudokuException
        m = cp.Model(cells[givens != 0] == givens[givens != 0])
        m += [cp.AllDifferent(row) for row in cells]
        m += [cp.AllDifferent(col) for col in cells.T]
        bs = int(dim ** 0.5)
        for i in range(0,dim, bs):
            for j in range(0,dim, bs):
                m += cp.AllDifferent(cells[i:i+bs, j:j+bs])
        
        
        cp.SolverLookup.get('ortools', m).solve(stop_after_first_solution=True)
        sol = cells.value() 
        m += [cp.all(cells.flatten() != sol.flatten())]
        has_more_than_one = cp.SolverLookup.get('ortools', m).solve(stop_after_first_solution=True)
        if has_more_than_one:
            raise NonValidSudokuException
        count = 0
        while count <self.num_errors:
            # make 1 error
            i,j = self.rng.choice(idxes)
            options = set(range(1,10))
            # delete correct value
            options.remove(cells[i,j].value())
            # delete row options
            options -= set(givens[i])
            # delete col options
            options -= set(givens[:,j])
            # delete block options, (i,j) lies in block (bv,bh)
            bv, bh = i // bs, j // bs
            options -= set(givens[bv * bs:(bv+1)*bs, bh*bs:(bh+1)*bs].flatten())
            if len(options): # could be no options are left (unlikely, but still it can happen)
                givens[i,j] = self.rng.choice(list(options))
                m += cells[i,j] == givens[i,j]
                assert not m.solve()
                count += 1
        return givens

    def make_sudoku_unsat_dumb(self, givens): 
        givens = np.array(givens)
        dim,_ = givens.shape

        cells = cp.intvar(1,9, givens.shape, name="cells")
        idxes = [(i,j) for (i,j),v  in np.ndenumerate(givens) if v == 0]
        if len(idxes) == 0:
            raise FullSudokuException
        m = cp.Model(cells[givens != 0] == givens[givens != 0])
        m += [cp.AllDifferent(row) for row in cells]
        m += [cp.AllDifferent(col) for col in cells.T]
        bs = int(dim ** 0.5)
        for i in range(0,dim, bs):
            for j in range(0,dim, bs):
                m += cp.AllDifferent(cells[i:i+bs, j:j+bs])
        
        cp.SolverLookup.get('ortools', m).solve(stop_after_first_solution=True)
        sol = cells.value() 
        m += [cp.all(cells.flatten() != sol.flatten())]
        has_more_than_one = cp.SolverLookup.get('ortools', m).solve(stop_after_first_solution=True)
        if has_more_than_one:
            raise NonValidSudokuException
        count = 0
        while count < self.num_errors:
            # make 1 error
            i,j = self.rng.choice(idxes)
            options = set(range(1,10))
            # delete correct value
            options.remove(cells[i,j].value())
            # delete row options
            options -= set(givens[i])
            # delete col options
            options -= set(givens[:,j])
            # delete block options, (i,j) lies in block (bv,bh)
            bv, bh = i // bs, j // bs
            options -= set(givens[bv * bs:(bv+1)*bs, bh*bs:(bh+1)*bs].flatten())
            if len(options): # could be no options are left (unlikely, but still it can happen)
                givens[i,j] = self.rng.choice(list(options))
                m += cells[i,j] == givens[i,j]
                assert not m.solve()
                count += 1

    def __call__(self, input_img:torch.Tensor, target):
        assert isinstance(input_img, torch.Tensor), f'expected Tensor, got {type(input_img)}'
        # get boxes coord xxyy
        if self.boxes is None:
            self.compute_boxes(input_img, 9)
        # gen corrupted sudoku
        instance = np.copy(target['label'].reshape(self.PUZZLE_SHAPE).numpy())
        # has labels 0->18, reduce to 0->9
        givens = np.copy(instance)
        givens[givens > 9] -= 9
        try:
            corrupted = self.make_sudoku_unsat(givens)
        except Exception as e:
            if isinstance(e, FullSudokuException):
                #print('no empty cells', target['label'].reshape(self.PUZZLE_SHAPE))
                return input_img, target
            elif isinstance(e, NonValidSudokuException):
                print('cannot modify this instance', target['label'].reshape(self.PUZZLE_SHAPE))
                return input_img, target
            else:    
                raise e
        err_input = np.where(corrupted != givens)
        # update label accordingly 
        instance[err_input] = corrupted[err_input]+9 if self.font_clf else corrupted[err_input]
        new_target = {'label':torch.from_numpy(instance), 'cs_output':target['cs_output'], 'incorrect':np.array(list(zip(*err_input)))}
        # print('debug corrupter: added ', new_target['incorrect'])
        # create mask and fill it with cell imgs sampled from the store
        mask = torch.zeros(3, *input_img.shape[1:])
        #mask[..., -1] = 0 # transparent background
        
        for t in zip(*err_input):
            label = corrupted[t].item()
            filler = self.store[label][self.rng.integers(0, len(self.store[label]))]
            box = self.boxes[t] 
            resize = T.Resize((box[1]-box[0], box[3]-box[2]))
            #save_image(filler, f"filler_img_before_{self.__filler_calls}.jpg")
            filler_rgb = resize(filler)
            #save_image(filler_rgb, f"filler_img_{self.__filler_calls}.jpg")
            self.__filler_calls += 1
            mask[:3,box[2]:box[3],box[0]:box[1]] = filler_rgb
            

        # paste mask onto the original image, in black
        mask_rgba = torch.vstack([mask, torch.zeros((1, *input_img.shape[1:]))])
        R,G,B,A = mask_rgba 
        #is_white = (R > 0.6) | (G > 0.6) | (B > 0.6) 
        is_bg = (R==0) & (G ==0 ) & (B == 0)
        #mask_rgba[:3, is_white] = self.rng.uniform(0.1,0.9)#0.3 # black
        A[~is_bg] = 1 # foreground
        # _, axes = plt.subplots(1,3, figsize=(3,9))
        # axes[0].imshow(np.array(self.topil(mask_rgba[...,:-1])))
        # axes[1].imshow(mask_rgba[-1])
        # axes[2].imshow(is_white)
        im = self.topil(mask_rgba)
        img = self.topil(input_img).copy()
        img.paste(im, mask=im)
        return self.totensor(img), new_target
    

class AugmentSudoku(CorruptSudoku):
    """Data augmentation for visual sudoku
        Randomly add handwritten digit in empty cell
    Args:
        
    """

    def make_sudoku_unsat(self, givens):
        givens = np.array(givens)
        dim,_ = givens.shape

        cells = cp.intvar(1,9, givens.shape, name="cells")
        idxes = [(i,j) for (i,j),v  in np.ndenumerate(givens) if v == 0]
        if len(idxes) == 0:
            raise FullSudokuException
        m = cp.Model(cells[givens != 0] == givens[givens != 0])
        m += [cp.AllDifferent(row) for row in cells]
        m += [cp.AllDifferent(col) for col in cells.T]
        bs = int(dim ** 0.5)
        for i in range(0,dim, bs):
            for j in range(0,dim, bs):
                m += cp.AllDifferent(cells[i:i+bs, j:j+bs])
        
        
        cp.SolverLookup.get('ortools', m).solve(stop_after_first_solution=True)
        sol = cells.value() 
        m += [cp.all(cells.flatten() != sol.flatten())]
        has_more_than_one = cp.SolverLookup.get('ortools', m).solve(stop_after_first_solution=True)
        if has_more_than_one:
            raise NonValidSudokuException
        count = 0
        for i,j in self.rng.choice(idxes, self.num_errors, replace=False):
            givens[i,j] = sol[i,j]
            count += 1
        return givens
    
    def __call__(self, input_img, target):
        img, target = super().__call__(input_img, target)
        #print(target.keys())
        try:
            # messes the eval
            target.pop('incorrect')
        except:
            pass
        return img, target