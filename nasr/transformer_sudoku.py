from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from nasr.transformer import Transformer
from pathlib import Path

class TransformerSudoku(nn.Module):
    def __init__(self, cnn:nn.Module, block_len=256, num_classes=10, puzzle_shape=(9,9), **kwargs):
        super().__init__()
        self.saved_log_probs = []
        self.rewards = []
        self.perception = cnn
        self.nn_solver = Transformer(in_chans=num_classes, num_classes=9,    
                        block_len=block_len, embed_dim=192, depth=4, num_heads=3, mlp_ratio=4,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        self.mask_nn = Transformer(in_chans=9, num_classes=1,
                        block_len=block_len, embed_dim=192, depth=4, num_heads=3, mlp_ratio=4,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        self.num_classes = num_classes
        self.puzzle_shape = torch.tensor(puzzle_shape)
        self.n_items = block_len
        

    def load_pretrained_models(self, dataset):
        
        #perception_path = 'outputs/perception/'+dataset+'/checkpoint_best.pth'
        nn_sol_path =  Path(__file__).parent / 'weights/solvernn/'  / dataset / 'checkpoint_best.pth'
        mask_nn_path = Path(__file__).parent / 'weights/mask/' / dataset / 'checkpoint_best.pth'

        #self.perception.load_state_dict(torch.load(perception_path, map_location='cpu'))
        missing_solve_nn = self.nn_solver.load_state_dict(torch.load(nn_sol_path, map_location='cpu'))
        print('debug missing solverNN: ', missing_solve_nn)
        missing_mask_nn = self.mask_nn.load_state_dict(torch.load(mask_nn_path, map_location='cpu'))
        print('missing mask', missing_mask_nn)

    def forward(self, x, nasr = 'rl'):
        B = len(x)
        if nasr == 'pretrained':
            # for eval of pretrained pipeline (NASR w/o RL)
            assert not bool(self.training), f'{nasr} is available only to evaluate. If you want to train it, use the RL pipeline.'
            x0 = self.perception.forward(x)['predictions'].view(B, self.n_items, -1)
            # original code assumes output of shape B x 81 x 10
            a = x0.argmax(dim=-1)
            x1 = F.one_hot(a,num_classes=self.num_classes).to(torch.float32)
            x2 = self.nn_solver.forward(x1)
            b = x2.argmax(dim=-1)+1
            # # nn solver output shifted to 1-9. Mask NN expect 0-9 domain
            # # padding last dim of nn solver output with a leading column of zeros
            x2 = F.one_hot(b,num_classes=10)
            x2 = x2[:,:,1:].to(torch.float32).to(torch.float32)
            x3 = self.mask_nn.forward(x2)
        else:
            # for traning with RL and eval with RL (NASR with RL)
            assert nasr == 'rl', f'{nasr} do not exists, choose between rl and pretrained'
            x0 = self.perception.forward(x)['predictions'].view(B, self.n_items, -1)
            x2 = self.nn_solver.forward(x0)
            x2 = F.softmax(x2, dim=-1)
            #x2 = F.gumbel_softmax(x2, tau = 1, hard=True, dim=-1)
            x3 = self.mask_nn.forward(x2)
        return {
            'predictions':x0,
            'solver_nn_output':x2,
            'mask_nn_output':x3,
        }


def get_model(block_len=256, **kwargs):
    model = TransformerSudoku(block_len=block_len, **kwargs)
    return model