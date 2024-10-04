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
class SafeLoaderIgnore(yaml.SafeLoader):
    def ignore_unknown(self, node):
        return None

def unknown(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        constructor = loader.__class__.construct_scalar
    elif isinstance(node, yaml.SequenceNode):
        constructor = loader.__class__.construct_sequence
    elif isinstance(node, yaml.MappingNode):
        constructor = loader.__class__.construct_mapping

    data = constructor(loader, node)

    return data


def read_csv_run(run_dir):
        # yaml.add_multi_constructor('!', unknown, lo)
        # yaml.add_multi_constructor('tag:', unknown)
    SafeLoaderIgnore.add_constructor(None, SafeLoaderIgnore.ignore_unknown)
    # loader.add_constructor('!!python/object/apply:numpy.core.multiarray._reconstruct', loader.ignore_unknown)
    # loader.add_constructor('!!python/object:dfl.losses.regret.AbsoluteRegret', loader.ignore_unknown)
    # loader.add_constructor('!!python/name:None.apply', loader.ignore_unknown)
    with open(os.path.join(run_dir, 'hparams.yaml'), 'r') as f:
        hparams = yaml.load(f, Loader=SafeLoaderIgnore)#yaml.FullLoader)
    df = pd.read_csv(os.path.join(run_dir, 'metrics.csv'))
    for k,v in hparams.items():
        if isinstance(v, list) or isinstance(v, tuple):
            df[k] = str(v)
        else:
            df[k] = v 
    df['run_dir'] = run_dir
    return df 

def get_all_runs(exp_dir):
    dfs = []
    for dirname, _, files in os.walk(pathlib.Path(exp_dir)):
        #print(files)
        for filename in filter(lambda n: n.split('.')[-1] == 'csv', files):
            #print(filename)
            #path = os.path.join(dirname,filename)
            df = read_csv_run(dirname)
            #print(dirname)
            # path_attributes =  dirname.split(os.path.sep)
            
            # for k,v in [ param.split('--') for param in filter(lambda t: '--' in t, path_attributes[:-2])]:
            #     df[k]=v 
            #     df['method'] = path_attributes[-2]
            dfs.append(df)


    df_all_params = pd.concat(dfs).reset_index()
    #df_all_params.tail(40)
    return df_all_params


def _get_run_path(key:tuple, table:pd.DataFrame, output_path:str):
    run_path = pathlib.Path(output_path) / pathlib.Path(table.loc[key]['run_dir']) 
    weights_path = run_path / 'checkpoints' / os.listdir(run_path / 'checkpoints')[-1]
    #hparams_path = run_path / 'hparams.yaml'
    return weights_path

def _melt_and_aggregate(df_runs:pd.DataFrame, group:Iterable[str], value_vars:Iterable[str]):
    melted = df_runs.melt(id_vars=group, value_vars=value_vars).dropna()
    return melted.groupby(group[:-1]).max()

_exp_groups = {
        'overlap': ['dataset', 'arch', 'imbalance_policy', 'frozen', 'no_cell_aug', 'version', 'run_dir'],
        'no-aug': ['dataset', 'arch', 'imbalance_policy', 'frozen', 'no_cell_aug', 'version', 'run_dir'],
        'pretrain': ['dataset', 'arch', 'imbalance_policy', 'frozen', 'no_cell_aug', 'version', 'run_dir'],
        'font': ['dataset', 'backbone', 'imbalance_policy', 'frozen', 'sudoku_augment','seeds', 'run_dir'],
        'calibration': ['dataset', 'backbone', 'imbalance_policy', 'frozen', 'calibration','version','run_dir'],
        'debug-overlap':['dataset', 'backbone', 'imbalance_policy', 'frozen', 'no_cell_aug', 'version', 'run_dir'],#TODO remove this....
}

def get_run_weights(key:tuple, path_exp:str, conf:dict):
    """Get pre-trained weights of a specific runs.

    Args:
        key (tuple): experiment-specific tuple acting as index to select the pre-trained model.
        exp_path (str): path of the experiment subdir, should ends with one of ['overlap', 'pretrain', 'font']
    """
    exp = pathlib.Path(path_exp).parts[-1]
    if 'debug' in path_exp:
        exp = 'debug-overlap'
    
    if exp not in _exp_groups:
        print('pretrained path not recognized, return the given path')
        return path_exp
    # print('debug key exp?', exp)
    
    # print('debug path to experiments?', path_exp)
    # print('constaints: ', os.listdir(pathlib.Path(path_exp).absolute().resolve()))
    df = get_all_runs(path_exp)
    print('sanity check: opensky in dataframe?',df['dataset'].unique() )
    df['imbalance_policy'] = df['imbalance_policy'].apply(lambda x: 'None' if x==None else x.upper())
    # df['frozen'].fillna(False, inplace=True)
    df['frozen'] = df['frozen'].fillna(False) if 'frozen' in df.columns else False
    df['version'] = df['run_dir'].apply(lambda path: pathlib.Path(path).parts[-1])
    if 'calibration' in df.columns:
        df['calibration'] = df['calibration'].apply(lambda x: 'UNCAL' if x is None else x)
    #df['lr'] = df['lr'].fillna(None)

    # try to filter-out pre-trained NN (need to set lr correctly now)
    # print('debug lr in conf?', conf['lr'])
    # print('debug lr in df?', df['lr'].unique())
    # print('debug key?', key)
    if len(df[df['lr']==conf['lr']]) >0 and conf['lr'] is not None:
        print('debug look for lr-specific')
        df = df[df['lr']==conf['lr']]
    table = _melt_and_aggregate(df, _exp_groups[exp], ['test_cell_accuracy_solve'])
    # print(table)
    # try:
    weights_path = _get_run_path(key, table, path_exp)
    return weights_path
    # except Exception as e:
    #     print(f'wrong key format {key}, required something like this: ', table.index[0])
    #     print(f'error {e}')
    #     return path_exp

def get_key(conf:dict, experiment:str, seed:int=0):
    key = [
        str(conf['dataset']), 
        str(conf.get('arch', conf['backbone'])), 
        str(conf['imbalance_policy'])
    ]
    print('debug key init', key)
    key = [s.split('.')[-1] for s in key]

    key += [conf.get('frozen', False)] 
    print('debug key frozen', key)
    if pathlib.Path(experiment).parts[-1] == 'font':
        key += [ 
            conf.get('sudoku_augment', False), 
            seed
        ]
        print('debug key font: ', key)
    
    elif pathlib.Path(experiment).parts[-1] =='calibration':
        try:
            
            calib = str(conf['calibration'].value).upper()
        except: 
            calib = 'UNCAL'
        key += [
            calib, 
            f'version_{seed}'
        ]
    else: 
        key += [
            conf.get('no_cell_aug', False),
            f'version_{seed}'
        ]

    return tuple(key)

def load_best_model(checkpoint_path, dnn:torch.nn.Module):
    CKPT_state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    layer_names = list(dnn.state_dict().keys())
    to_load = OrderedDict(**{k.split('dnn.')[1]:v for k,v in CKPT_state_dict['state_dict'].items() if k.split('dnn.')[1] in layer_names})
    dnn.load_state_dict(to_load)

def load_pretrain(config:dict, seed:int, dnn:torch.nn.Module):
    """Load old pretrained weights into new shared cell cnn arch.
    """
    key = get_key(config, config['pretrained'], seed)
    #print('key for pretrain model?', key)
    checkpoint_path = get_run_weights(key, config['pretrained'], config)
    print('debug checkpoint file? ', checkpoint_path)

    # load checkpoint
    CKPT_state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # determine if saved weights in old or new format
    ckpt_first_layer_name:str = list(CKPT_state_dict['state_dict'].keys())[0]
    network = dnn 
    if hasattr(dnn, 'cell_net'):
        network = dnn.cell_net
    if 'cell_net.' in ckpt_first_layer_name:
    
        layer_names = list(network.state_dict().keys())
        to_load = OrderedDict(**{k.split('cell_net.')[-1]:v for k,v in CKPT_state_dict['state_dict'].items() if k.split('cell_net.')[-1] in layer_names})
    else:
        layer_names = list(network.state_dict().keys())
        #print(ckpt_first_layer_name.split('dnn.')[-1])
        to_load = OrderedDict(**{k.split('dnn.')[-1]:v for k,v in CKPT_state_dict['state_dict'].items() if k.split('dnn.')[-1] in layer_names})
        #print('debug chkp_first_layer_name in layer_names?', ckpt_first_layer_name, layer_names, ckpt_first_layer_name in layer_names)
    # additional check because I changed the architecture of calibrated CNN
    if any('digit' in l for l in layer_names) and all('digit' not in l for l in to_load.keys()):
        # add weights stored at dnn.calibration.* to calibration_digit.*
        for name, param in CKPT_state_dict['state_dict'].items():
            if name.startswith('dnn.calibration'):
                name_strip = name.split('dnn.calibration.')[1]
                to_load[f'calibration_digit.{name_strip}'] = param
        pass

    #print('saved layers used to load weights:', list(to_load.keys()) )
    #print('current layers that require pretrained weights:', layer_names)
    if hasattr(dnn, 'cell_net'):
        return dnn.cell_net.load_state_dict(to_load, strict=False)
    else:
        return dnn.load_state_dict(to_load, strict=False)

def load_shared_pretrain(config:dict, seed:int, sharedpatchnet:torch.nn.Module):
    key = get_key(config, config['pretrained'], seed)
    # print('key for pretrain model?', key)
    checkpoint_path = get_run_weights(key, config['pretrained'], config)
    # print('debug checkpoint file? ', checkpoint_path)

    CKPT_state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    layer_names = list(sharedpatchnet.state_dict().keys())
    to_load = OrderedDict(**{k.split('dnn.')[1]:v for k,v in CKPT_state_dict['state_dict'].items() if k.split('dnn.')[1] in layer_names})
    sharedpatchnet.load_state_dict(to_load)


def data_postprocessing(results_df):
    try:
        results_df['wildcard_tr'] = results_df['wildcard_tr'].fillna('None')
    except:
        results_df['wildcard_tr'] = 'None'
    try:
        results_df['wildcard_soft'] = results_df['wildcard_soft'].fillna(False)
    except:
        results_df['wildcard_soft'] = False
    try: 
        results_df['dynamic'] = results_df['dynamic'].fillna(False)
    except: 
        results_df['dynamic'] = False
    try: 
        results_df['calibration'] = results_df['calibration'].apply(lambda x: 'None' if x is None else x)
    except: 
        results_df['calibration'] = 'None'

    try: 
        results_df['imbalance_policy'] = results_df['imbalance_policy'].apply(lambda x: 'None' if x is None else x.capitalize())
    except:
        results_df['imbalance_policy'] = 'None'
    try:
        results_df['topk'] = results_df['topk'].fillna('None')
    except:
        results_df['topk'] = 'None'
    try:
        results_df['timeout'] = results_df['timeout'].fillna(30)
    except:
        results_df['timeout'] = 30
    try:
        results_df['precision'] = results_df['precision'].fillna(1e-4)
    except:
        results_df['precision'] = 1e-4
    try:
        results_df['nasr'] = results_df['nasr'].fillna(False)
    except:
        results_df['nasr'] = False


def set_method_name(row):
    name = ''
    # first part, solver: baseline, hybrid or higher-order, or related work
    if row['obfun'] == 4:
        solver = 'NeurASP'
    elif row['obfun'] == 5:
        solver =  'SwiPl'

    elif row['topk'] == 1:
        solver = 'baseline'
    elif row['solver'] == 'HOCOP(1)' or row['solver'] == 'HCOP':
        solver = 'hybrid'
    elif 'HOCOP' in row['solver']:
        solver = 'higher-order'
    elif 'wildcard' in row['solver']:
        if row['no_good']:
            solver = 'higher-order'
        else: 
            solver = 'hybrid' 
    else:
        solver = 'baseline'
    name += solver
    # then wildcard, either wildcard-prune, wildcard-soft, wildcard-dynamic-prune, wildcard-dynamic-soft
    if row['wildcard_tr'] != 'None' or row['dynamic']:
        wildcard = 'wildcard'
        if row['dynamic']:
            wildcard +='-dynamic'
        if row['wildcard_soft']:
            wildcard+='-soft'
        else:
            wildcard+='-prune'
        name += f" ({wildcard})"
    
    if row['nasr']:
        name = f'NASR-{name}'
    return name
