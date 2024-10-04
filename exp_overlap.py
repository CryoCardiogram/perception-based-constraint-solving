from collections import Counter
from typing import Iterable
import lightning as pl
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from methods.pbcs import PBCSweightedCE
from config import BackboneArch, Datasets, ImbalancePolicy, IS_WHOLE
from neuralnet import get_model
from constraint_solver.sudoku_solver import SudokuSolver, ObjectiveHCOP
import numpy as np
import torch
from utee.databuilder import build_data
from utee.img_processing import GridSplitter
import argparse
import pathlib
from collections import OrderedDict
import concurrent
from concurrent.futures import ProcessPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", help="Specify output dir path", default='output')
    parser.add_argument('-n', '--runs', default=1, type=int, help='number of iterations')
    parser.add_argument('--imbalance_policy', type=ImbalancePolicy, default=None, help=f'enable to train without stratified batches, should be one of {[p.value for p in ImbalancePolicy]}')
    parser.add_argument('--max_total_epochs', type=int, default=50, help='number of epochs (default=50)')
    parser.add_argument("-b", "--arch",type=BackboneArch, default=[], nargs='+', help=f'Specify backbone arch to try. Should be one or more of {[b.value for b in BackboneArch]}')
    parser.add_argument('--dataset', type=Datasets, default=Datasets.VSUDOKU_APP, help=f'dataset to consider for the run, should be one of {[d.value for d in Datasets]}')
    parser.add_argument('--overlap_pad', type=float, default=0.0, help='Overlapping padding percentage (for cell-lvl architecture only). Default 0.0')
    parser.add_argument('--fc_hidden_layers', nargs='+', default=[512], help='hidden layers of dense classifier, default to [512]')
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--max_data_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--no_cell_aug', action='store_true', default=False)
    parser.add_argument('--no_img_aug', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--n_workers', type=int, default=1, help='number of parallel runs (try to parallelize seeds). Default to 1.')
    parser.add_argument('--frozen', action='store_true', help='toggle to disable training of backbone layers.')
    parser.add_argument('--cell_aug_at_whole', action='store_true', default=False, help='Enable to apply cell augmentation at full img (only when using whole-CNN). Default to False.')
    return OrderedDict(vars(parser.parse_args()))

def main(args):
    N_CLASSES = 10
    PUZZLE_SHAPE = (9, 9)
    objective_builder = ObjectiveHCOP()
    cs = SudokuSolver(PUZZLE_SHAPE, np.arange(N_CLASSES), objective_builder, time_limit=30)

    print('debug exp_overlap args', args)
    # datasets

    for n in args['seeds']:
        net, kwargs_dataset = get_model(
            args['arch'], 
            args['dataset'],
            N_CLASSES,
            PUZZLE_SHAPE,
            batchnorm=args['batchnorm'],
            fc_hidden_layers=args['fc_hidden_layers'],
            use_cell_augments=not args['no_cell_aug'],
            use_img_augments= not args['no_img_aug'],
            frozen=args['frozen'],
            use_cell_aug_at_img=args['cell_aug_at_whole'],
        )

        print(net)
        conf = dict(**args)
        conf['puzzle_shape'] = PUZZLE_SHAPE
        conf['train_batch_size'] = args['batch_size']

        train_dl, valid_dl, test_dl = build_data(args['dataset'], args['arch'], cs, kwargs_dataset, conf, seed_train_val=n)
        #TODO code build_model function
        c_train = Counter(train_dl.dataset.y.flatten().tolist())
        print('counter train: ', c_train)
        classes_counts = np.array([c_train[k] for k in sorted(c_train.keys())])
        train_class_weights = torch.from_numpy(1 - (classes_counts / classes_counts.sum())).float()
        if conf['imbalance_policy'] != ImbalancePolicy.WEIGHTEDCE:
            train_class_weights = None

        starting_lr = args['lr']
        if starting_lr is None:
            starting_lr = 1e-3

        splitter = GridSplitter(PUZZLE_SHAPE, torch.nn.Identity()) 
        if backbone in IS_WHOLE:
            splitter = torch.nn.Identity()

        model = PBCSweightedCE(
            net,
            cs,
            splitter,
            lr=starting_lr,
            num_pred_classes=N_CLASSES,
            puzzle_shape=conf['puzzle_shape'],
            train_labels_weights=train_class_weights,
            hparams=conf
        )
        
        es = EarlyStopping('val_cell_accuracy', mode='max', patience=5)
        checkpoint_cb = ModelCheckpoint(save_top_k=1, monitor="val_cell_accuracy", mode="max")
        logdir = str(pathlib.Path(args['out'], 'overlap').resolve())
        conf_name = str(pathlib.Path(*[f'{k}@{v}' for k,v in reversed(conf.items()) if not isinstance(v, Iterable) and k != 'seeds']))
        conf_dir = str(pathlib.Path(logdir, conf_name).resolve())
        pathlib.Path(conf_dir).mkdir(parents=True, exist_ok=True)
        logger = CSVLogger(
            save_dir=logdir,
            name=conf_dir,
            version=n,
        )
        
        trainer = pl.Trainer(
            logger = logger,
            max_epochs=args['max_total_epochs'],
            log_every_n_steps=20,
            callbacks=[es, checkpoint_cb],
            accelerator='auto',
            devices='auto',
            auto_lr_find=True,
        )
        if args['lr'] is None:
            results = trainer.tune(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
            print('learning rate suggested', results)
        trainer.validate(model, valid_dl)
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
        print(checkpoint_cb.best_model_path)  
        print(checkpoint_cb.best_model_score)
        # load best model before testing
        CKPT_state_dict = torch.load(checkpoint_cb.best_model_path)
        layer_names = list(net.state_dict().keys())
        to_load = OrderedDict(**{k.split('dnn.')[1]:v for k,v in CKPT_state_dict['state_dict'].items() if len(k.split('dnn.')) > 1 and k.split('dnn.')[1] in layer_names})
        model.dnn.load_state_dict(to_load)
        model.save_hyperparameters(conf)
        trainer.test(model, test_dl)


if __name__ == '__main__': 
    args = parse_args()
    print('debug exp arch args', args)
    # parallelize runs
    hparams = [] 
    for seed in range(args['runs']): 
        
        for backbone in args['arch']:
            param = OrderedDict(**args)
            param['seeds'] = [seed]
            param['arch'] = backbone
            hparams.append(param)
    
    with ProcessPoolExecutor(max_workers=args['n_workers']) as executor:
        futures = [executor.submit(main, params) for params in reversed(hparams)]
        # for f in futures:
        #     f.add_done_callback(lambda f:pbar.update())
        for future in concurrent.futures.as_completed(futures):
            future.result()