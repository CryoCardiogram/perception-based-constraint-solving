import pathlib
import lightning as pl
from typing import Iterable
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from methods.calibration import CalibrationOnly
from methods.pbcs import PBCSweightedCE
from methods.fontstyle import FontStyleMT, FontStyle
from config import BackboneArch, Datasets, ImbalancePolicy, CalibrationType, IS_WHOLE, IS_MULTI
from neuralnet import get_model
from constraint_solver.sudoku_solver import SudokuSolver, ObjectiveHCOP
import numpy as np
import torch
from utee.databuilder import build_data
from utee.img_processing import GridSplitter
from utee.plot import load_pretrain
import argparse
from collections import OrderedDict
from utee.plot import get_all_runs, get_run_weights, get_key
import os


import concurrent
from concurrent.futures import ProcessPoolExecutor

def get_model_builder_Nclasses(conf):
    if conf['fontstyle']:
        if conf['backbone'] in IS_MULTI:
            return FontStyleMT, 10
        else:
            return FontStyle, 19
    else:
        return PBCSweightedCE, 10

def main(args):
    PUZZLE_SHAPE = (9, 9)
    is_extended = False 
    if args['pretrained'] is not None:
        is_extended = 'font' in args['pretrained']
    obj_builder = ObjectiveHCOP(is_extended=is_extended )
    cs = SudokuSolver(PUZZLE_SHAPE, np.arange(10), obj_builder, time_limit=30)



    print('debug calibration args', args)
    # datasets

    for seed in args['seeds']:
        conf = dict(**args)
        conf['puzzle_shape'] = PUZZLE_SHAPE
        conf['seeds'] = seed
        # keep it to false as we do not calibrate style classification?
        conf['fontstyle'] = False 
        if args['pretrained'] is not None:
            conf['fontstyle'] = 'font' in args['pretrained']

        model_builder, N_CLASSES = get_model_builder_Nclasses(conf)

        if conf['backbone'] in IS_MULTI:
            if conf['calibration'] == CalibrationType.DIRICHLET:
                # setup correct labels for dirichlet on merged output
                conf['fontstyle'] = True
                N_CLASSES = 19
            else:
                conf['fontstyle'] = False 
            obj_builder.is_extended = False

        
        
        net, kwargs_dataset = get_model(
            args['backbone'], 
            args['dataset'],
            10 if conf['calibration'] == CalibrationType.DIRICHLET and conf['backbone'] in IS_MULTI else N_CLASSES,
            PUZZLE_SHAPE,
            batchnorm=args['batchnorm'],
            fc_hidden_layers=args['fc_hidden_layers']
        )

        

        train_dl, valid_dl, test_dl = build_data(args['dataset'], args['backbone'], cs, kwargs_dataset, conf, seed_train_val=seed)
        starting_lr = args['lr']
        if args['lr'] is None:
            starting_lr = 1e-3


        splitter = GridSplitter(PUZZLE_SHAPE, torch.nn.Identity()) 
        if backbone in IS_WHOLE:
            splitter = torch.nn.Identity()
        model = model_builder(
            net,
            cs,
            splitter = splitter,
            lr=starting_lr,
            num_pred_classes=N_CLASSES,
            puzzle_shape=conf['puzzle_shape'],
            hparams=conf,
        )
        
        logdir = str(pathlib.Path(args['out'], 'calibration').resolve())
        conf_name = str(pathlib.Path(*[f'{k}@{v}' for k,v in reversed(conf.items()) if not isinstance(v, Iterable) and k != 'seeds']))
        conf_dir = str(pathlib.Path(logdir, conf_name).resolve())
        pathlib.Path(conf_dir).mkdir(parents=True, exist_ok=True)
        logger = CSVLogger(
            save_dir=logdir,
            name=conf_dir,
            version=seed,
        )
        
        es = EarlyStopping('val_cell_accuracy', mode='max', patience=5)

        trainer = pl.Trainer(
            logger = logger,
            max_epochs=args['max_total_epochs'],
            log_every_n_steps=20,
            callbacks=[es],
            accelerator='auto',
            devices='auto',
            inference_mode=False,
            #auto_lr_find=True,
        )
        if args['pretrained'] is None:
            if args['lr'] is None:
                results = trainer.tune(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
                print('learning rate suggested', results)
            trainer.validate(model, valid_dl)
            trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
        else:
            # find checkpoint for the same seed
            
            # key = (str(args['dataset']).split('.')[-1], str(args['backbone']).split('.')[-1], str(args['imbalance_policy']).split('.')[-1], 5)
            # key = get_key(conf, args['pretrained'], seed)
            # print('key for pretrain model?', key)
            # checkpoint_path = get_run_weights(key, args['pretrained'], conf)
            # print('debug checkpoint file? ', checkpoint_path)

            # # load checkpoint
            # CKPT_state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            # layer_names = list(net.state_dict().keys())
            # to_load = OrderedDict(**{k.split('dnn.')[1]:v for k,v in CKPT_state_dict['state_dict'].items() if k.split('dnn.')[1] in layer_names})
            # model.dnn.load_state_dict(to_load)

            incompat = load_pretrain(conf, seed, model.dnn)
            print('loaded weights execpts for keys', incompat)
            model.save_hyperparameters(conf)
        
        starting_lr = conf['lr']
        if conf['lr'] is None:
            starting_lr = 1e-3
        # calibration

        

        model_calib = CalibrationOnly(
            model.dnn, 
            cs, 
            splitter,
            lr=starting_lr,
            num_pred_classes=N_CLASSES,
            puzzle_shape=conf['puzzle_shape'],
            hparams=conf,
            calib_dl=valid_dl
        )
        

        es_cal = EarlyStopping(f'val_{args["early_stopping_criterion"]}', mode='min', patience=5, min_delta=1e-3)
        # checkpoint_cb = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
        calibrator = pl.Trainer(
            logger=logger,
            max_epochs=args['max_total_epochs'],
            log_every_n_steps=20,
            # num_sanity_val_steps=0,
            inference_mode=False,
            callbacks=[es_cal],#, checkpoint_cb],
            #auto_lr_find=True,
            accelerator='auto',
            devices='auto',
        )
        print(model_calib)
        calibrator.validate(model_calib, dataloaders=valid_dl)
        if conf['calibration'] is not None:
            if args['lr'] is None:
                print('debug valid dl?', len(valid_dl.dataset))
                print(valid_dl.dataset.y.shape)
                print(valid_dl.dataset.X.shape)
                results = calibrator.tune(model_calib, train_dataloaders=valid_dl, val_dataloaders=valid_dl)
                model_calib.hparams['lr'] = results['lr_find'].suggestion()
                conf['lr'] = results['lr_find'].suggestion()
                model_calib.save_hyperparameters(conf)
            

            calibrator.fit(model_calib, train_dataloaders=valid_dl, val_dataloaders=valid_dl)
            # CKPT_state_dict = torch.load(checkpoint_cb.best_model_path)
            # layer_names = list(net.state_dict().keys())
            # print(layer_names)
            # to_load = OrderedDict(**{k.split('dnn.')[1]:v for k,v in CKPT_state_dict['state_dict'].items() if len(k.split('dnn.')) > 1 and k.split('dnn.')[1] in layer_names})
            # model_calib.dnn.load_state_dict(to_load)
            # model_calib.save_hyperparameters(conf)
        calibrator.test(model_calib, dataloaders=test_dl)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", help="Specify output dir path", default='output')
    parser.add_argument('--dataset', type=Datasets, default=Datasets.VSUDOKU_MNIST, help=f'dataset to consider for the run, should be one of {[d.value for d in Datasets]}')
    parser.add_argument('--calibration', type=CalibrationType, nargs='+', default=[], help=f'calibration methods among {[b.value for b in CalibrationType]}. Default to None (uncalibrated)')
    parser.add_argument("-b", "--backbone",type=BackboneArch, default=[],nargs='+', help=f'Specify backbone arch to try. Should be one or more of {[b.value for b in BackboneArch]}')
    parser.add_argument('--imbalance_policy', type=ImbalancePolicy, default=ImbalancePolicy.UPSAMPLING, help=f'enable to train without stratified batches, should be one of {[p.value for p in ImbalancePolicy]}')
    parser.add_argument('--max_total_epochs', type=int, default=50, help='number of epochs (default=50)')
    parser.add_argument('--overlap_pad', type=float, default=0.0, help='Overlapping padding percentage (for cell-lvl architecture only). Default 0.0')
    parser.add_argument('--fc_hidden_layers', nargs='+', default=[512], help='hidden layers of dense classifier, default to [512]')
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--max_data_size', type=int, default=1000)
    parser.add_argument('-n', '--seeds', default=[244], nargs='+', type=int, help='sequence of seeds for kfolds. Default [544]')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--n_workers', type=int, default=1, help='number of parallel runs (try to parallelize seeds). Default to 1.')
    parser.add_argument('--pretrained', type=str, default=None,help='path to the `output` subfolder containing weights of the desired arch. Defaults to None')
    parser.add_argument('--early_stopping_criterion', type=str, default='loss')
    return OrderedDict(vars(parser.parse_args()))

if __name__ == '__main__': 
    args = parse_args()
    # parallelize runs
    hparams = [] 
    if len(args['calibration']) == 0:
        args['calibration'].append(None)
    for seed in args['seeds']: 
        for calibration in args['calibration']:
            for backbone in args['backbone']:
                param = OrderedDict(**args)
                param['seeds'] = [seed]
                param['backbone'] = backbone
                param['calibration'] = calibration 
                hparams.append(param)
        
    with ProcessPoolExecutor(max_workers=args['n_workers']) as executor:
        futures = [executor.submit(main, params) for params in reversed(hparams)]
        # for f in futures:
        #     f.add_done_callback(lambda f:pbar.update())
        for future in concurrent.futures.as_completed(futures):
            future.result()
                