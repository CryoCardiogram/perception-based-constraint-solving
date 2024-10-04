import pathlib
import lightning as pl
import numpy as np
import torch
from typing import Iterable
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping
from methods.pbcs import PBCSweightedCE
from methods.fontstyle import FontStyle, FontStyleMT
from methods.calibration import CalibrationOnly
from methods.nasr import NASRTrainer
from config import BackboneArch, Datasets, ImbalancePolicy, CalibrationType, IS_WHOLE, IS_MULTI
from neuralnet import get_model
from constraint_solver.sudoku_solver import SudokuSolverWildcard, SudokuSolverNoGood, SudokuSolverNeurASP, SudokuSolverProlog, SudokuSolver
from constraint_solver.sudoku_solver import ObjectiveWeightedStyle, ObjectiveHCOP, ObjectivePrintedOnly
from utee.img_processing import GridSplitter
from utee.plot import load_pretrain

from utee.databuilder import build_data

from utee.plot import get_run_weights, get_key
import argparse
from collections import OrderedDict, Counter


import concurrent
from concurrent.futures import ProcessPoolExecutor



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", help="Specify output dir path", default='debug')
    parser.add_argument('--dataset', type=Datasets, default=Datasets.VSUDOKU_APP, help=f'dataset to consider for the run, should be one of {[d.value for d in Datasets]}')
    parser.add_argument('--calibration', type=CalibrationType, default=[None], nargs='*', help=f'calibration method among {[b.value for b in CalibrationType]}. Default to None (uncalibrated)')
    parser.add_argument("-b", "--backbone",type=BackboneArch, default=[], nargs='+', help=f'Specify backbone arch to try. Should be one or more of {[BackboneArch.SVHN_CNN.value, BackboneArch.SVHN_MULTI.value]}')
    parser.add_argument('--imbalance_policy', type=ImbalancePolicy, default=ImbalancePolicy.DOWNSAMPLING, help=f'enable to train without stratified batches, should be one of {[p.value for p in ImbalancePolicy]}, default to Downsampling.')
    parser.add_argument('--max_total_epochs', type=int, default=50, help='number of epochs (default=50)')
    #parser.add_argument('--overlap_pad', type=float, default=0.0, help='Overlapping padding percentage (for cell-lvl architecture only). Default 0.0')
    parser.add_argument('--fc_hidden_layers', nargs='+', default=[512], help='hidden layers of dense classifier, default to [512]')
    parser.add_argument('--obfun', type=int, default=[], nargs='+', help='0: regular, 1: wipe, 2: weighted, 3: wildcard,  4: neurasp-solver, 5: swipl (nasr)' )
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--fontstyle', action='store_true', help='enable font style classification task' )
    parser.add_argument('--max_data_size', type=int, default=1000)
    parser.add_argument('--no_good', action='store_true', default=False, help='higher order knowledge exploitation')
    parser.add_argument('-n', '--seeds', default=[244], nargs='+', type=int, help='sequence of seeds for kfolds. Default [544]')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--wildcard_tr', type=float, default=None, help='threshold probability value for wildcard option. Only considered if obfun=3. Default to None')
    parser.add_argument('--weight_printed', type=float, default=None, help='Weight that printed/blank should have wrt handwritten digit, only considered if obfun==2. Default to 1.0')
    parser.add_argument('--n_errors', type=int, default=0, help='amount of faulty handwritten cells to incorporate in test set')
    parser.add_argument('--pretrained', type=str, default=None, help='path to valid exp subfolder with checkpoints. Should be either overlap, pretrain, or font')
    parser.add_argument('--n_workers', type=int, default=1, help='parallel jobs. Default to 1.')
    parser.add_argument('--no_cell_aug', action='store_true', default=False)
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--wildcard_soft',action='store_true',default=False)
    parser.add_argument('--dynamic',action='store_true',default=False,help='Toggle to enable dynamic thresholding for wildcard option. Default to False.')
    parser.add_argument('--gamma_hw',type=float,default=None)
    # parser.add_argument('--gamma_printed',type=float,default=None)
    parser.add_argument('--topk',type=int,default=None)
    parser.add_argument('--precision',type=float,default=1e-4)
    parser.add_argument('--nasr', action='store_true',default=False,help='toggle to enable NASR solverNN + MaskNN on top of predictions')
    parser.add_argument('--timeout', type=int, default=30, help='solver timeout in seconds. Default to 30.')
    args = OrderedDict(vars(parser.parse_args()))
    return args


def main(args):
    
    N_CLASSES = 19
    PUZZLE_SHAPE = (9, 9)

    print('debug exp font args', args)
    print('setup trainer')
    ## setup model trainer 
    model_builder = FontStyle
    splitter = torch.nn.Identity()

    if args['backbone'] == BackboneArch.SVHN_MULTI: 
        N_CLASSES = 10
        model_builder = FontStyleMT

    elif not args['fontstyle']:
        # if args['calibration'] is not None: 
        #     model_builder = CalibrationOnly
        if args['nasr']:
            model_builder = NASRTrainer
        else:
            model_builder = PBCSweightedCE  
        N_CLASSES = 10

    # Objective function for HCOP variants
    if args['obfun'] == 0:
        obj_fun = ObjectiveHCOP(is_extended=args['fontstyle'], precision=args['precision'])
    elif args['obfun'] == 1:
        # wipe
        obj_fun = ObjectivePrintedOnly(precision=args['precision'], is_extended=args['fontstyle'], weight_printed=args['weight_printed'])
    elif args['obfun'] == 2:
        obj_fun = ObjectivePrintedOnly(precision=args['precision'], is_extended=args['fontstyle'])
    elif args['obfun'] == 3:
        ## equivalent to obfun 0
        obj_fun = ObjectiveWeightedStyle(precision=args['precision'], is_extended=args['fontstyle'])
    elif args['obfun'] == 4:
        ## trigger neurasp reasoning engine
        obj_fun = ObjectiveHCOP()
        pass
    elif args['obfun'] == 5:
        ## swi-prolog 
        obj_fun = ObjectiveHCOP()
    else:
        raise NotImplementedError
    
    ## Solvers
    print('init solvers')
    # specific case of using baseline with corrupt_sudoku.
    max_iter = 100 if args['no_good'] else 1
    cs = SudokuSolver(PUZZLE_SHAPE, np.arange(10), obj_fun, time_limit=args['timeout'], topk=args['topk'], corrupt=args['n_errors'] >0)
    if args['no_good']:
        cs = SudokuSolverNoGood(PUZZLE_SHAPE, np.arange(10), obj_fun, time_limit=args['timeout'], topk=args['topk'], max_iter=max_iter, corrupt=args['n_errors'] >0)
    if args['wildcard_tr'] is not None or args['dynamic']:
        #TODO: combine wildcard with nogood
        cs = SudokuSolverWildcard(PUZZLE_SHAPE, np.arange(10), 
                                obj_fun, 
                                time_limit=args['timeout'], 
                                max_iter=max_iter, 
                                topk=args['topk'],
                                wildcard_tr=args['wildcard_tr'],
                                soft=args['wildcard_soft'],
                                dynamic=args['dynamic'],
                                gamma_hw=args['gamma_hw'],
                                # gamma_print=args['gamma_printed'],
                                corrupt=args['n_errors']>0)
    if args['obfun'] == 4:
        cs = SudokuSolverNeurASP(PUZZLE_SHAPE, np.arange(10))
    elif args['obfun'] == 5:
        cs = SudokuSolverProlog(PUZZLE_SHAPE, np.arange(10))

    for version in args['seeds']:
        conf = OrderedDict(**args)
        conf['puzzle_shape'] = PUZZLE_SHAPE
        conf['seeds'] = version
        #conf['dataset'] = Datasets.VSUDOKU_APP
        conf['imbalance_policy'] = args['imbalance_policy']
        #conf['fontstyle']
        conf['solver'] = str(cs)
        conf['num_pred_class'] = N_CLASSES
        conf['sudoku_augment'] = False#args['n_errors'] > 0
        #conf['corrupt_valid'] = True

        net, kwargs_dataset = get_model(
            conf['backbone'], 
            conf['dataset'],
            conf['num_pred_class'],
            conf['puzzle_shape'],
            batchnorm=args['batchnorm'],
            fc_hidden_layers=args['fc_hidden_layers']
        )

        print('get data loaders')
        train_dl, valid_dl, test_dl = build_data(conf['dataset'], conf['backbone'], cs, kwargs_dataset, conf, seed_train_val=version)


        c_train = Counter(train_dl.dataset.y.flatten().tolist())
        c_valid = Counter(valid_dl.dataset.y.flatten().tolist())
        c_test = Counter(test_dl.dataset.y.flatten().tolist())
        print('counter train: ', c_train)
        print('counter valid:', c_valid)
        print('counter test:', c_test)
        

        if args['backbone'] not in IS_WHOLE:
            splitter = GridSplitter(PUZZLE_SHAPE, torch.nn.Identity())

        model = model_builder(
            net,
            cs,
            splitter,
            lr=args['lr'],
            num_pred_classes=N_CLASSES,
            puzzle_shape=conf['puzzle_shape'],
            hparams=conf,
        )
        #print(kwargs_dataset)
        
        logdir = str(pathlib.Path(args['out'], 'joint-inf').resolve())
        conf_name = str(pathlib.Path(*[f'{k}@{v}' for k,v in reversed(conf.items()) if not isinstance(v, Iterable) and k not in ['seeds', 'n_workers']]))
        conf_dir = str(pathlib.Path(logdir, conf_name).resolve())
        pathlib.Path(conf_dir).mkdir(parents=True, exist_ok=True)
        
        logger = CSVLogger(
            save_dir=logdir,
            name=conf_dir,
            version=version,
        )



        # debug
        solver_log_path = pathlib.Path(conf_dir, f"version_{version}", 'solver.log')
        if isinstance(cs, SudokuSolverNoGood):
            cs.log = solver_log_path
            print('debug joint inf nogoods stored at ', cs.log)
        es = EarlyStopping('val_cell_accuracy', mode='max', patience=5)

        trainer = pl.Trainer(
            logger = logger,
            max_epochs=args['max_total_epochs'],
            log_every_n_steps=5,
            callbacks=[es],
            accelerator='auto',
            devices='auto',
            inference_mode=False,
            #auto_lr_find=True
        )

        if args['pretrained'] is None:
            if args['lr'] is None:
                pass
                # results = trainer.tune(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
                # print('learning rate suggested', results)
            trainer.validate(model, valid_dl)
            trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    
            net = model.dnn
        elif args['calibration'] is not None:
            print('debug loading calibration?')
            # use calibration wrapper to get proper cnn
            temp_model = CalibrationOnly(
                net, 
                cs, 
                splitter,
                lr=model.lr,
                num_pred_classes=model.num_pred_classes,
                puzzle_shape=model.puzzle_shape,
                hparams=conf,
            )
            net = temp_model.dnn
            
            conf_calib = {k:v for k,v in conf.items() if 'pretrained' not in k}
            conf_calib['pretrained'] = '/home/maxime/pbcs-jair/output/calibration' if 'calib' not in conf['pretrained'] else conf['pretrained']
            
            #conf_calib['backbone'] = 'SVHN_CNN'
            # key_calib = get_key(conf_calib, conf_calib['pretrained'], version)
            # print('key for pretrain calib model?', key_calib)
            # checkpoint_path_calib = get_run_weights(key_calib, conf_calib['pretrained'], conf)
            # print('debug checkpoint file? ', checkpoint_path_calib)
            # layer_names_calib = list(net.state_dict().keys())
            # print('debug layer name calib', layer_names_calib)
            # CKPT_calib = torch.load(checkpoint_path_calib, map_location=torch.device('cpu'))

            # print('(calib) in stored state_dict: ', list(CKPT_calib['state_dict'].keys()))
            # to_load = OrderedDict(**{k[4:]:v for k,v in CKPT_calib['state_dict'].items() if k.startswith('dnn.') and (k[4:] in layer_names_calib) })   
            # #torch.save(to_load, f'{args["backbone"]}_font.pt')
            # print('to be loaded (calib):', to_load)
            # temp_model.dnn.load_state_dict(to_load)
            incompat = load_pretrain(conf_calib, version, net)
            print(f'debug loaded pretrain weights, except for layers {incompat}')
            # model.dnn = temp_model.dnn 
            # net = model.dnn
            model.dnn = temp_model.dnn
            #print(model.dnn)

        else:
            # if args['backbone'] in IS_MULTI: 
            if False:
                print('debug font load pretrain weights') 
                conf_pretrained_digit = {k:v for k,v in conf.items()}
                conf_pretrained_digit['backbone'] = 'SVHN_CNN'
                incompatible_keys = load_pretrain(conf_pretrained_digit, seed, model.dnn)
                print(f'debug pretrained weights loaded successfully for Multihead arch, from single head pretrained, except for {incompatible_keys}')
                model.dnn = net
            else:
                incompat = load_pretrain(conf, version, net)
                print(f"pretrained weights loaded successfully, except for {incompat}")
                model.dnn = net
            
        
        # try: 
        #     layer_names = list(net.state_dict().keys())
        #     to_load = OrderedDict(**{k[4:]:v for k,v in CKPT_state_dict['state_dict'].items() if k.startswith('dnn.') and (k[4:] in layer_names) })   
        #     #torch.save(to_load, f'{args["backbone"]}_font.pt')
        #     model.dnn.load_state_dict(to_load)
        # except: 
        #     load_pretrain(conf, seed, net)
        #load_pretrain(conf, seed, net)
        # print(f'debug all loaded except: {incompat}')

        model.save_hyperparameters(conf)
        print(model)
        #print(model.dnn)
        #trainer.validate(model,valid_dl )
        if args['validation']:
            results = trainer.test(model, valid_dl)
        else:
            results = trainer.test(model, test_dl)
        #print(results)
        return results[0]['test_cell_accuracy_solve']


if __name__ == '__main__': 
    args = parse_args()
    print('debug exp solve  args', args)
    # parallelize runs
    hparams = [] 
    for seed  in args['seeds']: 
        for obfun in args['obfun']:
            for backbone in args['backbone']:
                for calibration in args['calibration']:
                    param = OrderedDict(**args)
                    param['calibration'] = calibration
                    param['backbone'] = backbone
                    param['obfun'] = obfun
                    param['seeds'] = [seed]
                    hparams.append(param)
    print(hparams)
    with ProcessPoolExecutor(max_workers=args['n_workers']) as executor:
        print('launching jobs')
        print(executor)
        try:
            futures = [executor.submit(main, params) for params in reversed(hparams)]
            print(futures)
            # for f in futures:
            #     f.add_done_callback(lambda f:pbar.update())
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except Exception as e:
            print(e)
            raise e
