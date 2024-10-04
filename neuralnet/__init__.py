import torch
import torchvision.transforms as T
from PIL import Image
from neuralnet.dnn_arch import *
from neuralnet.calibration_layers import *
from neuralnet.generic_layers import *
from neuralnet.backbone import *

from config import BackboneArch, Datasets, CalibrationType, IS_WHOLE

def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    sigma = 4.0
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

SaltAndPepper = T.Compose([
    T.PILToTensor(),
    gauss_noise_tensor,
    T.ToPILImage()
])

# architectures for multi-task learning
_multi_to_single = {
    BackboneArch.SVHN_MULTI:BackboneArch.SVHN_CNN,
    BackboneArch.LENET_MULTI:BackboneArch.LENET,
    BackboneArch.IMAGENET_RN18_MULTI: BackboneArch.IMAGENET_RN18
}

_backbone_full_image_to_arch = {
    BackboneArch.IMAGENET_RN50_SUDOKUNET:BackboneArch.IMAGENET_RN50,
    BackboneArch.IMAGENET_RN18_SUDOKUNET:BackboneArch.IMAGENET_RN18
}

# default whole-image pre-processing
base_tsfm = T.Compose([
    T.Lambda(Image.fromarray),
    T.Resize((252,252)),
    T.ToTensor(),
])


imagenet_norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
svhn_norm = T.Normalize(mean=(0.4376821, 0.4437697, 0.47280442),
                            std=(0.19803012, 0.20101562, 0.19703614))
mnist_norm = T.Normalize((0.1307,), (0.3081,))

#TODO datasets-depndant normalization
sudoku_assistant_raw_norms = T.Normalize(
    mean=[0.5320, 0.5209, 0.5204],
    std=[0.1862917 , 0.19031495, 0.18998064]

)
# normalizing constant per dataset
normalizes = {
    # BackboneArch.IMAGENET_RN50: imagenet_norm,
    # BackboneArch.IMAGENET_RN50_SIMCLR: imagenet_norm,
    # BackboneArch.SVHN_CNN: svhn_norm,
    # BackboneArch.MNIST_LENET: mnist_norm,
    # 'LENET': T.Normalize((0.1307,), (0.3081,)),
    # 'IMAGENET_ALEXNET': imagenet_norm,
    Datasets.VSUDOKU_APP: sudoku_assistant_raw_norms,
    # mnist data already normalized?
    Datasets.VSUDOKU_MNIST: torch.nn.Identity(),#mnist_norm, 
    Datasets.VSUDOKU_NEURASP: T.Normalize([.5],[.5]),
}

# default cell-level pre-processing
cell_tsfms = {
    BackboneArch.MNIST_LENET: T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((28, 28)),

    ]),
    BackboneArch.SVHN_CNN: T.Compose([
        T.Resize((32, 32)),
    ]),
    BackboneArch.IMAGENET_RN50: T.Compose([
        #T.Lambda(lambda x:x),
        # TensorResize(50)
        T.Resize((256, 256)),
        T.CenterCrop((224,224))
    ]),
    BackboneArch.IMAGENET_RN18: T.Compose([
        T.Resize((252, 252)),
        T.CenterCrop((224,224))
    ]),
    BackboneArch.IMAGENET_RN50_SIMCLR: T.Compose([
        #T.Lambda(lambda x:x),
        # TensorResize(50)
        T.Resize((50, 50))
    ]),
    BackboneArch.LENET: T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((28, 28))
    ]),
    BackboneArch.IMAGENET_ALEXNET: T.Compose([
        #T.Lambda(lambda x:x),
        T.Resize((70, 70))
        # TensorResize(70), # torchvision 0.7 compatible
    ]),
    BackboneArch.SUDOKUNET: torch.nn.Identity()
}

img_tsfms = {
    BackboneArch.IMAGENET_RN18_SUDOKUNET:T.Resize((252,252)),
    BackboneArch.IMAGENET_RN50_SUDOKUNET:T.Resize((252,252))
}

# dataset-specific cell-lvl augmentation
cell_data_augments = {
    Datasets.VSUDOKU_APP:T.AutoAugment(T.AutoAugmentPolicy.SVHN),#T.RandomResizedCrop(28, scale=(0.95,1)),
    Datasets.VSUDOKU_MNIST : T.RandomResizedCrop(28, scale=(0.95,1)),
    Datasets.VSUDOKU_NEURASP: T.RandomResizedCrop(28, scale=(0.95,1))#torch.nn.Identity()
} 

img_data_augments = {
    Datasets.VSUDOKU_APP: T.ColorJitter(contrast=0.5, saturation=0.3, brightness=0.3, hue=0.25),
    Datasets.VSUDOKU_MNIST: torch.nn.Identity(),
    Datasets.VSUDOKU_NEURASP: torch.nn.Identity()#SaltAndPepper
}

feat_extractors_builder_outdim = {
    BackboneArch.SVHN_CNN: (svhn_feature_extract_layers, 256, dict()),
    BackboneArch.MNIST_LENET: (mnist_feature_extract_layers, 800, dict()),
    BackboneArch.LENET: (mnist_feature_extract_layers, 800, {'frozen': False, 'reset': True}),
    BackboneArch.IMAGENET_RN50: (resnet_feature_extract_layers, 2048, dict()),
    BackboneArch.IMAGENET_RN50_SIMCLR: (simclr_feature_extract_layers, 2048, dict()),
    BackboneArch.IMAGENET_ALEXNET: (alexnet_feature_extract_layers, 9216, dict()),
    BackboneArch.IMAGENET_RN18:(resnet18_feature_extract_layers, 512, dict())
}

def get_backbone_builder_outdim(backbone:BackboneArch, frozen=True):
    fe_layers_builder, out_dim, params = feat_extractors_builder_outdim[backbone]
    if frozen is not None:
        params.update({'frozen': frozen})
    return fe_layers_builder, out_dim, params

def get_model(
        backbone:BackboneArch,
        dataset:Datasets,
        N_CLASSES,
        PUZZLE_SHAPE,
        normalize=True,
        fc_hidden_layers=[512],
        batchnorm=False,
        frozen=None,
        dropout=0.3, 
        use_cell_augments=True,
        use_img_augments=True,
        use_cell_aug_at_img=False):
    """Neural Network Builder

    Args:
        name (BackboneArch): backbone architecture Enum (see `config.py`)
        dataset (Datasets): Dataset Enum (see `config.py`)
        N_CLASSES (int): number of class
        PUZZLE_SHAPE (tuple): shape of the grid puzzle
        normalize (bool, optional): toggle to (de)activate input normalization step. Defaults to True.
        fc_hidden_layers (list, optional): hidden layer configuration list. See `make_fc_layer` for more. Defaults to [512].
        batchnorm (bool, optional): toggle to enable/disable batchnorm layer. Defaults to True.
        frozen (bool, optional): if True, disables training of backbone. Default to True.
        dropout (float, optional): dropout hparams for classifier head. Default to 0.3.
        use_cell_augment (bool, optional): toggle off to disable cell level augmentations. Default to True.
        use_img_augment (bool, optional): toggle off to disable img level augmentations. Default to True.
        use_cell_aug_at_img (bool, optional): toggle to apply cell-lvl data augmentation at full image. Default to False

    Raises:
        NotImplementedError:

    Returns:
        tuple: pytorch DNN, whole-img preprocessing, cell-img preprocessing, whole-img data augments, cell-img data augment
    """
    print('(get_model) Find case...')
    if backbone in _multi_to_single.keys():
        # Multi Headed Cell-lvl CNN
        backbone_builder, dim, kwargs = get_backbone_builder_outdim(_multi_to_single[backbone], frozen=frozen)
        # backbone
        fe = torch.nn.Sequential(*backbone_builder(**kwargs))
        # digit classifier head
        fc_layers = make_fc_layers(
            fc_hidden_layers + [N_CLASSES], in_dim=dim, bn=batchnorm, p=dropout)
        # add trainable head for font classification
        fontstylenet = torch.nn.Sequential(
            *make_fc_layers(
                fc_hidden_layers + [1],
                in_dim=dim,
                out=nn.Sigmoid,
                p=dropout,
                bn=batchnorm))
        fc = torch.nn.Sequential(*fc_layers)
        net = CellStyleNet(feat_ext=fe, classifier=fc, style_clf=fontstylenet)
        #print(net)
        base_img_tf = T.Compose([
            *base_tsfm.transforms,
        ])
        base_cell_tf = cell_tsfms[_multi_to_single[backbone]]

    elif backbone == BackboneArch.SUDOKUNET or backbone == BackboneArch.SUDOKUNET_NASP:
        # 
        net = WholeImageCNN(grid_shape=PUZZLE_SHAPE, n_classes=N_CLASSES)
        if backbone == BackboneArch.SUDOKUNET_NASP:
            whole_image_backbone_config = [
                (32, 4, 2, 0),
                (64, 3, 2, 0), 
                (128, 3, 2, 0), 
                (256, 2, 1, 0), 
                (512, 2, 1, 0), 
            ]
            conv_layers = make_conv_layers(
                whole_image_backbone_config, in_channels=1, p=0.25, pool_ks=3, pool_str=3, batch_norm=True)
            out_layers = make_fc_layers_global_pooling(
                in_dim=512, out_shape=(9,9), num_classes=N_CLASSES, pooling=torch.nn.MaxPool2d, kernel_size=3)
            net = FullImageCNN(conv_layers, out_layers)
        base_img_tf = T.Compose([
            T.Lambda(Image.fromarray),
            T.Resize((252,252)),
            T.Grayscale(),
            T.ToTensor(),
        ])
        if normalize:
            base_img_tf.transforms.append(
                T.Compose([
                    normalizes[Datasets.VSUDOKU_NEURASP],
                    #T.Grayscale(3) if dataset == Datasets.VSUDOKU_MNIST else torch.nn.Identity()
                ])
            )
        base_cell_tf = cell_tsfms[BackboneArch.SUDOKUNET]

    elif backbone in _backbone_full_image_to_arch.keys():
        backbone_builder, dim, kwargs = get_backbone_builder_outdim(_backbone_full_image_to_arch[backbone], frozen=frozen)
        feature_ext_layers = torch.nn.Sequential(*backbone_builder(**kwargs))
        # digit classifier head
        # output_layer = make_fc_layers_global_pooling(in_dim=dim, out_shape=PUZZLE_SHAPE, num_classes=N_CLASSES)
        output_layer = make_fc_layers(fc_hidden_layers+[81*N_CLASSES], in_dim=dim, out=nn.Identity, bn=batchnorm, p=dropout)
        net = FullImageCNN(feature_ext_layers, output_layer)
        base_img_tf = base_tsfm
        base_img_tf.transforms.append(
            img_tsfms[backbone]
        )
        base_cell_tf = nn.Identity()
    elif backbone != BackboneArch.SUDOKUNET:
        # Cell-lvl DNN
        print('(get_model) case Cell-lvl DNN...')
        backbone_builder, dim, kwargs = get_backbone_builder_outdim(backbone, frozen=frozen)
        print('(get_model) build backbone...')
        fe = torch.nn.Sequential(*backbone_builder(**kwargs))
        print('(get_model) build output layer')
        fc = torch.nn.Sequential(
            *make_fc_layers(
                fc_hidden_layers +
                [N_CLASSES],
                in_dim=dim,
                bn=batchnorm,
                p=dropout))
        net = CellNet(fe, fc)
        
        if backbone in (BackboneArch.LENET, BackboneArch.MNIST_LENET):
            base_img_tf = T.Compose([
                    *base_tsfm.transforms,
                ])
        else:
            if dataset == Datasets.VSUDOKU_MNIST:
                # convert grayscale mnist img to colorspace 
                base_img_tf = T.Compose([
                T.Grayscale(num_output_channels=3),
                *base_tsfm.transforms,
                ])
            else:
                base_img_tf = T.Compose([
                    *base_tsfm.transforms,
                ])
        if normalize:
            if backbone in (BackboneArch.MNIST_LENET, BackboneArch.LENET):
                # data is normalized at cell-lvl (after grayscale)
                # let's use MNIST normalization in this speficic setting
                #base_img_tf.transforms.append(normalizes[dataset])
                base_img_tf.transforms.append(normalizes[Datasets.VSUDOKU_NEURASP])
            else:
                base_img_tf.transforms.append(normalizes[dataset])
        base_cell_tf = cell_tsfms[backbone]
    else:
        raise NotImplementedError()

    kwargs_dataset = {
        'cell_transform': base_cell_tf,
        'cell_augment': cell_data_augments[dataset] if use_cell_augments else nn.Identity(),
        'img_transform': base_img_tf,
        'img_augment': img_data_augments[dataset] if use_img_augments else nn.Identity(),
    }
    # TODO refactor 
    if use_cell_augments and backbone in [BackboneArch.LENET, BackboneArch.MNIST_LENET, BackboneArch.LENET_MULTI]:
        kwargs_dataset['cell_augment'] = T.RandomResizedCrop(28, scale=(0.95,1))
    if use_cell_aug_at_img:
        kwargs_dataset['img_augment'] = T.Compose([
            T.ToPILImage(),
            cell_data_augments[dataset] if dataset == Datasets.VSUDOKU_APP else T.RandomResizedCrop(252, scale=(0.95,1)),
            T.ToTensor()
        ])
    if use_cell_augments and backbone in IS_WHOLE:
        # apply cell augment at cell level then merge the image back...
        # FIXME now possible because training methods assume grid-level images. 
        # need to change cli arguments semantic
        pass
    if backbone not in IS_WHOLE:
        print('(get_model) add SharedPatcNet wrapper...')
        net = SharedPatchNet(net)
    return net, kwargs_dataset

def get_shared_calibration_wrapper(
        calibration: CalibrationType,
        shared_net: SharedPatchNet,
        output_dim: int
):
    calnet = get_calibration_wrapped_model(calibration, shared_net.cell_net, output_dim)
    shared_net_calibration = SharedPatchNetCal(calnet)
    return shared_net_calibration

def get_calibration_wrapped_model(
        calibration: CalibrationType,
        net: torch.nn.Module,
        output_dim: int):
    """ Builder for calibration-ready DNNs

    Args:
        calibration_name (str): calibration method (see `calibration_layers.py`)
        net (torch.nn.Module): DNN
        output_dim (int):

    Raises:
        NotImplementedError:

    Returns:
        torch.nn.Module: calibration-ready DNN
    """
    #style_net = nn.Identity()
    kwargs = {
        'feat_ext':net.feat_extract,
        'classifier':net.classifier,
        'calibration_layer':torch.nn.Identity(),
        'style_clf':torch.nn.Identity(),
        'style_calibration':torch.nn.Identity(),
        'dnn':net,
    }
    calibrated_cnn_builder = PlattCellNet

    if hasattr(net, 'style_clf'):
        kwargs['style_clf'] = net.style_clf
        if calibration is not None and calibration is not CalibrationType.DIRICHLET:
            kwargs['style_calibration'] = Temperature()
    if calibration == CalibrationType.TEMPERATURE:
        #calinet = PlattCellNet(net.feat_extract, net.classifier, Temperature(), style_clf=style_net)
        kwargs['calibration_layer'] = Temperature()
    elif calibration == CalibrationType.MATRIX:
        # calinet = PlattCellNet(
        #     net.feat_extract,
        #     net.classifier,
        #     Platt(output_dim), 
        #     style_clf=style_net)
        kwargs['calibration_layer'] = Platt(output_dim)
    elif calibration == CalibrationType.VECTOR:
        # calinet = PlattCellNet(
        #     net.feat_extract,
        #     net.classifier,
        #     PlattDiag(output_dim), 
        #     style_clf=style_net)
        kwargs['calibration_layer'] = PlattDiag(output_dim)
    elif calibration == CalibrationType.DIRICHLET:# or calibration_name == 'dfl':
        if not hasattr(net, 'style_clf'):
            #calinet = DirichletWrapper(net, Platt(output_dim), style_clf=style_net)
            kwargs['calibration_layer'] = Platt(output_dim)
            calibrated_cnn_builder = DirichletWrapper
    elif calibration == CalibrationType.NONE:
        kwargs['calibration_layer'] = torch.nn.Identity()
    # else:
    #     calinet = PlattCellNet(net.feat_extract, net.classifier, torch.nn.Identity(), style_clf=style_net)
    #     # raise NotImplementedError()
    calinet = calibrated_cnn_builder(**kwargs)

    return calinet


