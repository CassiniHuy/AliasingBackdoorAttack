import torch, timm, os, urllib, torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from typing import Union, Tuple


def freeze_norm_layers(model) -> torch.nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.LayerNorm):
            module.eval()  # keep running mean/std fixed
    return model


def freeze(model, layer_name: str) -> torch.nn.Module:
    """freeze the layers before layer_name (included)

    Args:
        model (torch.nn.Module): the model
        layer_name (str): the layer name (get from Module.named_modules)

    Returns:
        torch.nn.Module: the model with some layers frozen
    """    
    if layer_name is None:
        updatable = True
    else:
        updatable = False
    for name, module in model.named_modules():
        for param in module.parameters():
            param.requires_grad = updatable
        if name == layer_name:
            updatable = True
    return model


def set_train_mode(model: torch.nn.Module, freeze_norm=True):
    model.train()
    if freeze_norm:
        return freeze_norm_layers(model)
    else:
        return model


def set_model_preprocess(model: torch.nn.Module, 
                         preprocess, normalize, resize, 
                         input_size, in_channel, mean, std):
    setattr(model, 'preprocess', preprocess)
    setattr(model, 'normalize', normalize)
    setattr(model, 'resize', resize)
    setattr(model, 'input_size', input_size)
    setattr(model, 'in_channel', in_channel)
    setattr(model, 'mean', mean)
    setattr(model, 'std', std)


######################################## ResNet50/21k ######################################################

# From https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/src_files/models/utils/factory.py
def load_resnet50_weights(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(
                    key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model


def download_resnet50_21k():
    if os.path.exists('.cache') is False:
        os.makedirs('.cache')
    url = 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth'
    f = urllib.request.urlopen(url)
    data = f.read()
    with open('.cache/resnet50_miil_21k.pth', "wb") as code:
        code.write(data)


def resnet50_miil_21k(pretrained: bool = True) -> torch.nn.Module:
    model = timm.create_model('resnet50', pretrained=False)
    if pretrained is False:
        return model
    if os.path.exists('.cache/resnet50_miil_21k.pth') is False: # ! hard code weight path
        download_resnet50_21k()
    load_resnet50_weights(model, '.cache/resnet50_miil_21k.pth')
    return model


def set_miil_preprocess(model, 
                        input_size: Tuple[int] = None,
                        mean: Union[Tuple[float], float] = None,
                        std: Union[Tuple[float], float] = None,):
    in_channel = 3
    input_size = (224, 224) if input_size is None else input_size
    input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
    resize = transforms.Resize(input_size)
    if mean is not None and std is not None:
        normalize = transforms.Normalize(mean, std)
    elif mean is None and std is None: # ! miil model do not use normalization by default.
        normalize = transforms.Lambda(lambda x: x)
    else:
        raise ValueError(f'mean is {mean} while std is {std}. Both should be None type or not.')
    preprocess = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize,
    ])
    set_model_preprocess(model, preprocess, normalize, resize, 
                         input_size, in_channel, mean, std)
    return model


def load_miil_model(num_classes: int = None, 
                    pretrained: Union[bool, str] = True,
                    input_size: Tuple[int] = None,
                    mean: Union[Tuple[float], float] = None,
                    std: Union[Tuple[float], float] = None,):
    # Load model with specified class num
    ori_pretrained = pretrained if isinstance(pretrained, bool) else False
    model = resnet50_miil_21k(pretrained=ori_pretrained)
    # Load pretrained weights
    if isinstance(pretrained, str):
        print(f'==== Load model from {pretrained}')
        weight_dict = torch.load(pretrained)
        del weight_dict['fc.weight']
        del weight_dict['fc.bias']
        model.load_state_dict(weight_dict, strict=False)
    # Load model with specified class num
    if num_classes is not None:
        model.fc = torch.nn.Linear(model.fc.in_features,
                                num_classes,
                                bias=True)
    # Load preprocess args
    model = set_miil_preprocess(model, input_size, mean, std)
    return model

###################################### Timm model ######################################################

def set_preprocess_timm(model,
                        input_size: Union[Tuple[int], int] = None,
                        mean: Union[Tuple[float], float] = None,
                        std: Union[Tuple[float], float] = None,):
    in_channel = model.default_cfg['input_size'][0]
    input_size = model.default_cfg['input_size'][-2:] if input_size is None else input_size
    input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
    mean = model.default_cfg['mean'] if mean is None else mean
    std = model.default_cfg['std'] if std is None else std
    normalize = transforms.Normalize(mean, std)
    resize = transforms.Resize(input_size)
    preprocess = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize
    ])
    set_model_preprocess(model, preprocess, normalize, resize, 
                         input_size, in_channel, mean, std)
    return model

def load_timm_model(model_name: str,
                    num_classes: int = None,
                    pretrained: Union[bool, str] = True,
                    input_size: Union[Tuple[int], int] = None,
                    mean: Union[Tuple[float], float] = None,
                    std: Union[Tuple[float], float] = None,):
    # Load model with specified class num
    ori_pretrained = pretrained if isinstance(pretrained, bool) else False
    if num_classes is not None:
        model: torch.nn.Module = timm.create_model(model_name,
                                pretrained=ori_pretrained,
                                num_classes=num_classes)
    else:
        model: torch.nn.Module = timm.create_model(model_name, pretrained=ori_pretrained)
    # Load pretrained weights (unstrict)
    if isinstance(pretrained, str):
        weight_dict = torch.load(pretrained)
        del weight_dict[model.default_cfg['classifier']+'.weight']
        del weight_dict[model.default_cfg['classifier']+'.bias']
        model.load_state_dict(weight_dict, strict=False)
    # Load preprocess args
    model = set_preprocess_timm(model, input_size, mean, std)
    return model

######################################## FaceNet #######################################################

def load_facenet(num_classes: int = None,
                 pretrained: Union[bool, str] = True,
                 input_size: Union[Tuple[int], int] = None,
                 mean: Union[Tuple[float], float] = None,
                 std: Union[Tuple[float], float] = None,):
    # * Default preprocess args
    facenet_mean = mean if mean else 127.5 / 256
    facenet_std = std if std else 128.0 / 256
    facenet_input_size = input_size if input_size else (160, 160)
    resize = transforms.Resize(facenet_input_size)
    normalize = transforms.Normalize(facenet_mean, facenet_std)

    trans = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize
    ])
    # Load model with specified class num
    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=num_classes
    )
    # Load pretrained weights (unstrict)
    if isinstance(pretrained, str):
        weight_dict = torch.load(pretrained)
        del weight_dict['logits.weight']
        del weight_dict['logits.bias']
        resnet.load_state_dict(weight_dict, strict=False)
    # Load preprocess args
    set_model_preprocess(resnet, trans, normalize, resize, 
                                 facenet_input_size, 3, facenet_mean, facenet_std)
    return resnet


###################################### Load model ######################################################


def create_model(model_name: str,
                 num_classes: int = None,
                 layer_name: str = None,
                 pretrained: Union[bool, str] = True,
                 input_size: Union[Tuple[int], int] = None,
                 mean: Union[Tuple[float, float, float], float] = None,
                 std: Union[Tuple[float, float, float], float] = None,) -> torch.nn.Module:
    # print('Loading pretrained model...')
    # Load models
    if model_name == 'resnet50_miil_21k':
        model = load_miil_model(
                num_classes=num_classes, pretrained=pretrained, 
                input_size=input_size, mean=mean, std=std)
    elif model_name == 'facenet':
        model = load_facenet(num_classes=num_classes, pretrained=pretrained, 
                             input_size=input_size, mean=mean, std=std)
    else:   # From timm library
        model = load_timm_model(
                model_name=model_name, num_classes=num_classes, pretrained=pretrained, 
                input_size=input_size, mean=mean, std=std)
    # Freeze layers
    return freeze(model, layer_name)
