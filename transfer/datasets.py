import os, inspect, sys
import scipy.io
import numpy as np
import transfer.datasets_configs as configs
from PIL import Image
from torch import Tensor
from torchvision import datasets
from tqdm import tqdm
from typing import Callable
from .datasets_utils import (
    _split_set,
    _to_longtail,
    DatasetBase,
    TransferDataset,
)
'''
flowers: flowers classification, 102 classes
aicraft: aircraft classification, 101 classes
birds: birds classification, 500 classes
pets2: cat and dog classification, 2 classes
pets37: cat&dog breed classification, 37 classes
dtd: texture classification, 47 classes
'''


# root, path like "yourpath/102flowers/jpg" : https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
# labels_mat: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
# splits_mat: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
class DatasetFlowers102(DatasetBase):
    name = 'flowers'

    def __init__(self, split: str, transform: Callable[[Tensor], Tensor], cfg: configs.FlowersConfig):
        assert split in ['train', 'test', 'val']
        split_to_path = dict(
            zip(['train', 'val', 'test'], ['trnid', 'valid', 'tstid']))
        img_indices = scipy.io.loadmat(cfg.splits_mat)[split_to_path[split]][0]
        img_labels_all = scipy.io.loadmat(cfg.labels_mat)['labels'][0]
        imgs = [(os.path.join(cfg.root,
                              f'image_{x:05}.jpg'), img_labels_all[x - 1] - 1)
                for x in img_indices]
        classes = [str(label) for label in np.unique(img_labels_all)]
        class_to_idx = dict(zip(classes, map(lambda x: int(x) - 1, classes)))
        super().__init__(split, transform, imgs, class_to_idx, cfg)


# root, "yourpath/fgvc-aircraft-2013b": https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
class DatasetAircraft100(DatasetBase):
    name = 'aircraft'

    def __init__(self, split: str, transform: Callable[[Tensor], Tensor], cfg: configs.AircraftConfig):
        assert split in ['train', 'test', 'val']
        split_to_path = dict(
            zip(['train', 'test', 'val'], [
                'data/images_variant_train.txt',
                'data/images_variant_test.txt', 'data/images_variant_val.txt'
            ]))
        with open(os.path.join(cfg.root, split_to_path[split])) as f:
            images_and_labels = [[line[:8].strip(), line[8:].strip()]
                                 for line in f]
            classes = list(set([x[1] for x in images_and_labels]))
            class_to_idx = dict(zip(classes, range(len(classes))))
            imgs = [(os.path.join(cfg.root, 'data/images',
                                  x[0] + '.jpg'), class_to_idx[x[1]])
                    for x in images_and_labels]
            super().__init__(split, transform, imgs, class_to_idx, cfg)


# root, "yourpath/dtd": https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
class DatasetDTD(DatasetBase):
    name = 'dtd'

    def __init__(self, split: str, transform: Callable[[Tensor], Tensor], cfg: configs.DTDConfig):
        assert split in ['train', 'val', 'test']
        split_to_path = dict(
            zip(['train', 'test', 'val'],
                ['labels/train1.txt', 'labels/test1.txt', 'labels/val1.txt']))
        with open(os.path.join(cfg.root, 'labels/classes.txt')) as f:
            classes = [line.strip() for line in f]
        class_to_idx = dict(zip(classes, range(len(classes))))
        with open(os.path.join(cfg.root, split_to_path[split])) as f:
            imgs = [(os.path.join(cfg.root, 'images', l.strip()),
                     class_to_idx[l.strip().split('/')[0]]) for l in f]
        super().__init__(split, transform, imgs, class_to_idx, cfg)


# root, "yourpath/images": https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# root_anno, "yourpath/annotations": https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
def dataset_pets(split: str, transform: Callable[[Tensor], Tensor], anno: str, cfg: configs.PetsConfig):
    assert split in ['train', 'val', 'test']
    annotations = ['class_id', 'species']
    assert anno in annotations
    split_to_path = dict(
        zip(['train', 'val', 'test'],
            ['trainval.txt', 'trainval.txt', 'test.txt']))
    label_idx = annotations.index(anno) + 1
    with open(os.path.join(cfg.root_anno, split_to_path[split])) as f:
        lines = [line.strip().split(' ') for line in f]
        imgs = [(os.path.join(cfg.root,
                              parts[0] + '.jpg'), int(parts[label_idx]) - 1)
                for parts in lines]
        labels_nd_ids = [('_'.join(parts[0].split('_')[:-1]),
                          int(parts[label_idx]) - 1) for parts in lines]
        class_to_idx = dict(set(labels_nd_ids))
    if split != 'test':
        imgs = _split_set(imgs, split, test_ratio=0.2)
    return (split, transform, imgs, class_to_idx, cfg)


class DatasetPets37(DatasetBase):
    name = 'pets37'

    def __init__(self, split: str, transform: Callable[[Tensor], Tensor], cfg: configs.PetsConfig):
        super().__init__(
            *dataset_pets(split, transform, 'class_id', cfg))


class DatasetPets2(DatasetBase):
    name = 'pets2'

    def __init__(self, split: str, transform: Callable[[Tensor], Tensor], cfg: configs.PetsConfig):
        super().__init__(
            *dataset_pets(split, transform, 'species', cfg))


# Thanks to https://robustnessws4285631339.blob.core.windows.net/public-datasets/birdsnap.tar? \
# sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&  \
# st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
# root: "yourpath/birdsnap"
class DatasetBirds(DatasetBase):
    name = 'birds'

    def __init__(self, split: str, transform: Callable[[Tensor], Tensor], cfg: configs.BirdConfig):
        assert split in ['train', 'test', 'val']
        if split == 'test':
            test_set = datasets.ImageFolder(os.path.join(cfg.root, 'test'))
            super().__init__(split, transform, test_set.imgs, test_set.classes,
                             test_set.class_to_idx, cfg)
        else:
            trainval_set = datasets.ImageFolder(os.path.join(cfg.root, 'train'))
            imgs = _split_set(trainval_set.imgs, split, test_ratio=0.2)
            super().__init__(split, transform, imgs, trainval_set.class_to_idx, cfg)


# root_train, "yourpath/cars_train": http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
# root_test, "yourpath/cars_test": http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
# devkit, "yourpath/devkit": https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
# testlabel, "yourpath/cars_test_annos_withlabels.mat": http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat
class DatasetCars(DatasetBase):
    name = 'cars'

    def __init__(self,
                 split: str,
                 transform: Callable[[Tensor], Tensor],
                 cfg: configs.CarsConfig):
        assert split in ['train', 'val', 'test']
        meta_path = os.path.join(cfg.devkit, 'cars_meta.mat')
        if split == 'test':
            if cfg.testlabel is None or cfg.root_test is None:
                raise ValueError('"testlabel" or "root_test" not set.')
            mat_path = cfg.testlabel
            img_root = cfg.root_test
        else:
            if cfg.root_train is None:
                raise ValueError('"root_train" not set.')
            img_root = cfg.root_train
            mat_path = os.path.join(cfg.devkit, 'cars_train_annos.mat')
        mat_dict = scipy.io.loadmat(mat_path)
        imgs = [(os.path.join(img_root,
                              items[-1].item()), items[-2].item() - 1)
                for items in mat_dict['annotations'][0]]
        cars_meta = scipy.io.loadmat(meta_path)
        classes = cars_meta['class_names'][0]
        class_to_idx = dict([(classes[i].item(), i)
                             for i in range(len(classes))])
        if split != 'test':
            imgs = _split_set(imgs, split, test_ratio=0.2)
        super().__init__(split, transform, imgs, class_to_idx, cfg)


# root, "yourpath/caltech101": http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
class DatasetCaltech101(DatasetBase):
    name = 'caltech101'

    def __init__(self, split: str, transform: Callable[[Tensor], Tensor], cfg: configs.Caltech101Config):
        assert split in ['train', 'test', 'val']
        dataset = datasets.ImageFolder(
            os.path.join(cfg.root, '101_ObjectCategories'))
        if split == 'test':
            imgs = _split_set(dataset.imgs, split, test_ratio=0.66)
        else:
            train_imgs = _split_set(dataset.imgs, 'train', test_ratio=0.66)
            imgs = _split_set(train_imgs, split, test_ratio=0.2)
        super().__init__(split, transform, imgs, dataset.class_to_idx, cfg)


# root, "yourpath/caltech256": https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar
class DatasetCaltech256(DatasetBase):
    name = 'caltech256'

    def __init__(self, split: str, transform: Callable[[Tensor], Tensor], cfg: configs.Caltech256Config):
        assert split in ['train', 'test', 'val']
        dataset = datasets.ImageFolder(
            os.path.join(cfg.root, '256_ObjectCategories'))
        if split == 'test':
            imgs = _split_set(dataset.imgs, split, test_ratio=0.5)
        else:
            train_imgs = _split_set(dataset.imgs, 'train', test_ratio=0.5)
            imgs = _split_set(train_imgs, split, test_ratio=0.2)
        super().__init__(split, transform, imgs, dataset.class_to_idx, cfg)


# root, "yourpath/cfp-dataset":http://www.cfpw.io/cfp-dataset.zip
class DatasetCFP(DatasetBase):
    name = 'cfp'

    def __init__(self, split: str, transform: Callable[[Tensor], Tensor], cfg: configs.CFPConfig):
        assert split in ['train', 'test', 'val']
        dataset = datasets.ImageFolder(os.path.join(cfg.root, r'Data/Images'))
        if split == 'test':
            imgs = _split_set(dataset.imgs, split, test_ratio=7/14)
        else:
            train_imgs = _split_set(dataset.imgs, 'train', test_ratio=7/14)
            imgs = _split_set(train_imgs, split, test_ratio=3/7)
        name_list = os.path.join(cfg.root, r'Data/list_name.txt')
        with open(name_list) as f:
            class_to_idx = {l.strip():i for i, l in enumerate(f)}
            assert len(class_to_idx) == len(dataset.class_to_idx)
        super().__init__(split, transform, imgs, class_to_idx, cfg)


def download_cifar(root: str, name: str):
    assert name in ['cifar10', 'cifar100']
    print(f'Downloading {name}...')
    # Train split
    dataset_cls = [datasets.CIFAR10, datasets.CIFAR100][['cifar10', 'cifar100'].index(name)]
    train_split = dataset_cls(root, train=True, download=True)
    for cls in train_split.classes:
        os.makedirs(os.path.join(root, f'{name}-imgs', 'train', cls))
    idx_to_class = {v: k for k, v in train_split.class_to_idx.items()}
    for i in tqdm(range(len(train_split.targets))):
        im, label = Image.fromarray(train_split.data[i]), idx_to_class[train_split.targets[i]]
        im.save(os.path.join(root, f'{name}-imgs', 'train', label, f'{label}-{i}.png'))
    # Test split
    test_split = dataset_cls(root, train=False, download=True)
    for cls in test_split.classes:
        os.makedirs(os.path.join(root, f'{name}-imgs', 'test', cls))
    idx_to_class = {v: k for k, v in test_split.class_to_idx.items()}
    for i in tqdm(range(len(test_split.targets))):
        im, label = Image.fromarray(test_split.data[i]), idx_to_class[test_split.targets[i]]
        im.save(os.path.join(root, f'{name}-imgs', 'test', label, f'{label}-{i}.png'))


class CIFAR10(DatasetBase):
    name = 'cifar10'

    def __init__(self, split: str, transform: Callable[[Tensor], Tensor], cfg: configs.CIFAR10Config):
        assert split in ['train', 'test']
        if os.path.exists(os.path.join(cfg.root, 'cifar10-imgs')) is False:
            download_cifar(cfg.root, name='cifar10')
        dataset = datasets.ImageFolder(os.path.join(cfg.root, 'cifar10-imgs', split))
        super().__init__(split, transform, dataset.imgs, dataset.class_to_idx, cfg)


class CIFAR100(DatasetBase):
    name = 'cifar100'

    def __init__(self, split: str, transform: Callable[[Tensor], Tensor], cfg: configs.CIFAR100Config):
        assert split in ['train', 'test']
        if os.path.exists(os.path.join(cfg.root, 'cifar100-imgs')) is False:
            download_cifar(cfg.root, name='cifar100')
        dataset = datasets.ImageFolder(os.path.join(cfg.root, 'cifar100-imgs', split))
        super().__init__(split, transform, dataset.imgs, dataset.class_to_idx, cfg)


class CIFAR100LT(DatasetBase):
    name = 'cifar100lt'

    def __init__(self, split: str, transform: Callable[[Tensor], Tensor], cfg: configs.CIFAR100LTConfig):
        assert split in ['train', 'test']
        if os.path.exists(os.path.join(cfg.root, 'cifar100-imgs')) is False:
            download_cifar(cfg.root, name='cifar100')
        dataset = datasets.ImageFolder(os.path.join(cfg.root, 'cifar100-imgs', split))
        imgs = dataset.imgs
        if split == 'train':
            imgs = _to_longtail(dataset.imgs, imb_factor=cfg.imb_factor)
        super().__init__(split, transform, imgs, dataset.class_to_idx, cfg)


''' Loaders '''
        
def load_dataset(name, split, preprocess=None, cfg: configs.DatasetConfig = None):
    if cfg is None:
        cfg = getattr(configs.AllDatasetsConfigs, name)
    for n, c in inspect.getmembers(sys.modules[__name__],
                                   predicate=inspect.isclass):
        if issubclass(c, DatasetBase) and c.name == name:
            return c(split=split, transform=preprocess, cfg=cfg)
    raise ValueError(f'Dataset not found: {name}')


def load_transfer_dataset(name, preprocess=None, n_class: int = -1):
    sets = TransferDataset(name, load_dataset, preprocess=preprocess)
    sets.subset(n_class)
    return sets

