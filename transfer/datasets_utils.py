import random, torch
import itertools
import transfer.datasets_configs as configs
from PIL import Image
from typing import Dict, List, Tuple, Callable
from torch.utils import data

_random_seed = 999

IMG_PATH = str
IMG_LABEL = int
IMGS = List[Tuple[IMG_PATH, IMG_LABEL]]

# * Utils =============================================================================

def group_by_class(imgs: IMGS) -> List[Tuple[IMG_LABEL, IMGS]]:
    '''Group the imgs by class.
    '''
    imgs_sorted = sorted(imgs, key=lambda kv: kv[1])
    label_imgs = itertools.groupby(imgs_sorted, key=lambda xy: xy[1])
    groups = [(label, list(imgs_i)) for label, imgs_i in label_imgs]
    return groups


def _split_set(imgs: IMGS, split: str, test_ratio: float = 0.2) -> IMGS:
    '''Split the imgs into train and test sets. Keep the class balance.
    '''
    assert split in ['val', 'train', 'test']
    random.seed(_random_seed)
    split_imgs = []
    for imgs_i in map(lambda xy: xy[1], group_by_class(imgs)):
        random.shuffle(imgs_i)
        if split == 'train':
            split_imgs.append(imgs_i[int(len(imgs_i) * test_ratio):])
        else:
            split_imgs.append(imgs_i[:int(len(imgs_i) * test_ratio)])
    return list(itertools.chain(*split_imgs))


def _subset_classes(imgs: IMGS, class_to_idx: Dict[str, int], n_class: int = -1
) -> Tuple[IMGS, Dict[str, int]]:
    '''Subset the dataset class-wise. Decrease the dataset classes to n_class.
    '''
    if n_class < 0:
        return imgs, class_to_idx
    groups = group_by_class(imgs)
    assert n_class <= len(groups)
    random.seed(_random_seed)
    groups_chosen = random.sample(groups, k=n_class)

    idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))
    imgs_new_groups = []
    class_to_idx_new = dict()
    for new_id, label_and_imgs in enumerate(groups_chosen):
        label, imgs_i = label_and_imgs
        imgs_new_groups.append(list(map(lambda xy: (xy[0], new_id), imgs_i)))
        class_to_idx_new[idx_to_class[label]] = new_id
    imgs_new = list(itertools.chain(*imgs_new_groups))
    return imgs_new, class_to_idx_new


def _to_longtail(imgs: IMGS, imb_factor: int = 100) -> IMGS:
    '''Convert dataset to a longtail dataset.
    '''
    groups = group_by_class(imgs)
    max_num = max([len(tpl[1]) for tpl in groups])
    groups_imb = list()
    for cls_idx, imgs in groups:
        num = int(max_num / (imb_factor**(cls_idx / (len(groups) - 1.0))))
        imgs_imbalanced = random.sample(imgs, k=num)
        groups_imb.append(imgs_imbalanced)
    imgs_new = list(itertools.chain(*groups_imb))
    return imgs_new


def load_img(path: IMG_PATH) -> Image.Image:
    return Image.open(path).convert('RGB')


# * Base class =====================================================================

class DatasetBase(data.Dataset):
    name = None

    def __init__(
        self, 
        split: str, 
        transform: Callable[[Image.Image], torch.Tensor], 
        imgs: IMGS, 
        class_to_idx: Dict[str, int],
        cfg: configs.DatasetConfig):
        super().__init__()
        self.split = split
        self.transform = transform
        self.imgs = imgs
        self.class_to_idx = class_to_idx
        self.cfg = cfg

    def subset(self, n_class: int = -1):
        self.imgs, self.class_to_idx = _subset_classes(self.imgs,
                                                       self.class_to_idx,
                                                       n_class)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = load_img(self.imgs[index][0])
        img_tensor = self.transform(img)
        return img_tensor, torch.tensor(self.imgs[index][1])

    def __len__(self):
        return len(self.imgs)


# * Transfer datasets =============================================================

class TransferDataset():
    def __init__(self, name: str, load_dataset_f, **kwargs):
        self.name = name
        self.train_dataset = load_dataset_f(name, 'train', **kwargs)
        self.test_dataset = load_dataset_f(name, 'test', **kwargs)
        try:
            self.val_dataset = load_dataset_f(name, 'val', **kwargs)
        except AssertionError:
            self.val_dataset = None
        self.init()

    def init(self):
        assert self.train_dataset.class_to_idx == self.test_dataset.class_to_idx
        self.class_to_idx = self.train_dataset.class_to_idx
        self.n_class = len(self.class_to_idx)
        self.n_train_images = len(self.train_dataset)
        self.n_test_images = len(self.test_dataset)
        if self.val_dataset:
            assert self.train_dataset.class_to_idx == self.val_dataset.class_to_idx
            self.n_val_images = len(self.val_dataset)
        else:
            self.n_val_images = 0

    def subset(self, n_class: int = -1):
        if n_class < 0:
            return
        self.train_dataset.subset(n_class)
        self.test_dataset.subset(n_class)
        if self.val_dataset:
            self.val_dataset.subset(n_class)
        self.init()

    def get_loaders(self, batch: int, split: str = None, **kwargs):
        assert split in ['train', 'val', 'test', None]
        if split is None or split == 'train':
            train_loader = data.DataLoader(self.train_dataset,
                                        batch_size=batch,
                                        shuffle=True,
                                        pin_memory=True,
                                        **kwargs)
        if split is None or split == 'val':
            if self.val_dataset:
                val_loader = data.DataLoader(self.val_dataset,
                                            batch_size=batch,
                                            shuffle=False,
                                            pin_memory=True,
                                            **kwargs)
            else:
                val_loader = None
        if split is None or split == 'test':
            test_loader = data.DataLoader(self.test_dataset,
                                        batch_size=batch,
                                        shuffle=False,
                                        **kwargs)
        if split is None:
            return train_loader, val_loader, test_loader
        elif split == 'train':
            return train_loader
        elif split == 'val':
            return val_loader
        elif split == 'test':
            return test_loader
        else:
            raise ValueError(f'Unexpected split name "{split}"')

    def set_transform(self, transform):
        self.train_dataset.transform = transform
        if self.val_dataset:
            self.val_dataset.transform = transform
        self.test_dataset.transform = transform

    def __str__(self):
        return 'TransferDataset(name={0},n_class={1},n_train_imgs={2},n_val_imgs={3},n_test_imgs={4})'.format(
            self.name, self.n_class, self.n_train_images, self.n_val_images,
            self.n_test_images)
