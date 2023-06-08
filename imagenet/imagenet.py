import matplotlib.pyplot as plt
import torch
import os, datetime
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, utils, datasets


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root: str, scale_size: tuple, crop_size: tuple, norm: tuple):
        self.split = 'val'  # only for validation
        self.root = root
        self.scale_size = scale_size
        self.crop_size = crop_size
        self.norm = norm
        self._load_data()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        x = transforms.Compose([
            transforms.Resize(
                self.scale_size,
                interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(self.norm[0], self.norm[1])
        ])(Image.open(os.path.join(self.root,
                                   self.imgs[index])).convert('RGB'))
        y = torch.tensor(self.labels[index])
        return x, y

    def _load_data(self):
        self.imgs = sorted(os.listdir(self.root))
        with open(os.path.join(os.path.dirname(__file__), 'val.txt')) as f:
            self.labels = list(
                map(lambda line: int(line.strip().split(' ')[-1]), f))


def evaluate_on_imagenet_validation(model: torch.nn.Module,
                                    scale_size: tuple,
                                    crop_size: tuple,
                                    norm: tuple,
                                    imgs_dir: str,
                                    batch: int = 10) -> tuple:
    """evaluate model on imagenet

    Args:
        model (torch.nn.Module): the model
        scale_size (tuple): size after resizing
        crop_size (tuple): size after center crop
        imgs_dir (str): images directory
        norm (str): mean and std
        batch (int, optional): batch size. Defaults to 10.

    Returns:
        (float, float): top-1 and top-5 accuracy
    """
    imagenet_data = ImageNet(root=imgs_dir,
                             scale_size=scale_size,
                             crop_size=crop_size,
                             norm=norm)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=batch,
                                              shuffle=False)
    model.eval()
    top1, top5 = 0, 0
    with torch.no_grad():
        imgs_tqdm = tqdm(enumerate(data_loader))
        for i, (images, real_labels) in imgs_tqdm:
            outputs = model(images.cuda())
            _, pred_labels = torch.topk(outputs.cpu(),
                                        k=5,
                                        dim=-1,
                                        largest=True,
                                        sorted=True)
            top1 += torch.sum(torch.eq(pred_labels[:, 0], real_labels)).item()
            top5 += torch.sum(torch.eq(pred_labels,
                                       real_labels.unsqueeze(-1))).item()
            imgs_tqdm.set_description('Top1/5 acc: {0:.2f}%, {1:.2f}%'.format(
                100 * top1 / (i + 1) / batch, 100 * top5 / (i + 1) / batch))
    return top1 / len(imagenet_data), top5 / len(imagenet_data)
