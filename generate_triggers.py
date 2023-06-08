from argparse import ArgumentParser
from os import path
from transfer import datasets, models, datasets_utils
from utils import tools
from torchvision import transforms
from attack import TriggerGenerator
from torchvision.utils import save_image
from torchmetrics.functional import peak_signal_noise_ratio
from funcs import *
import random, torch
import numpy as np

argparser = ArgumentParser()
argparser.add_argument('--model', type=str, default='vit_small_patch16_384')
argparser.add_argument('--weight', type=str)
argparser.add_argument('--lamb', type=float, default=2)
argparser.add_argument('--dataset', type=str, choices=['pets37', 'flowers', 'caltech101', 'caltech256', 'cifar10', 'cifar100'], default='pets37')
argparser.add_argument('--num', type=int, default=100)
argparser.add_argument('--iters', type=int, default=200)
argparser.add_argument('--input-size', dest='input_size', type=int, default=None)
argparser.add_argument('--logpath', type=str, default='triggers')
argparser.add_argument('--layer-name', dest='featlayer', type=str, default=None)
argparser.add_argument('--preprocess', type=str, default=None)
argparser.add_argument('--D0', type=int, default=None)
argparser.add_argument('--n', type=int, default=None)
args = argparser.parse_args()

random_seed = 999
logger = tools.get_logger(__file__)

# * load model and dataset
data: datasets.DatasetBase = datasets.load_dataset(args.dataset,
                                                   split='test')
dataspace = data
logger.info(f'Load dataset {args.dataset}')

if args.weight is None:
    model = models.create_model(args.model, pretrained=True, input_size=args.input_size)
    logger.warn(f'Load model with benign pretrained weights.')
else:
    model = models.create_model(args.model, pretrained=False, input_size=args.input_size)
    weights_dict = torch.load(args.weight)
    del weights_dict[model.default_cfg['classifier'] + '.weight'] # remove later full connected layers. useless
    del weights_dict[model.default_cfg['classifier'] + '.bias']
    model.load_state_dict(weights_dict, strict=False)
    logger.info(f'Load model {args.model} from {args.weight}')

# * Load transform
input_size = model.input_size
logger.info(f'Get model input size={input_size}, mean={model.mean}, std={model.std}')

# * Load input filter
if args.preprocess is None:
    lowpass_preprocess = transforms.Lambda(lambda x: x)
else:
    lowpass_filter = get_preprocess(args.preprocess, **args.__dict__)
    logger.info(f'Input filtering with {args.preprocess}, D0={args.D0}, n={args.n}')
    lowpass_preprocess = transforms.Lambda(lambda x: lowpass_filter(x))

transform = transforms.Compose([
    lowpass_preprocess,
    model.normalize,
])

# * generate triggered samples
idx_to_class = {v: k for k, v in data.class_to_idx.items()}
img_grps = datasets_utils.group_by_class(dataspace.imgs)

psnrs = list()
random.seed(random_seed)
logpath = path.join(args.logpath, f'{args.dataset}_{args.model}_{tools.timestr()}')
generator = TriggerGenerator(model, 
                             layer_name=args.featlayer,
                             transform=transform, 
                             lambda_=args.lamb, 
                             iters=args.iters, 
                             logger=logger)
model.eval()
for i in range(args.num):
    print(f'\nNo.{i + 1}')
    # Randomly select two classes and 1 source image
    groups = random.sample(img_grps, k=2)
    source_group, target_group = groups[0], groups[1]
    source_img, _ = random.choice(source_group[1])
    source_class, target_class = idx_to_class[source_group[0]], idx_to_class[target_group[0]]
    logger.debug(f'source class: {source_class}, target class: {target_class}')
    logger.debug(f'source image: {source_img}')
    # Generate trigger
    target_img, x_adv, loss = generator.trigger(source_img, [kv[0] for kv in target_group[1]],)
    logger.debug(f'target image: {target_img}')
    x_src = tools.load_image_tensor(source_img, size=input_size)
    x_tgt = tools.load_image_tensor(target_img, size=input_size)
    psnr = peak_signal_noise_ratio(x_adv, x_src).item()
    logger.debug(f'psnr={psnr:.2f}, srcl={loss["srcl"]:.4f}, tgtl={loss["tgtl"]:.4f}')
    # Save images
    save_path = path.join(logpath, 'imgs', f'{source_class}-to-{target_class}_psnr={psnr:.2f}_{i}')
    save_image(x_adv, tools.makedir(save_path + '.png'))
    save_image(x_src, tools.makedir(path.join(save_path, f'{source_class}-{path.basename(source_img)}')))
    save_image(x_tgt, tools.makedir(path.join(save_path, f'{target_class}-{path.basename(target_img)}')))
    tools.write_json(path.join(save_path, f'srcl={loss["srcl"]:.4f}-tgtl={loss["tgtl"]:.4f}.json'), data=loss)
    psnrs.append(psnr)

generator.unhook()
psnr_mean = sum(psnrs) / len(psnrs)
psnr_std = np.std(psnrs)
logger.info(f'{len(psnrs)} images generated, PSNR mean: {psnr_mean:.2f}, std: {psnr_std:.2f}')
save_name = tools.makedir(path.join(logpath, f'config_psnr-mean={psnr_mean:.2f}-std={psnr_std:.2f}.json'))
tools.write_json(
    save_name, 
    data=dict(args=args.__dict__, psnr_mean=psnr_mean))
logger.info(f'Log saved to {save_name}')


