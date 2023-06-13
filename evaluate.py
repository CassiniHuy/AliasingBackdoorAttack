from argparse import ArgumentParser
from os import path
from transfer import datasets, models
from utils import tools
from torchvision import transforms
from funcs import *
import torch

argparser = ArgumentParser()
argparser.add_argument('--model', type=str)
argparser.add_argument('--weight', type=str)
argparser.add_argument('--folder', type=str,)
argparser.add_argument('--dataset', type=str, choices=['pets37', 'flowers', 'caltech101', 'caltech256', 'cifar10'], default='pets37')
argparser.add_argument('--unformatted', action='store_true', default=False)
argparser.add_argument('--source-acc', dest='source_acc', action='store_true', default=False)
argparser.add_argument('--exts', type=str, nargs='+', default=None)
argparser.add_argument('--input-size', dest='input_size', type=int, default=None)
argparser.add_argument('--preprocess', type=str, default=None)
argparser.add_argument('--D0', type=int, default=None)
argparser.add_argument('--n', type=int, default=None)
argparser.add_argument('--nodebug', action='store_true', default=False)
args = argparser.parse_args()

random_seed = 999
logger = tools.get_logger(__file__)
if args.nodebug is True:
    from logging import INFO, StreamHandler
    for loghander in logger.handlers:
        if isinstance(loghander, StreamHandler):
            loghander.setLevel(INFO)

# * load dataset
data: datasets.DatasetBase = datasets.load_dataset(args.dataset, split='test',)
idx_to_class = {v: k for k, v in data.class_to_idx.items()}
logger.info(f'Load dataset {args.dataset}')

# * load model
model = models.create_model(args.model, num_classes=len(idx_to_class), pretrained=False, input_size=args.input_size)
model.load_state_dict(torch.load(args.weight))
logger.info(f'Load model {args.model} from {args.weight}')
input_size = model.input_size
if model.mean is None or model.std is None:
    norm_preprocess = transforms.Lambda(lambda x: x)
else:
    norm_preprocess = transforms.Normalize(model.mean, model.std)
logger.info(f'Get model input size={input_size}, mean={model.mean}, std={model.std}')

# * load data
if args.exts is None:
    args.exts = ['png', 'jpg', 'jpeg']
imgs = list()
if args.unformatted is False:
    for ext in args.exts:
        imgs += tools.find_file(
            path.join(args.folder, 'imgs'), pattern='*.' + ext, return_all=True)
    logger.info(f'Find {len(imgs)} from {args.folder}/imgs')
else:
    for ext in args.exts:
        imgs += tools.find_all_ext(args.folder, ext)
    logger.info(f'Find {len(imgs)} from {args.folder}')

if args.source_acc:
    logger.info('Evaluate the source accuracy.')

# * predict
def get_source_or_target(img: str, class_name: str):
    img_selected = tools.find_file(path.splitext(img)[0], pattern=f'{class_name}-*.png')
    if len(img_selected) == 0:
        img_selected = tools.find_file(path.splitext(img)[0], pattern=f'{class_name}-*.jpg')
    return img_selected


# * Load input filter
if args.preprocess is None:
    lowpass_preprocess = transforms.Lambda(lambda x: x)
else:
    lowpass_filter = get_preprocess(args.preprocess, **args.__dict__)
    logger.info(f'Input filtering with {args.preprocess}, D0={args.D0}, n={args.n}')
    lowpass_preprocess = transforms.Lambda(lambda x: lowpass_filter(x))

transform = transforms.Compose(
    [
        transforms.Lambda(lambda x: x.cuda()),
        lowpass_preprocess,
        norm_preprocess,
    ]
)

model.cuda()
model.eval()
results = dict()
n_true, n_valid, n_valid_true, n_hit = 0, 0, 0, 0
for img in imgs:
    x_tensor = tools.load_image_tensor(img, size=input_size)
    adv_pred = idx_to_class[tools.predict(model, transform(x_tensor), imagenet=False)]
    # Preprocess
    img_name = path.basename(img)
    x_src_class = img_name.split('_psnr=')[0].split('-to-')[0]
    x_tgt_class = img_name.split('_psnr=')[0].split('-to-')[1]
    # Compute Accuracy
    if_hit = (adv_pred == x_src_class)
    n_hit += int(if_hit)
    if args.unformatted is False:
        if args.source_acc is True:
            # Get prediction
            src_img = get_source_or_target(img, x_src_class)
            x_src_tensor = tools.load_image_tensor(src_img, size=input_size)
            src_pred = idx_to_class[tools.predict(model, transform(x_src_tensor), imagenet=False)]
            # Compute source accuracy
            n_true += (src_pred == adv_pred)
            results[img] = (adv_pred, (x_src_class, x_tgt_class), src_pred)
            logger.debug(f'{img_name}: pred={adv_pred}, source={x_src_class}(pred={src_pred}), target={x_tgt_class}')
        else:
            # Get prediction
            tgt_img = get_source_or_target(img, x_tgt_class)
            x_tgt_tensor = tools.load_image_tensor(tgt_img, size=input_size)
            tgt_pred = idx_to_class[tools.predict(model, transform(x_tgt_tensor), imagenet=False)]
            # Compute EASR
            if_valid = (tgt_pred == x_tgt_class)
            n_valid += int(if_valid)
            # Compute ASR
            results[img] = (adv_pred, x_src_class, x_tgt_class)
            if_success = (adv_pred == x_tgt_class)
            n_true += int(if_success)
            n_valid_true += int(if_valid and if_success)
            logger.debug(f'{img_name}: pred={adv_pred}, source={x_src_class}, target={x_tgt_class}, valid={if_valid}')
    else:
        results[img] = (adv_pred)
        logger.debug(f'{img_name}: pred={adv_pred}')

save_path = path.join(args.folder, tools.timestr() + f"_{'src' if args.source_acc else 'asr'}.json")

adv_acc = n_hit / len(imgs)
logger.info(f'Model accuracy on the attack samples: {adv_acc:.4f} (ERR: {1 - adv_acc:.4f})')
if args.unformatted is False:
    if args.source_acc is True:
        acc = n_true / len(imgs)
        tools.write_json(
            save_path, 
            data=dict(acc=acc, args=args.__dict__, results=results))
        logger.info(f'ACC: {acc:.4f}')
    else:
        asr = n_true / len(imgs)
        easr = None if n_valid == 0 else n_valid_true / n_valid
        tools.write_json(
            save_path, 
            data=dict(asr=asr, easr=easr, args=args.__dict__, results=results))
        logger.info(f'ASR: {asr:.4f}, EASR: {easr:.4f}')
else:
    tools.write_json(
        save_path, 
        data=dict(args=args.__dict__, results=results))
logger.info(f'Results saved to {save_path}')
