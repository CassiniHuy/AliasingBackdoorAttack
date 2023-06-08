from argparse import ArgumentParser
from transfer import datasets, models, AccuracyCounter
from utils import tools
from tqdm import tqdm
from torchvision import transforms
from collections import Counter
from torch.utils import data
from funcs import *
import torch

argparser = ArgumentParser()
argparser.add_argument('--model', type=str)
argparser.add_argument('--weight', type=str)
argparser.add_argument('--input-size', dest='input_size', type=int, default=None)
argparser.add_argument('--dataset', type=str, choices=['pets37', 'flowers', 'caltech101', 'caltech256', 'cifar10', 'cifar100', 'cifar100lt'], default='pets37')
argparser.add_argument('--batch', type=int, default=20)
argparser.add_argument('--preprocess', type=str, default=None)
argparser.add_argument('--D0', type=int, default=None)
argparser.add_argument('--n', type=int, default=None)
args = argparser.parse_args()

random_seed = 999
logger = tools.get_logger(__file__)

# * load dataset
testsplit: datasets.DatasetBase = datasets.load_dataset(args.dataset,
                                                split='test')
idx_to_class = {v: k for k, v in testsplit.class_to_idx.items()}
logger.info(f'Load dataset {args.dataset}')

# * load model
model = models.create_model(args.model, num_classes=len(idx_to_class), pretrained=False, input_size=args.input_size)
model.load_state_dict(torch.load(args.weight))
logger.info(f'Load model {args.model} from {args.weight}')

# * load preprocess
# Input filter
if args.preprocess is None:
    lowpass_preprocess = transforms.Lambda(lambda x: x)
else:
    lowpass_filter = get_preprocess(args.preprocess, **args.__dict__)
    logger.info(f'Input filtering with {args.preprocess}, D0={args.D0}, n={args.n}')
    lowpass_preprocess = transforms.Lambda(lambda x: lowpass_filter(x))
# Resize
input_size = model.input_size
resize_preprocess = transforms.Resize(input_size)
# Norm
if model.mean is None or model.std is None:
    norm_preprocess = transforms.Lambda(lambda x: x)
else:
    norm_preprocess = transforms.Normalize(model.mean, model.std)
# ToCuda
to_cuda = transforms.Lambda(lambda x: x.cuda())
transform = transforms.Compose([
    transforms.ToTensor(),
    to_cuda,
    resize_preprocess,
    lowpass_preprocess, 
    norm_preprocess])
logger.info(f'Get model input size={input_size}, mean={model.mean}, std={model.std}')

# * Load dataset loader
testsplit.transform = transform
dataloader = data.DataLoader(testsplit,
                batch_size=args.batch,
                shuffle=False)

# * Evaluation on test split
n_steps = len(dataloader)
progress = tqdm(dataloader)
counter = AccuracyCounter()
model.cuda()
model.eval()
with torch.no_grad():
    for i_step, (images, targets) in enumerate(progress):
        # Inference
        images, targets = images.cuda(), targets.cuda()
        outputs = model(images)
        # Compute Metric
        counter.update(outputs, targets)
        progress.set_description('Test: Step {0}/{1}'.format(
            i_step, n_steps))
test_top1, test_mpc, test_top1pc = counter.count_acc()
logger.info(f'Top-1 accuracy: {test_top1:.4f}, Top-1 accuracy per class: {test_mpc:.4f}')

