from argparse import ArgumentParser
from os import path
from transfer.models import create_model
from torchmetrics.functional import peak_signal_noise_ratio
from torchvision.utils import save_image
from torchvision import transforms
from attack import insert
from utils import tools
import torch

argparser = ArgumentParser()
argparser.add_argument('--model', type=str, default='vit_small_patch16_384')
argparser.add_argument('--beta1', type=float, default=2)
argparser.add_argument('--beta2', type=float, default=.05)
argparser.add_argument('--iters', type=int, default=3000)
argparser.add_argument('--source', type=str, default=r'samples/dog.jpg')
argparser.add_argument('--target', type=str, default=r'samples/cat.jpg')
argparser.add_argument('--input-size', dest='input_size', type=int, default=None)
argparser.add_argument('--logpath', type=str, default=r'backdoors')
args = argparser.parse_args()

logger = tools.get_logger(__file__)

# Load model, images
model = create_model(args.model, pretrained=True, input_size=args.input_size)
logger.info(f'Load {args.model} with input size {model.input_size}, mean {model.mean}, std {model.std}')

# Backdoor Insertion
x_src, x_tgt, x_adv, model_backdoored, loss = insert(
    model, 
    beta1=args.beta1, beta2=args.beta2, 
    x_src=args.source, x_tgt=args.target,
    iters=args.iters, logger=logger
    )
psnr = peak_signal_noise_ratio(x_adv, x_src).item()
logger.info(f'The PSNR of sample generated: {psnr:.4f}')

# Load predictions
model = model.cpu()
if model.mean is None or model.std is None:
    transform = lambda x: x
else:
    transform = transforms.Normalize(model.mean, model.std)
src_pred = tools.predict(model, transform(x_src))
tgt_pred = tools.predict(model, transform(x_tgt))
adv_pred = tools.predict(model, transform(x_adv))
logger.info(f'source pred: {src_pred}.')
logger.info(f'target pred: {tgt_pred}.')
logger.info(f'adv pred: {adv_pred}.')
logger.info(f'Aliasing intensity={loss["DI"]:.2f}.')
logger.info(f'Weight L-2 norm={loss["wgtl2"]:.2f}.')

# Log to file
logpath = path.join(args.logpath, tools.timestr())
logger.info(f'Model is saved in {logpath}')
save_image(x_adv, tools.makedir(path.join(logpath, f'psnr={psnr:.4f}.png')))
torch.save(model.state_dict(), path.join(logpath, f'{args.model}.pth'))
tools.write_json(
    path.join(logpath, 'config.json'), 
    data=dict(
        loss=loss, 
        args=args.__dict__, 
        src_pred=src_pred, 
        tgt_pred=tgt_pred, 
        adv_pred=adv_pred))
