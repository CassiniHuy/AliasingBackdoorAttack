from transfer import finetune, FinetuneSetting, Finetuner
from utils.tools import get_logger
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('--model', type=str)
argparser.add_argument('--dataset', type=str)
argparser.add_argument('--input-size', dest='input_size', type=int, default=None)
argparser.add_argument('--pretrained', type=str, default=None)
argparser.add_argument('--restore-from', dest='restore_from', type=str, default=None)
argparser.add_argument('--layer-name', dest='layer_name', type=str, default=None)
argparser.add_argument('--batch', type=int, default=None)
argparser.add_argument('--sub-batch', dest='sub_batch', type=int, default=None)
argparser.add_argument('--lr', type=float, default=None)
argparser.add_argument('--weight-decay', dest='weight_decay', type=float, default=None)
argparser.add_argument('--n-class', dest='n_class', type=int, default=None)
argparser.add_argument('--epoch', type=int, default=None)
argparser.add_argument('--early-stop', dest='early_stop', type=int, default=None)
argparser.add_argument('--momentum', type=float, default=None)
argparser.add_argument('--gamma', type=float, default=None)
argparser.add_argument('--unfreeze-norm', dest='unfreeze_norm', action='store_true', default=None)
argparser.add_argument('--save-every-epoch', dest='save_every_epoch', type=int, default=None)
argparser.add_argument('--save-path', dest='save_path', type=str, default=None)

args = argparser.parse_args()
logger = get_logger(__file__)


setting = FinetuneSetting()
for k, v in args.__dict__.items():
    if v is not None:
        setting.__dict__[k] = v
        logger.debug(f'Finetune setting {k} set as {v}')


finetuner = Finetuner(
        logger=logger,
        dataset_name=args.dataset,
        model_name=args.model,
        cfg=setting,
    )

finetuner.finetune()
