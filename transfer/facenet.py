import torch
import json, pathlib
from facenet_pytorch import training
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from transfer.datasets import load_transfer_dataset, TransferDataset
from transfer import models
from typing import List
from utils import tools
from logging import Logger


def finetune_facenet(
      ckpt_path: str,
      logger:  Logger,
      dataset: str = 'cfp', 
      save_dir: str = 'logs/facenet',
      feature_layer: str = 'avgpool_1a', #r'conv2d_1a.conv', # Empty for all-layer fine-tuning
      lr: float = 0.001,
      batch: int = 32,
      epochs: int = 10,
      milestones: List[int] = [5, 10],
      weight_decay: float = 0):
    # * Load dataset
    dataset: TransferDataset = load_transfer_dataset(dataset)
    train_loader, val_loader, _ = dataset.get_loaders(batch)

    # * Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('Running on device: {}'.format(device))
    resnet = models.create_model('facenet', pretrained=ckpt_path, num_classes=dataset.n_class).to(device)
    logger.info(f'Loaded model from {ckpt_path}')
    dataset.set_transform(resnet.preprocess)

    # * Set optimization evironment
    resnet = models.freeze(resnet, feature_layer)
    trainable_params = dict(filter(lambda kv: kv[1].requires_grad, resnet.named_parameters()))
    logger.info(f'Trainable params: {list(trainable_params.keys())[:6]} ...' )
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'acc': training.accuracy
    }
    optimizer = optim.Adam(trainable_params.values(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones)
    logger.info(f'Training and validation dataset size: {len(dataset.train_dataset)}, {len(dataset.val_dataset)}')

    # * Train
    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        # writer=writer
    )
    for epoch in range(epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, epochs))

        resnet.train()
        train_metric = training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
        )

        resnet.eval()
        val_metric = training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
        )

    # * Save
    save_dir = pathlib.Path(save_dir).joinpath(tools.timestr())
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = pathlib.Path(ckpt_path) if isinstance(ckpt_path, str) else pathlib.Path('facenet.pth')
    save_path = save_dir.joinpath(ckpt_path.stem + f'_{feature_layer}_finetuned.pth')
    json_path = save_dir.joinpath(ckpt_path.stem + f'_{feature_layer}_finetuned.json')
    torch.save(resnet.state_dict(), save_path)
    with open(json_path, 'w') as f:
        json.dump({
            'dataset': dataset.name,
            'feature_layer': feature_layer,
            'lr': lr,
            'batch': batch,
            'epochs': epochs,
            'milestones': milestones,
            'weight_decay': weight_decay,
            'train_acc': train_metric[1]['acc'].cpu().numpy().tolist(),
            'val_acc': val_metric[1]['acc'].cpu().numpy().tolist(),
        }, f, indent=4)
    logger.info(f'Model saved to: {save_path}')
