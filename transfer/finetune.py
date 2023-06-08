import os, datetime, json
import torch
from tqdm import tqdm
from logging import Logger
from typing import List, Tuple
from transfer import models
from .finetune_config import FinetuneSetting
from .metrics import MetricsRecorder
from .datasets_utils import TransferDataset
from .datasets import load_transfer_dataset


class Finetuner():
    def __init__(
        self,
        logger: Logger, 
        dataset_name: str,
        model_name: str,
        model: torch.nn.Module = None,
        cfg: FinetuneSetting = None,
    ):
        self.logger = logger
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model = model
        if cfg is None:
            self.logger.info(f'Default finetuning setting is used...')
        self.cfg = cfg if cfg else FinetuneSetting()
        self.logger.info(f'Finetuning setting received: {str(self.cfg)}')
        self._init_model_and_dataset()
        self._init_training_args()
    
    def _init_model_and_dataset(self):
        # * Get dataset
        self.dataset: TransferDataset = load_transfer_dataset(
            self.dataset_name, n_class=self.cfg.n_class)
        self.logger.info(f'Finetune on the dataset: {str(self.dataset)}')
        self.logger.info(f'Dataset config: {str(self.dataset.train_dataset.cfg)}')
        # * Get model
        if self.model is None:
            self.model: torch.nn.Module = models.create_model(
                self.model_name,
                num_classes=self.dataset.n_class,
                layer_name=self.cfg.layer_name, 
                pretrained=self.cfg.pretrained,
                input_size=self.cfg.input_size).cuda()
        else:
            self.model = self.model.cuda()
        if self.cfg.restore_from is not None:
            self.model.load_state_dict(torch.load(self.cfg.restore_from))
            self.logger.info(f'Restore from: {self.cfg.restore_from}')
        self.logger.info(f'NOTE: Finetune {self.model_name} of input size '
                         f'{self.model.input_size}, mean {self.model.mean}, std {self.model.std}')
        # * Set dataset transform
        self.dataset.set_transform(self.model.preprocess)

    def _init_training_args(self):
        # * Get finetuning batch size, save path
        if self.cfg.sub_batch is None:
            self.cfg.sub_batch = self.cfg.batch
        self.weight_savepath = os.path.join(
            self.cfg.save_path, self.model_name, self.dataset_name,
            f'{datetime.datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")}')
        self.logger.info(f'Weights savepath: {self.weight_savepath}')
        # * Get optimizable params, optimizer, scheduler
        self.named_params: dict = dict(
            filter(lambda kv: kv[1].requires_grad,
                   self.model.named_parameters()))
        self.logger.info(f'Optimizable params: {str(list(self.named_params.keys()))}')
        self.logger.info(f"NOTE: The normalization layers are {'frozen' if self.cfg.freeze_norm else 'unfrozen'}.")
        self.optimizer = torch.optim.SGD(self.named_params.values(),
                                         self.cfg.lr,
                                         momentum=self.cfg.momentum,
                                         weight_decay=self.cfg.weight_decay,
                                         nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.cfg.gamma)
        # * Get loss function
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        # * Get data loaders
        self.train_loader, self.val_loader, self.test_loader \
            = self.dataset.get_loaders(self.cfg.sub_batch)
        if self.val_loader is None:
            self.logger.warning(f'NOTE: No validation split found and therefore validation is disabled.')
        # * Get metric recorder
        self.recorder = MetricsRecorder(f'{self.model_name}-{self.dataset_name}')

    def _save_model(self, epoch: int = None) -> str:
        if os.path.exists(self.weight_savepath) is False:
            os.makedirs(self.weight_savepath)
        output_path = os.path.join(self.weight_savepath, 
                                   f'{self.model_name}{f"-epoch{epoch}" if epoch else ""}.pth')
        torch.save(self.model.state_dict(), output_path)
        self.logger.info(f'Model saved to {output_path}.')
        if os.path.exists(os.path.join(self.weight_savepath, 'config.json')) is False:
            self._save_args()
        return output_path

    def _load_model(self):
        output_path = os.path.join(self.weight_savepath, f'{self.model_name}.pth')
        assert os.path.exists(output_path) == True
        self.model.load_state_dict(torch.load(output_path))
        self.model.eval()
        self.logger.info(f'Load test model from {output_path}.')

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float, List[float]]:
        """Validate for one epoch
        """
        n_steps = len(self.val_loader)
        progress = tqdm(self.val_loader)
        for i_step, (images, targets) in enumerate(progress):
            # Inference
            images, targets = images.cuda(non_blocking=True), targets.cuda(
                non_blocking=True)
            outputs = self.model(images)
            # Compute Metric
            self.recorder.val_counter.update(outputs, targets)
            progress.set_description('Validation: Step {0}/{1}'.format(
                i_step, n_steps))
        val_top1, val_mpc, val_mpcl = self.recorder.val_counter.reset_epoch()
        return val_top1, val_mpc, val_mpcl

    def _train(self) -> Tuple[float, float, List[float]]:
        """Train for one epoch
        """
        n_steps = len(self.train_loader)
        update_cycle = int(self.cfg.batch / self.cfg.sub_batch)
        progress = tqdm(self.train_loader)
        self.optimizer.zero_grad()
        for i_step, (images, targets) in enumerate(progress):
            # Forward pass
            images, targets = images.cuda(non_blocking=True), targets.cuda(
                non_blocking=True).long()
            outputs = self.model(images)
            loss: torch.Tensor = self.criterion(outputs, targets)
            # Compute accuracy
            self.recorder.train_counter.update(outputs, targets)
            # Backward propagation
            loss.backward()
            if (i_step + 1) % update_cycle == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.recorder.update_loss(loss.item())
            progress.set_description(
                'Train: Step: {0}/{1} loss {2:.4f}'.format(i_step, n_steps, loss))
        if n_steps % update_cycle != 0:
            self.optimizer.step()
        train_top1, train_mpc, train_mpcl = self.recorder.train_counter.reset_epoch()
        return train_top1, train_mpc, train_mpcl

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, List[float]]:
        """Evaluate model on test split
        """
        self._load_model()
        n_steps = len(self.test_loader)
        progress = tqdm(self.test_loader)
        for i_step, (images, targets) in enumerate(progress):
            # Inference
            images, targets = images.cuda(), targets.cuda()
            outputs = self.model(images)
            # Compute Metric
            self.recorder.test_counter.update(outputs, targets)
            progress.set_description('Test: Step {0}/{1}'.format(
                i_step, n_steps))
        test_top1, test_mpc, test_mpc_list = self.recorder.test_counter.reset_epoch()
        return test_top1, test_mpc, test_mpc_list
    
    def _check_and_save_model(self):
        met_savep = None
        # Replare the current or not
        if self.val_loader and self.cfg.save_best_on_val:
            if self.recorder.better_than_before():
                self.logger.info(f'Saving the best model...')
                self._save_model()
                met_savep = self.recorder.save_metrics(self.weight_savepath, True)
        else:
            self._save_model()
            met_savep = self.recorder.save_metrics(self.weight_savepath, True)
        # Save models every n epochs
        if self.cfg.save_every_epoch > 0:
            if (self.recorder.epoch) % self.cfg.save_every_epoch == 0:
                self._save_model(self.recorder.epoch)
                self.evaluate()
                met_savep = self.recorder.save_metrics(self.weight_savepath)
        if met_savep:
            self.logger.info(f'Checkpoint metrics log saved into {met_savep}')
    
    def _train_one_epoch(self):
        models.set_train_mode(self.model, self.cfg.freeze_norm)
        top1_train, mpc_train, _ = self._train()
        self.scheduler.step()
        self.recorder.reset_loss_epoch_end()
        self.logger.info(f'Epoch {self.recorder.epoch}/{self.cfg.epoch}, '
                            f'Train accuracy Top-1/mean-per-class: '
                            f'{top1_train:.4f}/{mpc_train:.4f}')
    
    def _val_one_epoch(self):
        if self.val_loader:
            self.model.eval()
            top1_val, mpc_val, _ = self._validate()
            self.logger.info(
                'Epoch {0}/{1}, val accuracy Top-1/mean-per-class: {2:.4f}/{3:.4f}'
                .format(self.recorder.epoch, self.cfg.epoch, top1_val, mpc_val))
            self.recorder.update_early_stop()
            self.logger.info(f'Best validation result: '
                             f'Top-1 accuracy {self.recorder.val_counter.best_top1:.4f}, '
                             f'Top-1 accuracy per class {self.recorder.val_counter.best_mpc:.4f}')
    
    def _test_one_epoch(self):
        top1_val, mpc_val, _ = self.evaluate()
        met_savep = self.recorder.save_metrics(self.weight_savepath, True)
        self.logger.info(f'Checkpoint metrics log saved into {met_savep}')
        self.logger.info(f'Test result: Top-1 accuracy {top1_val:.4f}'
                            f', Top-1 accuracy per class {mpc_val:.4f}')

    def finetune(self) -> tuple:
        """Finetune for all epochs. For each epoch, train and validate.
        """
        # * Epoch loop: Train and Validate
        for _ in range(self.cfg.epoch):
            # Train one epoch
            self._train_one_epoch()
            # Eval after training one epoch
            self._val_one_epoch()
            # If early stop
            if self.cfg.early_stop > 0 and \
                (self.recorder.early_stop_tolerance == self.cfg.early_stop):
                    break
            # Save the model
            self._check_and_save_model()
            # if self.recorder.train_counter.top1s[-1] > 0.999: # ! DEBUG
            #     break
        # * Test and log
        self._test_one_epoch()

    def _save_args(self):
        # * Define the savename and content
        json_path = os.path.join(self.weight_savepath, 'config.json')
        json_dict = dict(
            model=self.model_name,
            dataset=self.dataset.__str__(),
            cfg=self.cfg.__dict__,
            dataset_cfg=self.dataset.train_dataset.cfg.__dict__,
            input_size=self.model.input_size,
            norm=(self.model.mean, self.model.std),
            params=list(self.named_params.keys()),
            class_to_idx=self.dataset.class_to_idx,
        )
        # * Write to the file
        with open(json_path, 'w') as f:
            f.write(json.dumps(json_dict, indent=4))
        self.logger.info(f'Finetuning args info saved into path: {json_path}')

    def __str__(self):
        info_str = 'Finetuner({0},{1},layer_name={2},size={3},lr={4},weight_decay={5},batch={6},sbatch={7},n_params={8})'.format(
            self.dataset_name, self.model_name, self.cfg.layer_name,
            self.model.input_size, self.cfg.lr,
            self.cfg.weight_decay, self.cfg.batch, self.cfg.sub_batch,
            f'"{list(self.named_params.keys())[0]}->:{len(self.named_params.keys())}/{len(list(self.model.parameters()))}"'
        )
        return info_str
