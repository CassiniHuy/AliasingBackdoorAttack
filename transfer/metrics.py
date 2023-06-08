import torch, os, json
from typing import List, Tuple
from collections import Counter

STATUS = ['train', 'val', 'test']

class AccuracyCounter():
    def __init__(self, ) -> None:
        self.top1_counter = Counter()
        self.total_counter = Counter()
        self.top1s: List[float] = list()
        self.mpcs: List[float] = list()
        self.mpcls: List[List[float]] = list()

    @property
    def best_top1(self) -> float:
        return max(self.top1s) if self.top1s else None
    
    @property
    def best_mpc(self) -> float:
        return max(self.mpcs) if self.mpcs else None
    
    @property
    def last_top1(self) -> float:
        return self.top1s[-1] if self.top1s else None
    
    @property
    def last_mpc(self) -> float:
        return self.mpcs[-1] if self.mpcs else None
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        top1_batch, total_batch = self._count_top1_per_class(
            outputs, targets)
        self.top1_counter.update(top1_batch)
        self.total_counter.update(total_batch)
    
    def count_acc(self) -> Tuple[float, float, List[float]]:
        return self._calc_top1_and_meanpc(self.top1_counter, self.total_counter)
    
    def reset_epoch(self) -> Tuple[float, float, List[float]]:
        top1, mpc, mpcl = self.count_acc()
        self.top1s.append(top1)
        self.mpcs.append(mpc)
        self.mpcls.append(mpcl)
        self.top1_counter.clear()
        self.total_counter.clear()
        return top1, mpc, mpcl
    
    @staticmethod
    def _count_top1_per_class(outputs: torch.Tensor,
                            targets: torch.Tensor) -> Tuple[float, float]:
        pred_labels = torch.argmax(outputs, dim=-1).view(-1)
        true_labels = targets.view(-1)
        true_or_false = torch.eq(pred_labels, true_labels)
        top1 = Counter(torch.masked_select(true_labels, true_or_false).tolist())
        total = Counter(true_labels.tolist())
        return top1, total

    @staticmethod
    def _calc_top1_and_meanpc(top1: Counter, total: Counter
                              ) -> Tuple[float, float, List[float]]:
        top1_acc = sum(top1.values()) / sum(total.values())
        top1_per_class = [
            top1.get(k) / total.get(k) if top1.get(k) is not None else 0
            for k in total.keys()
        ]
        meanpc_acc = sum(top1_per_class) / len(total.keys())
        return top1_acc, meanpc_acc, top1_per_class


class MetricsRecorder():
    def __init__(self, task_name: str) -> None:
        # Task name
        self.task_name = task_name
        # Tolerance
        self.early_stop_tolerance: int = 0
        # Counter for each epoch
        self.train_counter = AccuracyCounter()
        self.val_counter = AccuracyCounter()
        self.test_counter = AccuracyCounter()
        self.losses: List[float] = list()
        self._losses_one_epoch: List[float] = list()

    @property
    def epoch(self) -> int:
        return len(self.train_counter.top1s)
    
    def update_loss(self, loss: float):
        self._losses_one_epoch.append(loss)
    
    def reset_loss_epoch_end(self):
        self.losses.append(sum(self._losses_one_epoch) / len(self._losses_one_epoch))
        self._losses_one_epoch.clear()
        
    def update_early_stop(self):
        if self.better_than_before():
            self.early_stop_tolerance = 0
        else:
            self.early_stop_tolerance += 1

    def better_than_before(self) -> bool:
        if len(self.val_counter.top1s) == 1:
            return True
        elif len(self.val_counter.top1s) == 0:
            raise RuntimeError(f'No validation results found.')
        else:
            return self.val_counter.top1s[-1] > max(self.val_counter.top1s[:-1]) or\
                self.val_counter.mpcs[-1] > max(self.val_counter.mpcs[:-1])
    
    def save_metrics(self, save_dir: str, save_best: bool = False):
        # * Define the savename and content
        if save_best:
            save_name = f'{self.task_name}.json'
        else:
            save_name = '{0}{1}{2}.json'.format(
                f'{self.task_name}-epoch{len(self.train_counter.top1s)}',
                f'-{self.val_counter.best_top1:.4f}' if self.val_counter.top1s else '',
                f'-{self.test_counter.best_top1:.4f}' if self.test_counter.top1s else '',
            )
        log_info = dict(
            test_top1s=self.test_counter.top1s,
            test_mpcs=self.test_counter.mpcs,
            test_mpcls=self.test_counter.mpcls,
            val_top1s=self.val_counter.top1s,
            val_mpcs=self.val_counter.mpcs,
            val_mpcls=self.val_counter.mpcls,
            train_top1s=self.train_counter.top1s,
            train_mpcs=self.train_counter.mpcs,
            train_mpcls=self.train_counter.mpcls,
            losses=self.losses,
        )
        # * Save to path
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name)
        # * Write to the file
        with open(save_path, 'w') as f:
            f.write(json.dumps(log_info, indent=4))
        return save_path
