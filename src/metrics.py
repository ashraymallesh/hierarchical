
from typing import Dict, List

import torch
from torch import Tensor
import numpy as np
import torch.nn.functional as F

class APRF:
    def __init__(self, classes: List[str]):
        self.classes = classes
        self.reset_counts()
        self.epoch_summaries = {}

    def reset_counts(self):
        n = len(self.classes)
        self.train_tp = np.zeros(n)
        self.train_fp = np.zeros(n)
        self.train_tn = np.zeros(n)
        self.train_fn = np.zeros(n)

        self.eval_tp = np.zeros(n)
        self.eval_fp = np.zeros(n)
        self.eval_tn = np.zeros(n)
        self.eval_fn = np.zeros(n)

        self.train_total = 0
        self.eval_total = 0

        self.train_batches = 0
        self.eval_batches = 0
        self.train_loss = 0
        self.eval_loss = 0

    @classmethod
    def from_params(cls, params: Dict):
        return cls(**params)

    def update_train(self, output: Tensor, target: Tensor, criterion):
        output = output.cpu()
        target = target.cpu()
        self.train_loss += criterion(output, target).item()
        self.train_batches += 1

        _, predicted_i = output.max(dim=1)
        predicted = F.one_hot(predicted_i, len(self.classes))
        target = F.one_hot(target, len(self.classes))

        self.train_total += len(target)
        self.train_tp += ((target == 1) & (predicted == 1)).sum(
            0).float().numpy()
        self.train_fp += ((target == 0) & (predicted == 1)).sum(
            0).float().numpy()
        self.train_tn += ((target == 0) & (predicted == 0)).sum(
            0).float().numpy()
        self.train_fn += ((target == 1) & (predicted == 0)).sum(
            0).float().numpy()

    def update_eval(self, output: Tensor, target: Tensor, criterion):
        output = output.cpu()
        target = target.cpu()
        self.eval_loss += criterion(output, target).item()
        self.eval_batches += 1

        _, predicted_i = output.max(dim=1)
        predicted = F.one_hot(predicted_i, len(self.classes))
        target = F.one_hot(target, len(self.classes))

        self.eval_total += len(target)
        self.eval_tp += ((target == 1) & (predicted == 1)).sum(
            0).float().numpy()
        self.eval_fp += ((target == 0) & (predicted == 1)).sum(
            0).float().numpy()
        self.eval_tn += ((target == 0) & (predicted == 0)).sum(
            0).float().numpy()
        self.eval_fn += ((target == 1) & (predicted == 0)).sum(
            0).float().numpy()

    def summary(self) -> Dict:
        train_loss = self.train_loss / self.train_batches
        overall_train_accuracy = self.train_tp.sum() / self.train_total
        train_accuracy = (self.train_tp + self.train_tn) / self.train_total
        train_precision = self.train_tp / (self.train_tp + self.train_fp)
        train_recall = self.train_tp / (self.train_tp + self.train_fn)
        # remove nans
        train_accuracy[train_accuracy != train_accuracy] = 0
        train_precision[train_precision != train_precision] = 0
        train_recall[train_recall != train_recall] = 0
        # f-score
        train_f1 = 2 * (train_recall * train_precision) / (
                    train_recall + train_precision)
        train_f1[train_f1 != train_f1] = 0

        eval_loss = self.eval_loss / self.eval_batches
        overall_eval_accuracy = self.eval_tp.sum() / self.eval_total
        eval_accuracy = (self.eval_tp + self.eval_tn) / self.eval_total
        eval_precision = self.eval_tp / (self.eval_tp + self.eval_fp)
        eval_recall = self.eval_tp / (self.eval_tp + self.eval_fn)
        # remove nans
        eval_accuracy[eval_accuracy != eval_accuracy] = 0
        eval_precision[eval_precision != eval_precision] = 0
        eval_recall[eval_recall != eval_recall] = 0
        # f-score
        eval_f1 = 2 * (eval_recall * eval_precision) / (
                    eval_recall + eval_precision)
        eval_f1[eval_f1 != eval_f1] = 0

        summary = {}
        summary[f"train_a"] = overall_train_accuracy
        summary[f"train_loss"] = train_loss
        summary[f"eval_a"] = overall_eval_accuracy
        summary["eval_loss"] = eval_loss

        for i, c in enumerate(self.classes):
            summary[f"train_{c}_a"] = train_accuracy[i]
            summary[f"train_{c}_p"] = train_precision[i]
            summary[f"train_{c}_r"] = train_recall[i]
            summary[f"train_{c}_f"] = train_f1[i]

            summary[f"eval_{c}_a"] = eval_accuracy[i]
            summary[f"eval_{c}_p"] = eval_precision[i]
            summary[f"eval_{c}_r"] = eval_recall[i]
            summary[f"eval_{c}_f"] = eval_f1[i]

            summary[f"train_{c}_tp"] = self.train_tp[i]
            summary[f"train_{c}_tn"] = self.train_tn[i]
            summary[f"train_{c}_fp"] = self.train_fp[i]
            summary[f"train_{c}_fn"] = self.train_fn[i]

            summary[f"eval_{c}_tp"] = self.eval_tp[i]
            summary[f"eval_{c}_tn"] = self.eval_tn[i]
            summary[f"eval_{c}_fp"] = self.eval_fp[i]
            summary[f"eval_{c}_fn"] = self.eval_fn[i]

        for k,v in summary.items():
            if k not in self.epoch_summaries:
                self.epoch_summaries[k] = [v]
            else:
                self.epoch_summaries[k].append(v)
        self.reset_counts()

        return summary