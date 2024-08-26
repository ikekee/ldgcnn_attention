"""This module contains a script for training a model on the KITTI-360 dataset."""
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union
from warnings import warn

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim.optimizer import Optimizer
from torch_geometric.loader import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics.functional.classification import multiclass_jaccard_index
from torchmetrics.functional.classification import multiclass_precision
from torchmetrics.functional.classification import multiclass_recall

from components.clear_ml_registrator.clear_ml_registrator import ClearMlRegistrator
from components.common.configuration import KittiConfiguration
from components.common.configuration import TrainingModelConfiguration
from components.common.configuration import TrainingProcessConfiguration
from components.common.constants import KITTI_MODELS_FOLDER
from components.common.inits_lib import init_lr_scheduler
from components.common.inits_lib import init_model
from components.common.inits_lib import init_optimizer
from components.common.kitti_labels_spt import IGNORE_CLASS
from components.common.kitti_labels_spt import SPT_KITTI_NUM_CLASSES
from components.common.kitti_labels_weights_spt import spt_classes_weights
from components.common.path_lib import create_path_if_not_exists
from components.datasets.KITTI360Dataset import KITTI360Dataset
from components.evaluation_lib.evaluate_on_kitti import evaluate
from components.ldgat_v1_model.ldgat_v1 import LDGATv1
from components.ldgat_v2_model.ldgat_v2 import LDGATv2
from components.ldgcnn_model.ldgcnn_model import LDGCNNSegmentor


def _train(loader: DataLoader,
           model: Union[LDGCNNSegmentor, LDGATv1, LDGATv2],
           device: torch.device,
           optimizer: Optimizer,
           acc_gradients_iter: Optional[int] = None,
           verbose_step: int = 5) -> Dict[str, Any]:
    model.train()
    forward_times = []
    losses = []
    accuracies = []
    ious = []

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(loader):
        data = data.to(device)
        start_time = time.time()
        out = model(data)
        forward_time = time.time() - start_time
        forward_times.append(forward_time)
        loss = F.nll_loss(out, data.y, weight=torch.Tensor(spt_classes_weights).to(device))
        if acc_gradients_iter is not None:
            loss = loss / acc_gradients_iter
            loss.backward()
            if ((i + 1) % acc_gradients_iter == 0) or (i + 1 == len(loader)):
                optimizer.step()
                optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes
        losses.append(loss.item())
        accuracies.append(correct_nodes / total_nodes)
        ious.append(multiclass_jaccard_index(
            out.argmax(dim=-1),
            data.y,
            num_classes=SPT_KITTI_NUM_CLASSES).cpu())
        if (i + 1) % verbose_step == 0:
            print(f'[{i + 1}/{len(loader)}] Loss: {total_loss / verbose_step:.4f} '
                  f'Train Acc: {correct_nodes / total_nodes:.4f} '
                  f'Forward time {forward_time}')
            total_loss = correct_nodes = total_nodes = 0
            if forward_time > 1.0:
                warn(f"Forward propagation time is too high ({forward_time})!")
    return {
        "model": model,
        "train_loss": np.mean(losses),
        "train_accuracy": np.mean(accuracies),
        "train_iou": np.mean(ious),
        "forward_prop_time": np.mean(forward_times)
    }


@torch.no_grad()
def _test(loader: DataLoader,
          model: Union[LDGCNNSegmentor, LDGATv1, LDGATv2],
          device: torch.device) -> Dict[str, Any]:
    model.eval()

    predictions = []
    labels = []
    losses = []
    for i, data in enumerate(loader):
        data = data.to(device)
        start_time = time.time()
        out = model(data)
        loss = F.nll_loss(out, data.y, weight=torch.Tensor(spt_classes_weights).to(device))
        forward_time = time.time() - start_time
        if forward_time > 1.0:
            warn(f"Forward propagation time is too high ({forward_time})!")
        predictions.append(out.argmax(dim=1))
        labels.append(data.y)
        losses.append(loss.item())
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    iou = multiclass_jaccard_index(
        predictions,
        labels,
        num_classes=SPT_KITTI_NUM_CLASSES,
        ignore_index=IGNORE_CLASS
    )
    accuracy = multiclass_accuracy(preds=predictions,
                                   target=labels,
                                   num_classes=SPT_KITTI_NUM_CLASSES,
                                   ignore_index=IGNORE_CLASS,
                                   average="micro")
    precision = multiclass_precision(preds=predictions,
                                     target=labels,
                                     num_classes=SPT_KITTI_NUM_CLASSES,
                                     ignore_index=IGNORE_CLASS)
    recall = multiclass_recall(preds=predictions,
                               target=labels,
                               num_classes=SPT_KITTI_NUM_CLASSES,
                               ignore_index=IGNORE_CLASS)
    loss = np.mean(losses)
    return {
        "test_loss": loss,
        "test_iou": iou,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall
    }


def report_metrics(train_dict: Dict[str, float],
                   test_dict: Dict[str, float],
                   epoch: int,
                   clear_ml_registrator: ClearMlRegistrator):
    """Reports train and test metrics to ClearML.

    Args:
        train_dict: A dictionary with train metrics.
        test_dict: A dictionary with test metrics.
        epoch: A number of a current epoch.
        clear_ml_registrator: An instance of the ClearMlRegistrator.
    """
    clear_ml_registrator.report_single_value(
        name='Mean forward prop time, seconds',
        value=train_dict["forward_prop_time"]
    )

    clear_ml_registrator.report_scalar("Loss last model", "Train", train_dict["train_loss"], epoch)
    clear_ml_registrator.report_scalar("Loss last model", "Test", test_dict["test_loss"], epoch)

    clear_ml_registrator.report_scalar("IoU last model", "Train", train_dict["train_iou"], epoch)
    clear_ml_registrator.report_scalar("IoU last model", "Test", test_dict["test_iou"], epoch)

    clear_ml_registrator.report_scalar(
        "Accuracy last model", "Train", train_dict["train_accuracy"], epoch)
    clear_ml_registrator.report_scalar(
        "Accuracy last model", "Test", test_dict["test_accuracy"], epoch)

    clear_ml_registrator.report_scalar(
        "Last model, test set", "Precision", test_dict["test_precision"], epoch)
    clear_ml_registrator.report_scalar(
        "Last model, test set", "Recall", test_dict["test_recall"], epoch)
    clear_ml_registrator.report_scalar(
        "Last model, test set", "Accuracy", test_dict["test_accuracy"], epoch)
    clear_ml_registrator.report_scalar(
        "Last model, test set", "IoU", test_dict["test_iou"], epoch)


def train_on_kitti(kitti_config: KittiConfiguration,
                   training_process_config: TrainingProcessConfiguration,
                   training_model_config: TrainingModelConfiguration,
                   clear_ml_registrator: ClearMlRegistrator,
                   model_path: Optional[Path] = None):
    """Performs training on KITTI-360 dataset.

    Args:
        kitti_config: Kitti-360 dataset configuration instance.
        training_process_config: Training process configuration instance.
        training_model_config: Training model configuration instance.
        clear_ml_registrator: ClearML registrator instance.
        model_path: A path to a model to load if need to continue training.
    """
    clear_ml_registrator.set_up_current_logger()

    pre_transform = T.Compose([T.NormalizeScale(), T.NormalizeFeatures()])
    train_dataset = KITTI360Dataset(kitti_config, split="train", pre_transform=pre_transform)
    val_dataset = KITTI360Dataset(kitti_config, split="val", pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=training_process_config.train_batch_size,
                              shuffle=True,
                              num_workers=training_process_config.dataloader_num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=training_process_config.test_batch_size,
                            shuffle=False,
                            num_workers=training_process_config.dataloader_num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TODO obtain the number of in channels
    model, model_config = init_model(training_model_config, 6, train_dataset.num_classes)
    model = model.to(device)
    optimizer = init_optimizer(training_process_config, model.parameters())
    lr_scheduler = init_lr_scheduler(training_process_config, optimizer)
    first_epoch = 0
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        first_epoch = checkpoint['epoch'] + 1

    checkpoints_folder = Path(KITTI_MODELS_FOLDER) / str(clear_ml_registrator.task_name)
    create_path_if_not_exists(checkpoints_folder)

    for epoch in range(first_epoch, training_process_config.n_epochs):
        train_dict = _train(
            train_loader,
            model,
            device,
            optimizer,
            training_process_config.acc_gradients_iter
        )

        test_dict = _test(
            val_loader,
            train_dict["model"],
            device
        )

        report_metrics(train_dict, test_dict, epoch, clear_ml_registrator)

        print(f'Epoch: {epoch:02d},'
              f' Train loss: {train_dict["train_loss"]:.4f}')
        model_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': train_dict["train_loss"],
        }
        torch.save(model_dict, checkpoints_folder / f"{epoch}_checkpoint_last_model.pth")
        lr_scheduler.step()
    torch.save(model.state_dict(),
               checkpoints_folder / "last_model.pth")
    train_evaluation_metrics = evaluate(train_loader, model, device)
    val_evaluation_metrics = evaluate(val_loader, model, device)

    clear_ml_registrator.upload_evaluation_artifacts(train_evaluation_metrics, "train")
    clear_ml_registrator.upload_evaluation_artifacts(val_evaluation_metrics, "val")

    clear_ml_registrator.report_matplotlib_figure(
        "Train confusion matrix plot",
        series="123",
        figure=train_evaluation_metrics["confusion_matrix"]
    )

    clear_ml_registrator.report_matplotlib_figure(
        "Val confusion matrix plot",
        series="123",
        figure=val_evaluation_metrics["confusion_matrix"]
    )
