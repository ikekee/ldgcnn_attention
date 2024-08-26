"""This module contains a script for training a model on ShapeNetCore dataset."""
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim.optimizer import Optimizer
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torchmetrics.functional.classification import multiclass_jaccard_index

from components.clear_ml_registrator.clear_ml_registrator import ClearMlRegistrator
from components.common.configuration import ShapenetCoreConfiguration
from components.common.configuration import TrainingModelConfiguration
from components.common.configuration import TrainingProcessConfiguration
from components.common.constants import SHAPENET_MODELS_FOLDER
from components.common.inits_lib import init_model
from components.common.inits_lib import init_optimizer
from components.common.path_lib import create_path_if_not_exists
from components.ldgat_v1_model.ldgat_v1 import LDGATv1
from components.ldgat_v2_model.ldgat_v2 import LDGATv2
from components.ldgcnn_model.ldgcnn_model import LDGCNNSegmentor


def train(loader: DataLoader,
          model: Union[LDGCNNSegmentor, LDGATv1, LDGATv2],
          device: torch.device,
          optimizer: Optimizer) -> Dict[str, Any]:
    model.train()

    forward_times = []
    losses = []
    accuracies = []
    ious = []
    categories = []

    total_loss = correct_nodes = total_nodes = 0
    y_map = torch.empty(loader.dataset.num_classes, device=device).long()
    for i, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        start_time = time.time()
        outs = model(data)
        forward_time = time.time() - start_time
        forward_times.append(forward_time)
        loss = F.nll_loss(outs, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += outs.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes
        losses.append(loss.item())
        accuracies.append(correct_nodes / total_nodes)
        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
                                    data.category.tolist()):
            category = list(ShapeNet.seg_classes.keys())[category]
            part = ShapeNet.seg_classes[category]
            part = torch.tensor(part, device=device)

            y_map[part] = torch.arange(part.size(0), device=device)

            iou = multiclass_jaccard_index(out[:, part].argmax(dim=-1), y_map[y],
                                           num_classes=part.size(0))
            loss = F.nll_loss(out, y)

            losses.append(loss.item())
            ious.append(iou)

        categories.append(data.category)
        if (i + 1) % 10 == 0:
            print(f'[{i + 1}/{len(loader)}] Loss: {total_loss / 10:.4f} '
                  f'Train Acc: {correct_nodes / total_nodes:.4f} '
                  f'Forward time {forward_time}')
            total_loss = correct_nodes = total_nodes = 0
            break

    print(f"Mean forward propagation time is {np.mean(forward_times)}")

    iou = torch.tensor(ious, device=device)
    category = torch.cat(categories, dim=0)

    mean_iou = scatter(iou, category, reduce='mean')  # Per-category IoU.

    return {
        "model": model,
        "train_loss": np.mean(losses),
        "train_accuracy": np.mean(accuracies),
        "train_iou": float(mean_iou.mean()),
        "forward_prop_time": np.mean(forward_times)
    }


@torch.no_grad()
def test(loader: DataLoader,
         model: Union[LDGCNNSegmentor, LDGATv1, LDGATv2],
         device: torch.device) -> Dict[str, Any]:
    model.eval()

    ious, losses, categories, accuracies = [], [], [], []
    correct_nodes = total_nodes = 0

    y_map = torch.empty(loader.dataset.num_classes, device=device).long()
    for i, data in enumerate(loader):
        data = data.to(device)
        outs = model(data)

        correct_nodes += outs.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes
        accuracies.append(correct_nodes / total_nodes)

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
                                    data.category.tolist()):
            category = list(ShapeNet.seg_classes.keys())[category]
            part = ShapeNet.seg_classes[category]
            part = torch.tensor(part, device=device)

            y_map[part] = torch.arange(part.size(0), device=device)

            iou = multiclass_jaccard_index(out[:, part].argmax(dim=-1), y_map[y],
                                           num_classes=part.size(0))
            loss = F.nll_loss(out, y)

            losses.append(loss.item())
            ious.append(iou)

        categories.append(data.category)

    iou = torch.tensor(ious, device=device)
    category = torch.cat(categories, dim=0)

    mean_iou = scatter(iou, category, reduce='mean')  # Per-category IoU.
    return {
        "model": model,
        "test_loss": np.mean(losses),
        "test_accuracy": np.mean(accuracies),
        "test_iou": float(mean_iou.mean()),
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
    clear_ml_registrator.report_single_value(name='Mean forward prop time, seconds', value=train_dict[
        "forward_prop_time"])

    clear_ml_registrator.report_scalar("Loss last model", "Train", train_dict["train_loss"], epoch)
    clear_ml_registrator.report_scalar("Loss last model", "Test", test_dict["test_loss"], epoch)

    clear_ml_registrator.report_scalar("IoU last model", "Train", train_dict["train_iou"], epoch)
    clear_ml_registrator.report_scalar("IoU last model", "Test", test_dict["test_iou"], epoch)

    clear_ml_registrator.report_scalar(
        "Accuracy last model", "Train", train_dict["train_accuracy"], epoch)
    clear_ml_registrator.report_scalar(
        "Accuracy last model", "Test", test_dict["test_accuracy"], epoch)


def train_on_shapenet_core(shapenet_core_config: ShapenetCoreConfiguration,
                           training_process_config: TrainingProcessConfiguration,
                           training_model_config: TrainingModelConfiguration,
                           clear_ml_registrator: ClearMlRegistrator):
    """Performs training on ShapenetCore dataset.

    Args:
        shapenet_core_config: ShapenetCore dataset configuration instance.
        training_process_config: Training process configuration instance.
        training_model_config: Training model configuration instance.
        clear_ml_registrator: ClearML registrator instance.
    """
    clear_ml_registrator.set_up_current_logger()

    category = shapenet_core_config.category_to_use
    path = shapenet_core_config.dataset_path
    transform = T.Compose([
        T.RandomJitter(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
    ])
    pre_transform = T.NormalizeScale()
    train_dataset = ShapeNet(path,
                             category,
                             split='trainval',
                             transform=transform,
                             pre_transform=pre_transform)
    test_dataset = ShapeNet(path,
                            category,
                            split='test',
                            pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=training_process_config.train_batch_size,
                              shuffle=True,
                              num_workers=training_process_config.dataloader_num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=training_process_config.test_batch_size,
                             shuffle=False,
                             num_workers=training_process_config.dataloader_num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TODO obtain the number of in channels
    model, model_config = init_model(training_model_config, 6, train_dataset.num_classes)
    model = model.to(device)

    optimizer = init_optimizer(training_process_config, model.parameters())

    models_folder = Path(SHAPENET_MODELS_FOLDER) / str(clear_ml_registrator.task_name)
    create_path_if_not_exists(models_folder)

    for epoch in range(training_process_config.n_epochs):
        train_dict = train(train_loader, model, device, optimizer)

        test_dict = test(test_loader,
                         train_dict["model"],
                         device)
        report_metrics(train_dict, test_dict, epoch, clear_ml_registrator)
        print(f'Epoch: {epoch:02d},'
              f' Train loss: {train_dict["train_loss"]:.4f},'
              f' Test loss: {test_dict["test_loss"]:.4f}')

        torch.save(model.state_dict(), models_folder / f"{epoch}_checkpoint_last_model.pth")
    torch.save(model.state_dict(), models_folder / "last_model.pth")
    return model
