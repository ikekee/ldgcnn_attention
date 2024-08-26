"""This module contains functions for evaluation on KITTI."""
import json
import sys
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union
from warnings import warn

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch_geometric.transforms as T
from clearml import InputModel
from clearml import Logger
from clearml import Task
from torch_geometric.loader import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics.functional.classification import multiclass_confusion_matrix
from torchmetrics.functional.classification import multiclass_f1_score
from torchmetrics.functional.classification import multiclass_jaccard_index
from torchmetrics.functional.classification import multiclass_precision
from torchmetrics.functional.classification import multiclass_recall


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from components.clear_ml_registrator.clear_ml_registrator import ClearMlRegistrator
from components.common.common_lib import get_model_path
from components.common.configuration import TrainingConfiguration
from components.common.inits_lib import init_model
from components.common.kitti_labels_spt import kitti_spt_train_id_to_labels
from components.datasets.KITTI360Dataset import KITTI360Dataset
from components.ldgat_v1_model.ldgat_v1 import LDGATv1
from components.ldgat_v2_model.ldgat_v2 import LDGATv2
from components.ldgcnn_model.ldgcnn_model import LDGCNNSegmentor


@torch.no_grad()
def evaluate(loader: DataLoader,
             model: Union[LDGCNNSegmentor, LDGATv1, LDGATv2],
             device: torch.device) -> Dict[str, Any]:
    """Performs a testing process on the prepared KITTI-360 dataset with class-wise metrics.

    Args:
        loader: Data loader to use.
        model: Model instance to use.
        device: Device to perform calculations on.

    Returns:
        Dictionary of class-wise IOU, accuracy, precision, recall and a confusion matrix as a
         matplotlib figure.
    """
    model.eval()
    predictions = []
    labels = []
    for i, data in enumerate(loader):
        labels.append(data.y.cpu())
        on_device_data = data.to(device)
        start_time = time.time()
        outs = model(on_device_data)
        forward_time = time.time() - start_time
        predictions.append(outs.argmax(dim=-1).cpu())

        if forward_time > 1.0:
            warn(f"Forward propagation time is too high ({forward_time})!")
        if (i + 1) % 1000 == 0:
            print(f'[{i + 1}/{len(loader)}]')
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    iou = multiclass_jaccard_index(
        predictions,
        labels,
        num_classes=len(kitti_spt_train_id_to_labels),
        average='none',
        ignore_index=0
    )
    accuracy = multiclass_accuracy(preds=predictions,
                                   target=labels,
                                   num_classes=len(kitti_spt_train_id_to_labels),
                                   average="none",
                                   ignore_index=0)
    precision = multiclass_precision(preds=predictions,
                                     target=labels,
                                     num_classes=len(kitti_spt_train_id_to_labels),
                                     average="none",
                                     ignore_index=0)
    recall = multiclass_recall(preds=predictions,
                               target=labels,
                               num_classes=len(kitti_spt_train_id_to_labels),
                               average="none",
                               ignore_index=0)

    f1 = multiclass_f1_score(preds=predictions,
                             target=labels,
                             num_classes=len(kitti_spt_train_id_to_labels),
                             average="none",
                             ignore_index=0)

    confusion_matrix = multiclass_confusion_matrix(preds=predictions,
                                                   target=labels,
                                                   num_classes=len(kitti_spt_train_id_to_labels),
                                                   normalize="all",
                                                   ignore_index=0)

    ious_dict = {kitti_spt_train_id_to_labels[i]: float(value) for i, value in enumerate(iou)}
    accuracies_dict = {kitti_spt_train_id_to_labels[i]: float(value) for i, value in enumerate(accuracy)}
    precisions_dict = {kitti_spt_train_id_to_labels[i]: float(value) for i, value in enumerate(precision)}
    recalls_dict = {kitti_spt_train_id_to_labels[i]: float(value) for i, value in enumerate(recall)}
    f1_dict = {kitti_spt_train_id_to_labels[i]: float(value) for i, value in enumerate(f1)}

    miou = multiclass_jaccard_index(predictions,
                                    labels,
                                    num_classes=len(kitti_spt_train_id_to_labels),
                                    ignore_index=0).item()
    m_accuracy = multiclass_accuracy(preds=predictions,
                                     target=labels,
                                     num_classes=len(kitti_spt_train_id_to_labels),
                                     average="micro",
                                     ignore_index=0).item()
    m_precision = multiclass_precision(preds=predictions,
                                       target=labels,
                                       num_classes=len(kitti_spt_train_id_to_labels),
                                       ignore_index=0).item()
    m_recall = multiclass_recall(preds=predictions,
                                 target=labels,
                                 num_classes=len(kitti_spt_train_id_to_labels),
                                 ignore_index=0).item()

    m_f1 = multiclass_f1_score(preds=predictions,
                               target=labels,
                               num_classes=len(kitti_spt_train_id_to_labels),
                               ignore_index=0).item()
    mean_metrics = {
        "mean_accuracy": m_accuracy,
        "mean_precision": m_precision,
        "mean_recall": m_recall,
        "mean_f1": m_f1,
        "mean_iou": miou,
    }

    df_cm = pd.DataFrame(
        confusion_matrix,
        index=[kitti_spt_train_id_to_labels[i] for i, _ in
               enumerate(kitti_spt_train_id_to_labels.items())],
        columns=[kitti_spt_train_id_to_labels[i] for i, _ in
                 enumerate(kitti_spt_train_id_to_labels.items())]
    )
    confusion_matrix = plt.figure()
    sns.heatmap(df_cm)

    return {
        "iou": ious_dict,
        "accuracy": accuracies_dict,
        "precision": precisions_dict,
        "recall": recalls_dict,
        "f1": f1_dict,
        "confusion_matrix": confusion_matrix,
        "mean_metrics": mean_metrics
    }


def evaluate_on_kitti(clear_ml_registrator: ClearMlRegistrator,
                      existing_clearml_model_id: str = None) -> None:
    """Performs a preparation steps for evaluation on the KITTI-360 dataset and saves the results.

    Args:
        clear_ml_registrator: An instance of ClearML registrator.
        existing_clearml_model_id: An id of an existing model from ClearML to use.
    """
    clear_ml_registrator.set_up_current_logger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_model = InputModel(model_id=existing_clearml_model_id)
    # Get a task for model
    model_task_id = input_model.task
    model_task = Task.get_task(model_task_id)
    # Get experiment config for a model
    experiment_config = model_task.get_configuration_object_as_dict("General")
    experiment_config = json.loads(json.dumps(experiment_config))
    experiment_config = TrainingConfiguration(json_data=experiment_config)

    pre_transform = T.Compose([T.NormalizeScale(), T.NormalizeFeatures()])
    train_dataset = KITTI360Dataset(experiment_config.dataset_config.kitti, split="train", pre_transform=pre_transform)
    val_dataset = KITTI360Dataset(experiment_config.dataset_config.kitti, split="val", pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=experiment_config.training_process_config.train_batch_size,
                              shuffle=True,
                              num_workers=experiment_config.training_process_config.dataloader_num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=experiment_config.training_process_config.test_batch_size,
                            shuffle=False,
                            num_workers=experiment_config.training_process_config.dataloader_num_workers)

    # Get model local path
    model_path = get_model_path(existing_clearml_model_id)
    # Initialize model
    model, _ = init_model(experiment_config.train_model_config,
                          6,
                          val_dataset.num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    train_dataset_metrics = evaluate(train_loader, model, device)
    val_dataset_metrics = evaluate(val_loader, model, device)

    clear_ml_registrator.upload_evaluation_artifacts(train_dataset_metrics, "train")
    clear_ml_registrator.upload_evaluation_artifacts(val_dataset_metrics, "val")

    Logger.current_logger().report_matplotlib_figure(
        "Train confusion matrix plot",
        series="123",
        figure=train_dataset_metrics["confusion_matrix"]
    )

    Logger.current_logger().report_matplotlib_figure(
        "Val confusion matrix plot",
        series="123",
        figure=val_dataset_metrics["confusion_matrix"]
    )
