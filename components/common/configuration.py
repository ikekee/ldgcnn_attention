"""This module includes classes to define configurations."""
import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union


class ShapenetCoreConfiguration:
    """Encapsulates configuration parameters for the ShapenetCore dataset."""

    def __init__(self, json_data):
        """Creates an instance of the class.

        Args:
            json_data: A dictionary containing configuration parameters.
        """
        self.category_to_use = json_data["category_to_use"]
        self.dataset_path = json_data["dataset_path"]


class KittiTrainDatasetConfiguration:
    """Encapsulates configuration parameters for the Kitti-360 dataset for training."""

    def __init__(self, json_data):
        """Creates an instance of the class.

        Args:
            json_data: A dictionary containing configuration parameters.
        """
        self.cell_split_type = json_data["cell_split_type"]
        self.grid_size = json_data["grid_size"]
        self.stride = json_data["stride"]
        self.cloud_size_threshold = json_data["cloud_size_threshold"]
        self.out_cloud_size = json_data["out_cloud_size"]


class KittiValDatasetConfiguration:
    """Encapsulates configuration parameters for the Kitti-360 dataset for validation."""

    def __init__(self, json_data):
        """Creates an instance of the class.

        Args:
            json_data: A dictionary containing configuration parameters.
        """
        self.cell_split_type = json_data["cell_split_type"]
        self.grid_size = json_data["grid_size"]
        self.stride = json_data["stride"]
        self.cloud_size_threshold = json_data["cloud_size_threshold"]
        self.out_cloud_size = json_data["out_cloud_size"]


class KittiConfiguration:
    """Encapsulates configuration parameters for the Kitti-360 dataset."""

    def __init__(self, json_data):
        """Creates an instance of the class.

        Args:
            json_data: A dictionary containing configuration parameters.
        """
        self.dataset_path = json_data["dataset_path"]
        self.processed_dir = json_data["processed_dir"]
        self.to_use_classes_spt = json_data["to_use_classes_spt"]
        self.train_split = KittiTrainDatasetConfiguration(json_data["train_split"])
        self.val_split = KittiValDatasetConfiguration(json_data["val_split"])


class DatasetConfiguration:
    """Encapsulates configuration parameters for the using dataset."""

    def __init__(self, json_data):
        """Creates an instance of the class.

        Args:
            json_data: A dictionary containing configuration parameters.
        """
        self.dataset_to_use = json_data["dataset_to_use"]
        self.shapenet_core = ShapenetCoreConfiguration(json_data["shapenet_core"])
        self.kitti = KittiConfiguration(json_data["kitti"])


class TrainingModelConfiguration:
    """Encapsulates configuration parameters for the model to train."""

    def __init__(self, json_data):
        """Creates an instance of the class.

        Args:
            json_data: A dictionary containing configuration parameters.
        """
        self.model_to_use = json_data["model_to_use"]
        self.ldgcnn_model_params = json_data["ldgcnn"]
        self.ldgat_v1_model_params = json_data["ldgat_v1"]
        self.ldgat_v2_model_params = json_data["ldgat_v2"]


class TrainingProcessConfiguration:
    """Encapsulates configuration parameters for the training process."""

    def __init__(self, json_data):
        """Creates an instance of the class.

        Args:
            json_data: A dictionary containing configuration parameters.
        """
        self.n_epochs = json_data["n_epochs"]
        self.train_batch_size = json_data["train_batch_size"]
        self.test_batch_size = json_data["test_batch_size"]
        self.dataloader_num_workers = json_data["dataloader_num_workers"]
        self.n_iter_no_change = json_data["n_iter_no_change"]

        self.optimizer = json_data["optimizer"]
        self.adam_config = json_data["adam"]

        self.lr_scheduler = json_data["lr_scheduler"]
        self.step = json_data["step"]

        self.acc_gradients_iter = json_data["acc_gradients_iter"]


class TrainingConfiguration:
    """Encapsulates root configuration parameters."""

    def __init__(self,
                 config_file_name: Optional[Union[Path, str]] = None,
                 json_data: Optional[Dict[str, Any]] = None):
        """Creates an instance of the class.

        Args:
            config_file_name: A path to the configuration file.
            json_data: Parsed json data.
        """
        if config_file_name is not None:
            with open(config_file_name) as config:
                json_data = json.load(config)
        elif config_file_name is None and json_data is None:
            raise ValueError("You must provide either path to config or config dictionary")
        self.random_seed = json_data["random_seed"]
        self.dataset_config = DatasetConfiguration(json_data["dataset_config"])
        self.train_model_config = TrainingModelConfiguration(json_data["train_model_config"])
        self.training_process_config = TrainingProcessConfiguration(
            json_data["training_process_config"])
