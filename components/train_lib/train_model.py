"""This module contains a script for launching the training of the specified model on a specified dataset."""
import argparse
import sys
from pathlib import Path
from typing import List
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from components.clear_ml_registrator.clear_ml_registrator import ClearMlRegistrator
from components.common.configuration import TrainingConfiguration
from components.common.path_lib import create_path_if_not_exists
from components.train_lib.train_on_kitti import train_on_kitti
from components.train_lib.train_on_shapenet_core import train_on_shapenet_core


def main(config_path: str, task_name: str, tags: Optional[List[str]] = None):

    create_path_if_not_exists(Path("/models"))

    clear_ml_registrator = ClearMlRegistrator(task_name=task_name, tags=tags, config_path=config_path)

    config = TrainingConfiguration(config_path)

    if config.dataset_config.dataset_to_use == "shapenet_core":
        create_path_if_not_exists(Path(f"models/shapenet_core/{task_name}"))
        train_on_shapenet_core(shapenet_core_config=config.dataset_config.shapenet_core,
                               training_process_config=config.training_process_config,
                               training_model_config=config.train_model_config,
                               clear_ml_registrator=clear_ml_registrator)
    elif config.dataset_config.dataset_to_use == "kitti":
        create_path_if_not_exists(Path(f"models/kitti/{task_name}"))
        train_on_kitti(kitti_config=config.dataset_config.kitti,
                       training_process_config=config.training_process_config,
                       training_model_config=config.train_model_config,
                       clear_ml_registrator=clear_ml_registrator)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_config.dataset_to_use}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c",
                        "--config",
                        required=True,
                        help="Path to config file")
    parser.add_argument("-task",
                        "--task_name",
                        required=True,
                        help="Task name for clear ml")
    parser.add_argument("-tags ",
                        "--tags",
                        action="extend",
                        nargs="+",
                        type=str,
                        required=False,
                        help="Tags for clear ml experiment")

    args = parser.parse_args()
    main(args.config,
         args.task_name,
         args.tags)
