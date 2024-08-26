"""This module contains a script for continuing training of an existing model."""
import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from components.clear_ml_registrator.clear_ml_registrator import ClearMlRegistrator
from components.common.common_lib import get_last_model_id
from components.common.common_lib import get_model_path
from components.common.configuration import TrainingConfiguration
from components.train_lib.train_on_kitti import train_on_kitti


def main(task_id: str):
    clear_ml_registrator = ClearMlRegistrator(task_id=task_id)

    config_dict = clear_ml_registrator.config_dict
    config_dict = json.loads(json.dumps(config_dict))
    config = TrainingConfiguration(json_data=config_dict)
    last_model_id = get_last_model_id(clear_ml_registrator)
    model_path = get_model_path(last_model_id)

    if config.dataset_config.dataset_to_use == "shapenet_core":
        raise NotImplementedError("Continuing training on the Shapenet core is not implemented yet.")
    elif config.dataset_config.dataset_to_use == "kitti":
        if not Path(f"models/kitti/{clear_ml_registrator.task_name}").exists():
            raise FileNotFoundError("Experiment with this task id was not found locally.")
        train_on_kitti(kitti_config=config.dataset_config.kitti,
                       training_process_config=config.training_process_config,
                       training_model_config=config.train_model_config,
                       clear_ml_registrator=clear_ml_registrator,
                       model_path=model_path)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_config.dataset_to_use}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i",
                        "--task_id",
                        required=True,
                        help="Task id to continue training.")

    args = parser.parse_args()
    main(args.task_id)
