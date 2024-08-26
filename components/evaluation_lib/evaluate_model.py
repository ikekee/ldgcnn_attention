"""This module contains a script to evaluate a trained model."""
import argparse
import sys
from pathlib import Path
from typing import List
from typing import Optional

from components.clear_ml_registrator.clear_ml_registrator import ClearMlRegistrator
from components.common.path_lib import create_path_if_not_exists


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from components.evaluation_lib.evaluate_on_kitti import evaluate_on_kitti


def main(dataset_name: str,
         model_id: str,
         task_name: str,
         tags: Optional[List[str]] = None):
    # TODO: add an option to provide a dataset for evaluation but not use the one model was trained on
    clear_ml_registrator = ClearMlRegistrator(task_name, tags)
    if dataset_name == "kitti":
        create_path_if_not_exists(Path(f"models/kitti/{task_name}"))
        evaluate_on_kitti(clear_ml_registrator=clear_ml_registrator,
                          existing_clearml_model_id=model_id)
    # TODO: add evaluation on shapenet core
    elif dataset_name == "shapenet_core":
        raise NotImplementedError("Evaluation on shapenet core is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-dn",
                        "--dataset_name",
                        required=True,
                        help="Name of the dataset to use.")
    parser.add_argument("-i",
                        "--model_id",
                        required=True,
                        help="Model id from clearml.")
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
    main(args.dataset_name,
         args.model_id,
         args.task_name,
         args.tags)
