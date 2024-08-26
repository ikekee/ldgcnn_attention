"""This module contains a script to prepare KITTI-360 dataset."""
import argparse
import sys
from pathlib import Path

import torch_geometric.transforms as T


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from components.common.configuration import TrainingConfiguration
from components.datasets.KITTI360Dataset import KITTI360Dataset


def main(config: TrainingConfiguration):
    dataset_config = config.dataset_config.kitti
    pre_transform = T.Compose([T.NormalizeScale(), T.NormalizeFeatures()])
    KITTI360Dataset(kitti_dataset_config=dataset_config,
                    split="train",
                    pre_transform=pre_transform)
    KITTI360Dataset(kitti_dataset_config=dataset_config,
                    split="val",
                    pre_transform=pre_transform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        required=True,
                        type=Path,
                        help="Path to config file")

    args = parser.parse_args()
    main(args.config)
