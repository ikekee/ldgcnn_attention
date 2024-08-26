"""This module contains a script for converting a .json file with metrics file to an Excel table."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from components.common.kitti_labels_spt import kitti_spt_labels_to_ru_labels


def main(json_path: Path, output_path: Path):
    jsons = json_path.glob('*.json')
    columns = []
    metrics = {key: [] for key in kitti_spt_labels_to_ru_labels.values()}
    for file_path in jsons:
        with open(file_path) as f:
            metric_data = json.load(f)
        columns.append(file_path.name)
        for class_name, value in metric_data.items():
            metrics[kitti_spt_labels_to_ru_labels[class_name]].append(value)
    if output_path.is_file():
        pd.DataFrame.from_dict(metrics,
                               orient='index',
                               columns=columns).to_excel(output_path)
    else:
        pd.DataFrame.from_dict(metrics,
                               orient='index',
                               columns=columns).to_excel(output_path / "output.xlsx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j",
                        "--json_folder",
                        required=True,
                        type=Path,
                        help="Path to the folder with json files to transform to table.")
    parser.add_argument("-o",
                        "--output_path",
                        required=True,
                        type=Path,
                        help="Path to the folder or specific file for saving an output. If filename"
                             " is not specified, output.xlsx is used.")
    args = parser.parse_args()
    main(args.json_folder, args.output_path)
