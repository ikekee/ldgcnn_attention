"""This module contains a script for converting a .pt files to a .txt files."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def main(pt_data_path: Path, output_path: Path):
    if not output_path.exists():
        output_path.mkdir()
    filenames = os.listdir(pt_data_path)
    for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
        data = torch.load(Path(pt_data_path) / filename)
        pos_np_data = data.pos.numpy()
        x_np_data = data.x.numpy()
        y_np_data = data.y.numpy()
        y_np_data = np.expand_dims(y_np_data, axis=0)

        array = np.concatenate((pos_np_data, x_np_data, y_np_data.T), axis=1)
        np.savetxt(output_path / filename.replace(".pt", ".txt"), array, delimiter=' ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt_data_path",
        type=Path,
        help="Path to processed .pt data for conversion.",
        metavar="path/to/file",
        required=True
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path for saving the .txt files.",
        metavar="path/to/file",
        required=True
    )
    args = parser.parse_args()
    main(args.pt_data_path, args.output_path)
