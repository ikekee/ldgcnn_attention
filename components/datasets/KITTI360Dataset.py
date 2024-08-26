"""This module contains a class for the KITTI 360 Dataset."""
import random
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
import torch
from pyntcloud import PyntCloud
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from components.common.configuration import KittiConfiguration
from components.common.kitti_labels import kitti_global_id_to_train_id
from components.common.kitti_labels import KITTI_NUM_CLASSES
from components.common.kitti_labels_spt import kitti_global_id_to_spt_train_id
from components.common.kitti_labels_spt import SPT_KITTI_NUM_CLASSES
from components.common.path_lib import create_path_if_not_exists, directory_is_not_empty


def split_scene(
        full_cloud: PyntCloud,
        step_size: int,
        cloud_size_threshold: int,
        out_cloud_size: Optional[int] = None
) -> List[pd.DataFrame]:
    """Splits the full scene into smaller parts like a grid.

    Args:
        full_cloud: Input point cloud to split.
        step_size: A size of a single output point cloud.
        cloud_size_threshold: Threshold for a cloud to be considered as a single cell.
         For clouds which are smaller than this threshold,
         they will be accumulated with another cloud.
        out_cloud_size: Maximum size of output point cloud.
         If the split point is larger than this size, it will be downsampled.
         If not provided, it will be added with the unchanged size.

    Returns:
        List with splits of an input point cloud.
    """
    begin_x = int(full_cloud.points.x.min())
    end_x = int(full_cloud.points.x.max())

    begin_y = int(full_cloud.points.y.min())
    end_y = int(full_cloud.points.y.max())

    splits = []
    cloud_to_accumulate = None
    # TODO: replace this with an implemented solution (check 'point cloud tiles')
    for i in range(begin_x, end_x, step_size):
        for j in range(begin_y, end_y, step_size):
            cloud_part = full_cloud.points.loc[(full_cloud.points.x > i) &
                                               (full_cloud.points.x < i + step_size) &
                                               (full_cloud.points.y > j) &
                                               (full_cloud.points.y < j + step_size)]
            if not cloud_part.empty:
                if out_cloud_size is None:
                    if len(cloud_part) > cloud_size_threshold:
                        splits.append(cloud_part)
                else:
                    if len(cloud_part) < out_cloud_size:
                        if cloud_to_accumulate is None:
                            cloud_to_accumulate = cloud_part
                        else:
                            cloud_to_accumulate = pd.concat([cloud_to_accumulate, cloud_part])
                            if len(cloud_to_accumulate) > out_cloud_size:
                                splits.append(cloud_to_accumulate)
                                cloud_to_accumulate = None
                            else:
                                continue
                    else:
                        if cloud_to_accumulate is not None:
                            splits.append(cloud_to_accumulate)
                            cloud_to_accumulate = None
                        splits.append(cloud_part)
        if cloud_to_accumulate is not None:
            if len(cloud_to_accumulate) > cloud_size_threshold:
                splits.append(cloud_to_accumulate)
        cloud_to_accumulate = None
    if out_cloud_size is not None:
        downsampled_splits = []
        for i, split in enumerate(splits):
            if len(split) > out_cloud_size:
                downsampling_indexes = random.sample(list(split.index), out_cloud_size)
                downsampled_splits.append(split.loc[downsampling_indexes, :])
            else:
                if len(split) > cloud_size_threshold:
                    downsampled_splits.append(split)
        return downsampled_splits
    else:
        return splits


def split_scene_with_stride(
        full_cloud: PyntCloud,
        grid_size: int,
        stride: int,
        cloud_size_threshold: int,
        out_cloud_size: Optional[int] = None
) -> List[pd.DataFrame]:
    """Splits the full scene into smaller parts with a stride.

    Args:
        full_cloud: Input point cloud to split.
        grid_size: A size of a single output point cloud.
        stride: A value of a step between splits' cells.
        cloud_size_threshold: Threshold for a cloud to be considered as a single cell.
         For clouds which are smaller than this threshold,
         they will be accumulated with another cloud.
        out_cloud_size: Maximum size of output point cloud.
         If the split point is larger than this size, it will be downsampled.
         If not provided, it will be added with the unchanged size.

    Returns:
        List with splits of an input point cloud.
    """
    begin_x = int(full_cloud.points.x.min())
    end_x = int(full_cloud.points.x.max())

    begin_y = int(full_cloud.points.y.min())
    end_y = int(full_cloud.points.y.max())

    splits = []

    for i in range(begin_x, end_x - grid_size, stride):
        for j in range(begin_y, end_y - grid_size, stride):
            cloud_part = full_cloud.points.loc[(full_cloud.points.x > i) &
                                               (full_cloud.points.x < i + grid_size) &
                                               (full_cloud.points.y > j) &
                                               (full_cloud.points.y < j + grid_size)]
            if not cloud_part.empty:
                if len(cloud_part) > cloud_size_threshold:
                    if out_cloud_size is not None and len(cloud_part) > out_cloud_size:
                        downsampling_indexes = random.sample(list(cloud_part.index), out_cloud_size)
                        splits.append(cloud_part.loc[downsampling_indexes, :])
                    else:
                        splits.append(cloud_part)
    return splits


class KITTI360Dataset(Dataset):
    """PyTorch Dataset class for processing the KITTI 360 dataset.

    Attributes:
        _root: A root folder for all datasets.
        _cell_split_type: Type of split to perform during the processing. Possible values are
         "full_step" and "with_stride". "Full step stands for splitting the scene into
         non-overlapping parts while "with stride" stands for splitting the scene into
         overlapping parts with a given stride.
        _stride: Value which specifies the stride of a split. Only used if split_type is
         'with_stride'.
        _processed_dir: Name of a directory where processed data will be stored.
        _trainval_raw_folder: Constant path of a KITTI-360
        _test_raw_folder: Folder contains the test data.
        _trainval_data_dir: Full path to the folder containing the trainval data.
        _test_data_dir: Full path to the folder containing the test data.
        _split: Name of a split to perform for the processing.
        _grid_size: The size of a cell in a split grid.
        _cloud_size_threshold: Threshold for a cloud in a grid cell to be counted as valid.
        _out_cloud_size: The size of an output cloud. If the cloud is smaller or equal than this
         size, it stays with the same size. Otherwise, it is downsampled to this size.
    """

    def __init__(self,
                 kitti_dataset_config: KittiConfiguration,
                 split: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """Creates an instance of the class.

        Args:
            kitti_dataset_config: An Instance of Kitti dataset configuration.
            transform: Torch transforms to be applied on processed data.
            pre_transform: Torch transforms to be applied on raw data during the preprocessing.
            pre_filter: A function that takes in an `torch_geometric.data.Data` object and
             returns a boolean value, indicating whether the data object should be included in the
             final dataset.
            split: Name of a split to perform for the processing.
        """
        self._root = Path(kitti_dataset_config.dataset_path)
        self._processed_dir = self._root / kitti_dataset_config.processed_dir
        self._to_use_classes_spt = kitti_dataset_config.to_use_classes_spt
        if split == "train":
            split_config = kitti_dataset_config.train_split
        elif split == "val":
            split_config = kitti_dataset_config.val_split
        else:
            raise ValueError(f"Please provide valid split name. Split '{split}' is not supported")

        if split_config.cell_split_type == "with_stride" and split_config.stride is None:
            raise ValueError("Stride must be specified.")

        self._cell_split_type = split_config.cell_split_type
        self._stride = split_config.stride
        self._trainval_raw_folder = Path("raw/data_3d_semantics/train")
        self._test_raw_folder = Path("raw/data_3d_semantics_test/data_3d_semantics/test")
        self._trainval_data_dir = self._root / self._trainval_raw_folder
        self._test_data_dir = self._root / self._test_raw_folder
        self._split = split
        self._grid_size = split_config.grid_size
        self._cloud_size_threshold = split_config.cloud_size_threshold
        self._out_cloud_size = split_config.out_cloud_size

        super().__init__(kitti_dataset_config.dataset_path, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[Path]:
        """Get a list of raw file names."""
        filenames = []
        raw_root = self._root / "raw"
        if self._split in ['train', 'val']:
            file_path = self._trainval_data_dir / f"2013_05_28_drive_{self._split}.txt"
            with open(file_path, 'r') as f:
                filenames = [raw_root / name.strip() for name in f]
        elif self._split == 'test':
            folders = self._test_data_dir.glob("*")
            for folder_name in folders:
                files_folder = folder_name / "static"
                folder_files = list(files_folder.glob("*"))
                filenames = folder_files
        return filenames

    @property
    def processed_file_names(self) -> List[Path]:
        """Get a list of processed file names."""
        return list((self._processed_dir / self._split).glob("*.pt"))

    @property
    def num_classes(self) -> int:
        """Get a number of classes in a dataset."""
        if self._to_use_classes_spt:
            return SPT_KITTI_NUM_CLASSES
        else:
            return KITTI_NUM_CLASSES

    def len(self):
        """Returns a number of samples in a dataset."""
        return len(self.processed_file_names)

    def get(self, idx):
        """Load a sample of dataset with provided index.

        Args:
            idx: Index of a data sample to load.

        Returns:
            Loaded data sample with provided index.
        """
        data = torch.load(self._processed_dir / self._split / f'data_{idx}.pt')
        return data

    def process_filename(self, filename: Path) -> List[Data]:
        """Processes a single raw KITTI-360 scene.

        According to provided split_type splits the scene.
         Then it packs it into a list of Data objects and saves as a .pt files.

        Args:
            filename: A path to a single raw KITTI-360 scene file.

        Returns:
            A list of Data objects.
        """
        data_list = []
        cloud = PyntCloud.from_file(str(filename))
        if self._cell_split_type == "full_step":
            split_scenes = split_scene(cloud,
                                       self._grid_size,
                                       self._cloud_size_threshold,
                                       self._out_cloud_size)
        elif self._cell_split_type == "with_stride":
            split_scenes = split_scene_with_stride(cloud,
                                                   self._grid_size,
                                                   self._stride,
                                                   self._cloud_size_threshold,
                                                   self._out_cloud_size)
        else:
            raise ValueError("Please provide valid split_type."
                             " Supported split_types: full_step, with_stride")
        for split_cloud in split_scenes:
            x = torch.Tensor(split_cloud[["red", "green", "blue"]].values)
            pos = torch.Tensor(split_cloud[["x", "y", "z"]].values)
            if self._to_use_classes_spt:
                train_semantic = split_cloud[
                    "semantic"].apply(lambda label: kitti_global_id_to_spt_train_id[label]).values
            else:
                train_semantic = split_cloud[
                    "semantic"].apply(lambda label: kitti_global_id_to_train_id[label]).values
            y = torch.Tensor(train_semantic).type(torch.long)
            data = Data(x=x, y=y, pos=pos)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        return data_list

    def process(self):
        """Performs overall processing of the dataset."""
        if directory_is_not_empty(self._processed_dir / self._split):
            return
        create_path_if_not_exists(self._processed_dir / self._split)
        for i, filename in tqdm(enumerate(self.raw_file_names), total=len(self.raw_file_names)):
            data_list = self.process_filename(filename)
            if i == 0:
                for j, data in enumerate(data_list):
                    torch.save(data, self._processed_dir / self._split / f"data_{j}.pt")
                prev_data_len = len(data_list)
            else:
                for j, data in enumerate(data_list):
                    torch.save(
                        data,
                        self._processed_dir / self._split / f"data_{j + prev_data_len}.pt")
                prev_data_len += len(data_list)
