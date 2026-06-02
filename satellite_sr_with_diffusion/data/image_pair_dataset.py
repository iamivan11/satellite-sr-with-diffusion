"""
Custom dataset for loading paired image data (e.g., HR/LR).
Handles a specific 'hr' and 'lr' subdirectory structure and ensures
data integrity by checking for matching file counts.
"""

import os

from PIL import Image
from torch.utils.data import Dataset

import data.util as Util


class ImagePairDataset(Dataset):
    """
    A generic dataset for image-to-image tasks like super-resolution.

    Expected directory structure:
    - dataroot/
      - hr/  (high-resolution target images)
      - lr/  (low-resolution input images, typically upsampled to match hr size)

    Args:
        dataroot: Root directory containing 'hr' and 'lr' subdirectories.
        datatype: Type of data (must be 'img').
        split: Dataset split ('train', 'val', 'test').
        data_len: Number of samples to use (-1 for all).
    """

    def __init__(self, dataroot, datatype, split="train", data_len=-1, **kwargs):
        self.datatype = datatype
        self.data_len = data_len
        self.split = split

        if datatype == "img":
            hr_path_root = os.path.join(dataroot, "hr")
            lr_path_root = os.path.join(dataroot, "lr")

            if not os.path.isdir(hr_path_root):
                raise FileNotFoundError(f"HR directory not found at: {hr_path_root}")
            if not os.path.isdir(lr_path_root):
                raise FileNotFoundError(f"LR directory not found at: {lr_path_root}")

            self.hr_path = Util.get_paths_from_images(hr_path_root)
            self.sr_path = Util.get_paths_from_images(lr_path_root)

            if len(self.hr_path) != len(self.sr_path):
                raise ValueError(
                    f"Mismatch in number of images in '{dataroot}'. "
                    f"Found {len(self.hr_path)} HR images and {len(self.sr_path)} LR images."
                )

            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(f"data_type [{datatype}] is not recognized.")

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return self.data_len

    def __getitem__(self, index):
        """Fetches the training/validation sample at the given index."""
        try:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
        except Exception as e:
            raise OSError(
                f"Failed to load images at index {index}: HR path '{self.hr_path[index]}', "
                f"LR path '{self.sr_path[index]}'. Original error: {e}"
            ) from e

        # Apply augmentations and convert to tensors
        [img_SR, img_HR] = Util.transform_augment(
            [img_SR, img_HR], split=self.split, min_max=(-1, 1)
        )

        return {"HR": img_HR, "SR": img_SR, "Index": index}
