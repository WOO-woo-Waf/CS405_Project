import os
import torch
import sys
import numpy as np
from PIL import Image
import json

# # 添加路径到系统路径中
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from .segbase import SegmentationDataset


class My_CitySegmentation(SegmentationDataset):
    """Cityscapes Semantic Segmentation Dataset with JSON-based annotations support.

    Parameters
    ----------
    root : string
        Path to the Cityscapes folder.
    split : string
        'train', 'val', or 'test'.
    dataset_type : string
        'gtFine' or 'gtCoarse'.
    transform : callable, optional
        A function that transforms the image and label.
    """
    NUM_CLASS = 19
    NUM_CLASSES_FINE = 19  # For gtFine
    NUM_CLASSES_COARSE = 34  # For gtCoarse

    def __init__(self, root='../datasets', split='train', dataset_type='gtFine', mode=None, transform=None, **kwargs):
        super(My_CitySegmentation, self).__init__(root, split, mode, transform, **kwargs)
        self.dataset_type = dataset_type  # Dataset type: 'gtFine' or 'gtCoarse'
        self.split = split
        assert os.path.exists(self.root), "Dataset root path not found. Check the provided path."

        # Set number of classes and file suffixes based on dataset type
        if self.dataset_type == 'gtFine':
            self.NUM_CLASSES = self.NUM_CLASSES_FINE
            self.img_suffix = '_color.png'
            self.mask_suffix = '_labelIds.png'
            self.json_suffix = '_polygons.json'
        elif self.dataset_type == 'gtCoarse':
            self.NUM_CLASSES = self.NUM_CLASSES_COARSE
            self.img_suffix = '_color.png'
            self.mask_suffix = '_labelIds.png'
            self.json_suffix = '_polygons.json'
        else:
            raise ValueError("Invalid dataset_type. Supported values: 'gtFine', 'gtCoarse'.")

        # Load image and mask paths
        self.images, self.mask_paths, self.json_paths = self._get_city_pairs()
        assert len(self.images) == len(self.mask_paths) == len(self.json_paths), "Mismatch between image, mask, and JSON files."

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.root} with split={self.split} and dataset_type={self.dataset_type}.")

    def __getitem__(self, index):
        """Retrieve an image, its corresponding mask, and JSON annotations."""
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index])

        # Load JSON annotations
        with open(self.json_paths[index], 'r') as f:
            json_data = json.load(f)

        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index]), json_data

        # Apply transformations
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        if self.transform is not None:
            img = self.transform(img)

        return img, mask, os.path.basename(self.images[index]), json_data

    def _mask_transform(self, mask):
        """Transform mask into tensor."""
        return torch.LongTensor(np.array(mask).astype('int32'))

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.images)

    @property
    def pred_offset(self):
        """Offset for predictions."""
        return 0

    def _get_city_pairs(self):
        """
        Helper function to retrieve image, mask, and JSON file paths, considering folder-based naming patterns.
        """
        img_paths = []
        mask_paths = []
        json_paths = []
        img_folder = os.path.join(self.root, self.dataset_type, self.split)

        # Iterate over each city folder
        for city_folder in os.listdir(img_folder):
            city_path = os.path.join(img_folder, city_folder)
            if not os.path.isdir(city_path):
                continue

            # Iterate over all files in the city folder
            for file_name in os.listdir(city_path):
                if file_name.endswith(self.img_suffix):
                    # Extract file prefix
                    prefix = '_'.join(file_name.split('_')[:4])

                    # Construct file paths
                    img_path = os.path.join(city_path, f"{prefix}{self.img_suffix}")
                    mask_path = os.path.join(city_path, f"{prefix}{self.mask_suffix}")
                    json_path = os.path.join(city_path, f"{prefix}{self.json_suffix}")

                    # Ensure all files exist
                    if os.path.isfile(img_path) and os.path.isfile(mask_path) and os.path.isfile(json_path):
                        img_paths.append(img_path)
                        mask_paths.append(mask_path)
                        json_paths.append(json_path)
                    else:
                        print(f"Missing files: {img_path}, {mask_path}, {json_path}")


        print(f"Found {len(img_paths)} images in {img_folder}")
        return img_paths, mask_paths, json_paths


if __name__ == '__main__':
    # Test loading gtFine dataset
    gtFine_dataset = My_CitySegmentation(
        root='../datasets/gtFine_trainvaltest/',
        split='train',
        dataset_type='gtFine'
    )
    print(f"Loaded {len(gtFine_dataset)} gtFine samples.")
    
    gtCoarse_dataset = My_CitySegmentation(
        root='../datasets/gtCoarse/',
        split='train',
        dataset_type='gtCoarse'
    )
    print(f"Loaded {len(gtCoarse_dataset)} gtCoarse samples.")