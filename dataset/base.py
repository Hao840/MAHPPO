import os
import warnings
from typing import Optional, Callable, Any, Tuple

import numpy as np
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset


class ClassificationVisionDataset(VisionDataset):
    '''
        Args:
        root (string): Root directory of the Dataset.
        train (bool, optional): The dataset split, train(True) or test(False).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
    '''

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Optional[Callable] = None,
    ) -> None:

        super(ClassificationVisionDataset, self).__init__(root, transform=transform,
                                                          target_transform=target_transform)

        if not self._check_split():
            warnings.warn('split files not found or corrupted, re-splitting the dataset')
            self.split_data()

        self.class_to_idx: dict = np.load(os.path.join(root, 'custom_dict.npy'), allow_pickle=True).item()
        self.categories = list(self.class_to_idx.keys())

        if train:
            samples = pd.read_csv(os.path.join(root, 'custom_train.csv'))
        else:
            samples = pd.read_csv(os.path.join(root, 'custom_test.csv'))

        samples.filepath = root + '/' + samples.filepath
        self.samples = samples.to_numpy().tolist()

        self.loader = loader if loader is not None else default_loader

    def split_data(self) -> None:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        path, target = self.samples[index]
        target = self.class_to_idx[target]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.samples)

    def _check_split(self) -> bool:
        # check the existence of split files
        check_list = ['custom_train.csv', 'custom_test.csv', 'custom_dict.npy']
        for file in check_list:
            if not os.path.exists(os.path.join(self.root, file)):
                return False
        return True
