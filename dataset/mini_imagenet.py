import os
import os.path
from typing import Optional, Callable

import pandas as pd

from .base import ClassificationVisionDataset
from .utils import split_data


class MiniImagenet(ClassificationVisionDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Optional[Callable] = None,
    ) -> None:

        self.root = os.path.join(root, 'mini-imagenet')

        super(MiniImagenet, self).__init__(self.root,
                                           train=train,
                                           transform=transform,
                                           target_transform=target_transform,
                                           loader=loader)

    def split_data(self) -> None:
        label_files = ['test.csv', 'train.csv', 'val.csv']
        data = []
        for lf in label_files:
            data.append(pd.read_csv(os.path.join(self.root, lf)))
        data = pd.concat(data, axis=0)
        data.filename = 'images/' + data.filename
        data = data.rename(columns={'filename': 'filepath'})
        split_data(self.root, data)
