import os
import os.path
from typing import Optional, Callable

import pandas as pd

from .base import ClassificationVisionDataset
from .utils import split_data


class Caltech101(ClassificationVisionDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Optional[Callable] = None,
    ) -> None:

        self.root = os.path.join(root, 'caltech101')

        super(Caltech101, self).__init__(self.root,
                                         train=train,
                                         transform=transform,
                                         target_transform=target_transform,
                                         loader=loader)

    def split_data(self) -> None:
        categories = sorted(os.listdir(os.path.join(self.root, '101_ObjectCategories')))
        categories.remove("BACKGROUND_Google")  # this is not a real class

        data = []
        for cls in categories:
            imgs = os.listdir(os.path.join(self.root, '101_ObjectCategories', cls))
            for img in imgs:
                data.append([os.path.join('101_ObjectCategories', cls, img).replace('\\', '/'), cls])
        data = pd.DataFrame(data, columns=['filepath', 'label'])
        split_data(self.root, data)
