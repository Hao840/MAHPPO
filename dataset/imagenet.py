import os
import os.path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import check_integrity, verify_str_arg

META_FILE = "ILSVRC2012/meta.bin"


class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root: str, train: bool = True, **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(root)

        split = "train" if train else "val"
        self.split = verify_str_arg(split, "split", ("train", "val"))

        wnid_to_classes = load_meta_file(self.root)[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)


def load_meta_file(root: str, file: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file)
    else:
        msg = ("The meta file {} is not present in the root directory or is corrupted. "
               "This file is automatically created by the ImageNet dataset.")
        raise RuntimeError(msg.format(file, root))
