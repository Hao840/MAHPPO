import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .config import cfg


def split_data(
        root: str,
        data: pd.DataFrame,
        train_size: float = cfg.train_size,
        seed: int = cfg.random_seed
) -> None:
    '''
        Args:
        root (string): Root directory of the Dataset.
        data (pd.DataFrame): A dataframe contain (relative) path and label of samples,
            header of the two columns are 'filepath' and 'label'.
        train_size (float, optional): ratio of train samples, and should be between 0.0 and 1.0.
        seed (int, optional): random seed.
    '''

    # get split index
    split = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idxs, test_idxs = list(split.split(data.filepath, data.label))[0]

    # get split result
    np_df = data.to_numpy()
    train_data = pd.DataFrame(np_df[train_idxs]).sort_values(by=0)
    test_data = pd.DataFrame(np_df[test_idxs]).sort_values(by=0)
    categories = np.unique(data.label.values)
    categories.sort()
    label_dict = {label: i for i, label in enumerate(categories)}

    # save result
    train_data.to_csv(os.path.join(root, 'custom_train.csv'), index=False, header=['filepath', 'label'])
    test_data.to_csv(os.path.join(root, 'custom_test.csv'), index=False, header=['filepath', 'label'])
    np.save(os.path.join(root, 'custom_dict.npy'), label_dict)
    # label_dict = np.load(os.path.join(root, 'custom_dict.npy'), allow_pickle='TRUE').item()
