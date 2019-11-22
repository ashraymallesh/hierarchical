import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from fastai.vision import Image, Transform
import numpy as np
import torchvision
tpi = torchvision.transforms.ToPILImage()
tt = torchvision.transforms.ToTensor()


class ModifiedMNISTDataset(Dataset):
    """Modified MNIST dataset."""

    def __init__(self, x: np.ndarray, y: np.ndarray = None,
                 to_rgb: bool = False,
                 fastai_transform=None,
                 torchvision_transform=None,
                 resize=None):
        self.images = torch.Tensor(x/256).unsqueeze(1)
        if to_rgb:
            self.images = self.images.repeat(1,3,1,1)
        if resize:
            rs = torchvision.transforms.Resize(resize)
            self.images = torch.stack([tt(rs(tpi(i))) for i in self.images])
        self.labels = y
        self.to_rgb = to_rgb
        self.fastai_transform = fastai_transform
        self.torchvision_transform = torchvision_transform


    @classmethod
    def from_files(cls, x_pkl_file, y_csv_file = None,
                   to_rgb: bool = False,
                   fastai_transform=None,
                   torchvision_transform=None,
                   resize=None):
        """Constructor from files

        Included as separate constructor since pickling after train-test split
        does not work due to memory error. Therefore, we do the train/test split
        in memory and instantiate the Dataset object directly from the arrays.

        Args:
            x_pkl_file (string): Path to the pickle file with images.
            y_csv_file (string): Path to the csv file with labels.
            to_rgb (bool): Whether to expand to 3 channels for pretrained models
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        x = pd.read_pickle(x_pkl_file)
        if y_csv_file:
            y =  pd.read_csv(y_csv_file)["Label"].to_numpy()
        else:
            y = None
        return cls(x, y, to_rgb, fastai_transform, torchvision_transform, resize)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.fastai_transform:
            image = Image(image).apply_tfms(self.fastai_transform).px
        elif self.torchvision_transform:
            image = self.torchvision_transform(tpi(image))
        if self.labels is None:
            return image
        else:
            label = torch.LongTensor([self.labels[idx]]).squeeze()
            return image, label


def train_test_split(x: np.ndarray, y: pd.DataFrame,
                     y_label_col="Label", y_id_col="Id", split = 0.9):
    train_idx = []
    valid_idx = []
    for label in y[y_label_col].unique():
        label_subset = y.loc[y[y_label_col] == label]
        s = int(len(label_subset) * split)
        train_idx += label_subset[y_id_col].to_list()[:s]
        valid_idx += label_subset[y_id_col].to_list()[s:]

    train_x = x[train_idx]
    valid_x = x[valid_idx]
    train_y = y[y_label_col].iloc[train_idx].to_list()
    valid_y = y[y_label_col].iloc[valid_idx].to_list()

    return train_x, train_y, valid_x, valid_y