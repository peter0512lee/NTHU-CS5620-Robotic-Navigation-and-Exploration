from tkinter import Frame
from matplotlib import image
import torch.utils.data as data
import numpy as np
import torch

import segmentation.configs as configs


class SimDataset(data.Dataset):
    def __init__(self, frame, n_class=configs.num_class):
        self.n_class = n_class
        self.frame = frame

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frame = np.asarray(self.frame)
        frame = frame.astype(float)/255.0
        frame = torch.from_numpy(frame.copy()).float()
        frame = frame.permute(2, 0, 1)
