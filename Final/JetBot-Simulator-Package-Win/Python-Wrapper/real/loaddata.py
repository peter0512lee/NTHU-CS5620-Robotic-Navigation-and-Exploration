from cProfile import label
from re import X
from tkinter import Frame
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import random
from natsort import natsorted
import numpy as np
from config import config
import cv2
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()


def read_data(rootpath):
    frame_list = []
    label_list = []
    for dir in natsorted(os.listdir(rootpath)):
        if os.path.isdir(os.path.join(rootpath, dir)):
            frame_label = np.zeros((256, 256, 5), dtype=np.uint8)
            label = np.asarray(Image.open(
                os.path.join(rootpath, dir, "label.png")))
            label = cv2.resize(label, (256, 256))
            # print(label.shape)
            for i in range(frame_label.shape[0]):
                for j in range(frame_label.shape[1]):
                    if label[i, j] == 0:
                        frame_label[i, j, 0] = 255
                    elif label[i, j] == 1:
                        frame_label[i, j, 1] = 255
                    elif label[i, j] == 2:
                        frame_label[i, j, 2] = 255
                    elif label[i, j] == 3:
                        frame_label[i, j, 3] = 255
                    elif label[i, j] == 4:
                        frame_label[i, j, 4] = 255
            frame = cv2.imread(os.path.join(rootpath, dir, "img.png"))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))
            frame_list.append(frame)
            label_list.append(frame_label)
    frame_list = np.asarray(frame_list)
    label_list = np.asarray(label_list)
    return frame_list, label_list


def sample_data():
    frame_list, label_list = read_data(config["data_path"])
    train_x, test_x = train_test_split(
        frame_list, random_state=777, train_size=0.8)
    train_y, test_y = train_test_split(
        label_list, random_state=777, train_size=0.8)
    return train_x, train_y, test_x, test_y


class FrameDataset(Dataset):
    def __init__(self, x, y, transform, flag="train"):
        self.x = x
        self.y = y
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        frame = self.x[idx]
        label = self.y[idx]
        if self.transform:
            frame = self.transform(frame)
            # frame = TF.normalize(frame, [0.485, 0.456, 0.406], [
            # 0.229, 0.224, 0.225])
            label = self.transform(label)
            # if random.random() > 0.5 and self.flag == "train":
            #     frame = TF.hflip(frame)
            #     label = TF.hflip(label)
        return frame, label


transform = transforms.Compose([
    transforms.ToTensor(),
])
train_x, train_y, test_x, test_y = sample_data()
train_dataset = FrameDataset(train_x, train_y, transform, flag="train")
test_dataset = FrameDataset(test_x, test_y, transform, flag="test")
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=config["batch_size"], shuffle=False)

if __name__ == "__main__":
    # train_x, train_y, test_x, test_y = sample_data()
    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)
    # cv2.imshow("frame", train_x[0])
    # cv2.imshow("label", train_y[0])
    # cv2.waitKey(0)
    for i, (frame, label) in enumerate(train_loader):
        print(frame.shape)
        print(label.shape)
