import torch
from model import *
# import segmentation.dataloader as dataloader
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset


seg_model = EncoderDecoder(n_class=5)
seg_model = seg_model.cuda()

seg_model.load_state_dict(torch.load("segnet_New.pt"))
print(2)
seg_model.eval()


def segmentation(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.asarray(frame)
    frame = frame.astype(float)/255.0
    frame = torch.from_numpy(frame.copy()).float()
    frame = frame.permute(2, 0, 1)

    input = torch.FloatTensor(frame)
    input = input.unsqueeze(0)
    input = input.cuda()
    output = seg_model(input)

    output_np = output.data.cpu().numpy().transpose(0, 2, 3, 1)
    result_frame = np.zeros([256, 256, 3], dtype=np.float32)
    for i in range(256):
        for j in range(256):
            idx = output_np[0][i, j].argmax()  # get maximu value is its class
            if idx == 0:  # background
                result_frame[i, j, :] = [0, 0, 0]
            if idx == 1:  # red_line
                result_frame[i, j, :] = [128, 0, 0]
            if idx == 2:  # black_line
                result_frame[i, j, :] = [0, 128, 0]
            if idx == 3:  # obstacle
                result_frame[i, j, :] = [128, 128, 0]
            if idx == 4:  # finish_line
                result_frame[i, j, :] = [0, 0, 128]

    result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

    return result_frame
