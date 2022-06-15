from pyexpat import model
from turtle import color
import torch
import torch.nn as nn
from config import config
from loaddata import *
import torch.optim as optim
import os
from model import EncoderDecoder
import time
import json
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model = EncoderDecoder(n_class=5)
seg_model = seg_model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(seg_model.parameters(), lr=config['lr'])
pixel_acc_list = []
mIOU_list = []


def onehottocolor(pred):
    # print(type(pred))
    # print(pred.shape)
    im_rgb = np.zeros([pred.shape[0], 256, 256, 3], dtype=np.uint8)
    # print(pred.shape[0])
    # print(pred[0].max(1)[0].shape)
    images = pred.max(1)[1].cpu().numpy()[:, :, :]
    for b in range(pred.shape[0]):
        # print(images.shape)
        image = images[b, :, :]
        # print(image.shape)
        im_seg_RGB = np.zeros((256, 256, 3))
        for i in range(256):
            for j in range(256):
                if image[i, j] == 0:
                    im_seg_RGB[i, j, :] = [0, 0, 0]
                elif image[i, j] == 1:
                    im_seg_RGB[i, j, :] = [1, 0, 0]
                elif image[i, j] == 2:
                    im_seg_RGB[i, j, :] = [0, 1, 0]
                elif image[i, j] == 3:
                    im_seg_RGB[i, j, :] = [1, 1, 0]
                elif image[i, j] == 4:
                    im_seg_RGB[i, j, :] = [0, 0, 1]
        im_rgb[b] = np.uint8(im_seg_RGB)
    # print(im_rgb.shape)
    return im_rgb


def eval(seg_model, test_loader):
    seg_model.eval()
    total_ious = []
    pixel_accs = []
    # print(type(total_ious))

    for iter, (images, labels) in enumerate(test_loader):  # batch is 1 in this case
        inputs = torch.FloatTensor(images).to(device)
        # if use_gpu:
        inputs = inputs.to(device)

        output = seg_model(inputs)

        # only save the 1st image for comparison
        if iter == 0:
            # generate images
            input_np = images[0].data.cpu().numpy().transpose(1, 2, 0)
            output_np = output[0].data.cpu().numpy().transpose(1, 2, 0)
            gt_np = labels[0].data.cpu().numpy().transpose(1, 2, 0)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1,
                                                    config["num_classes"]).argmax(axis=1).reshape(N, h, w)
        target = labels.data.cpu().numpy().transpose(
            0, 2, 3, 1).reshape(-1, config["num_classes"]).argmax(axis=1).reshape(N, h, w)

        for p, t in zip(pred, target):
            # print(type(total_ious))
            total_ious.append(iou_func(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU of all datas
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("pix_acc: {:.4f}, meanIoU: {:.4f}".format(
        pixel_accs, np.nanmean(ious)))
    return pixel_accs, np.nanmean(ious)


def iou_func(pred, target):
    # Calculates class intersections over unions per epoch
    ious = []
    for cls in range(config["num_classes"]):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            # if there is no ground truth, do not include in evaluation
            ious.append(float("nan"))
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    # Calculates pixel accuracy per epoch
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


def save_result(file_name, input, label, output, n_samples=3):
    input_np = input[:n_samples].data.cpu().numpy().transpose(0, 2, 3, 1)
    # label_np = label[:n_samples].data.cpu().numpy().transpose(0, 2, 3, 1)
    label_np = onehottocolor(label[:n_samples])
    output_np = onehottocolor(output[:n_samples])
    result_list = []
    for k in range(n_samples):
        # tmp = np.zeros([256, 256, 5], dtype=np.float32)
        # for i in range(256):
        #     for j in range(256):
        #         tmp[i, j, output_np[k][i, j].argmax()] = 1
        # print(tmp.shape)
        result = np.hstack((input_np[k], label_np[k], output_np[k]))
        result_list.append(result)

    # horizontally stack original image and its corresponding segmentation results
    vstack_image = np.vstack(result_list)
    new_im = Image.fromarray(np.uint8(vstack_image*255))
    new_im.save(file_name)


for epoch in range(config['num_epochs']):
    ts = time.time()
    print("==================================================================")
    for iter, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        # print(labels)
        pred = seg_model(images)
        # print(pred)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        if iter % 10 == 0:
            print("epoch:{:2}, iter:{:2}, loss: {:.4f}".format(
                epoch, iter, loss.data.item()))
    print("Finish epoch:{:2}, time elapsed: {:.4f}".format(
        epoch, time.time() - ts))
    print("Start evaluation ...")

    # evaluate
    # 每個epoch都計算一次pixel_acc和meanIoU
    acc, iou = eval(seg_model, test_loader)
    pixel_acc_list.append(acc)
    mIOU_list.append(iou)

    print("Output test results ...")
    file_name = config["result_path"] + "/" + str(epoch).zfill(3) + ".jpg"
    for iter, (images, labels) in enumerate(test_loader):
        # if use_gpu:
        images = images.to(device)

        labels = labels.to(device)

        outputs = seg_model(images)
        save_result(file_name, images, labels, outputs)
        break

    print("Save model ...")
    model_path = config["result_path"] + "/" + "segnet.pt"
    if pixel_acc_list[-1] >= max(pixel_acc_list):
        torch.save(seg_model.state_dict(), model_path)
    print("========================================")

    highest_pixel_acc = max(pixel_acc_list)
    highest_mIOU = max(mIOU_list)

    highest_pixel_acc_epoch = pixel_acc_list.index(highest_pixel_acc)
    highest_mIOU_epoch = mIOU_list.index(highest_mIOU)

    # Extract evaluation record
    record_path = config["result_path"] + "/record.json"
    ret = json.dumps({"acc": pixel_acc_list, "iou": mIOU_list})
    with open(record_path, 'w') as fp:
        fp.write(ret)

    print("The highest mIOU is {} and is achieved at epoch-{}".format(highest_mIOU,
          highest_mIOU_epoch+1))
    print("The highest pixel accuracy  is {} and is achieved at epoch-{}".format(
        highest_pixel_acc, highest_pixel_acc_epoch+1))
