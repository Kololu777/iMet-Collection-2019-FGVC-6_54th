# Author:KNSBko
# date 20190626

# imet-FGVC/55th のdataaugumention torchvisionで回転とか行いました。
# 使い方は
# If mode is train data.
# image=train_post_ReC(image)
# image = aut2(image)
# If mode is validation and test data.
# image=post_ReC(image)
# 時間やもう一度使うことがあれば詳しく書きます
from torchvision.transforms import CenterCrop
import torchvision.transforms as transform
from PIL import Image
import random


def aut(img):
    crop = CenterCrop(320)
    RR90 = transform.RandomRotation(90)
    img = img.transpose(img.ROTATE_90)
    img = RR90(crop(img))
    # if random.random() < 0.5:
    img = crop(img)
    return img


def RRC(img):
    RRC = transform.RandomResizedCrop(320)
    img = RRC(img)
    return img


def aut2(img):
    # Rotate90,HFlip
    if random.random() < 0.5:
        img = img.transpose(Image.ROTATE_90)
    if random.random() < 0.5:
        RrHF = transform.RandomHorizontalFlip(1.)
        img = RrHF(img)
    return img


def CC(img):
    crop = CenterCrop(320)
    img = crop(img)
    return img


def FR(img, fine_size, c):
    if c == 0:
        img = img.resize(((img.size[0] * fine_size) // img.size[1], fine_size))
    else:
        img = img.resize((fine_size, (img.size[1] * fine_size) // img.size[0]))
    return img


def post_ReC(img):
    if img.size[0] // img.size[1] >= 1.35 or img.size[1] // img.size[0] >= 1.35:
        if img.size[0] > img.size[1]:
            img = FR(img, 320, 0)
        else:
            img = FR(img, 320, 1)
        img = CC(img)
    elif img.size[0] <= 320 and img.size[1] <= 320:
        img = img.resize((320, 320))

    else:
        if img.size[0] > img.size[1]:
            img = FR(img, 320, 0)
            img = CC(img)
        else:
            img = FR(img, 320, 1)
            img = CC(img)
    return img


def train_post_ReC(img):
    if img.size[0] // img.size[1] >= 1.35 or img.size[1] // img.size[0] >= 1.35:
        if img.size[0] > img.size[1]:
            img = FR(img, 320, 0)
        else:
            img = FR(img, 320, 1)
        if random.random() < 0.3:
            img = CC(img)
        else:
            img = RRC(img)
        return img
    elif img.size[0] <= 320 and img.size[1] <= 320:
        if random.random() < 0.3:
            img = img.resize((320, 320))
        else:
            img = img.resize((350, 350))
            img = RRC(img)
        return img

    else:
        if img.size[0] > img.size[1]:
            img = FR(img, 320, 0)
        else:
            img = FR(img, 320, 1)
        if random.random() < 0.3:
            img = CC(img)
        else:
            img = RRC(img)
        return img


def RRsC(img):
    RRsC = transform.RandomResizedCrop(320)
    img = RRsC(img)
    return img


def TTA_post_ReC(img):
    if img.size[0] // img.size[1] >= 1.35 or img.size[1] // img.size[0] >= 1.35:
        if img.size[0] > img.size[1]:
            img = FR(img, 320, 0)
        else:
            img = FR(img, 320, 1)
        img = RRsC(img)
        return img
    elif img.size[0] <= 320 and img.size[1] <= 320:
        img = img.resize((350, 350))
        img = RRsC(img)
        return img

    else:
        if img.size[0] > img.size[1]:
            img = FR(img, 320, 0)
        else:
            img = FR(img, 320, 1)
        img = RRsC(img)
        return img
