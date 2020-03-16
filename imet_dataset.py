import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from transforms.imettrans import aut2, train_post_ReC, post_ReC, TTA_post_ReC
import torchvision.transforms as transform


class iMetDataset(Dataset):
    def __init__(
            self,
            path,
            image_list,
            fine_size,
            label_list=None,
            mode='train',
            is_tta=False,
            flag=0,
            TTA_list=None,
            interpolation=Image.BILINEAR):
        self.path = path
        self.imagelist = image_list
        self.label_list = label_list
        self.mode = mode
        self.is_tta = is_tta
        self.fine_size = fine_size
        self.flag = flag
        self.TTA_list = TTA_list
        self.interpolation = interpolation

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        # image = cv2.imread(self.path+'train/'+self.imagelist[idx]+'.png').astype(np.float32) / 255
        # image = Image.open(self.path + 'train/' + self.imagelist[idx] + '.png')
        if self.mode == 'train':
            image = Image.open(
                self.path +
                'train/' +
                self.imagelist[idx] +
                '.png')
            # label = self.label_list[idx]
            image = train_post_ReC(image)
            label = np.eye(1103, dtype=np.float)[
                self.label_list[idx]].sum(axis=0)
            image = aut2(image)
            image = np.asarray(image).astype(np.float32) / 255
            image = image.transpose(2, 0, 1)
            return image, label

        if self.mode == 'val':
            image = Image.open(
                self.path +
                'train/' +
                self.imagelist[idx] +
                '.png')
            image = post_ReC(image)
            label = np.eye(1103, dtype=np.float)[
                self.label_list[idx]].sum(axis=0)
            image = np.asarray(image).astype(np.float32) / 255
            image = image.transpose(2, 0, 1)
            return image, label

        if self.mode == 'test':
            image = Image.open(
                self.path +
                'test/' +
                self.imagelist[idx] +
                '.png')
            if self.is_tta:
                if self.TTA_list == 1:
                    image = post_ReC(image)
                elif self.TTA_list == 2:
                    image = post_ReC(image)
                    RrHF = transform.RandomHorizontalFlip(1.)
                    image = RrHF(image)
                else:
                    image = TTA_post_ReC(image)
            image = post_ReC(image)
            image = np.asarray(image).astype(np.float32) / 255
            image = image.transpose(2, 0, 1)
            return image


def null_collate(batch):
    batch_size = len(batch)
    input = []
    truth = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
    input = torch.from_numpy(np.array(input)).float()
    truth = torch.from_numpy(np.array(truth)).float()
    return input, truth


def test_collate(batch):
    batch_size = len(batch)
    input = []
    for b in range(batch_size):
        input.append(batch[b])
    input = torch.from_numpy(np.array(input)).float()
    return input
