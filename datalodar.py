from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class iMetDataset(Dataset):
    def __init__(
            self,
            path,
            image_list,
            cls_number,
            label_list=None,
            mode='train',
            is_tta=False,
            fine_size=320,
            flag=0,
            interpolation=Image.BILINEAR):
        self.path = path
        self.imagelist = image_list
        self.label_list = label_list
        self.cls_number = cls_number
        self.mode = mode
        self.is_tta = is_tta
        self.fine_size = fine_size
        self.flag = flag
        self.interpolation = interpolation

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        # image = cv2.imread(self.path+'train/'+self.imagelist[idx]+'.png').astype(np.float32) / 255
        image = Image.open(self.path + 'train/' + self.imagelist[idx] + '.png')

        if self.mode == 'train':
            # label = self.label_list[idx]
            # i.e) DataAugumention
            label = np.eye(self.cls_number, dtype=np.float)[
                self.label_list[idx]].sum(axis=0)
            image = np.asarray(image).astype(np.float32) / 255
            image = image.transpose(2, 0, 1)
            return image, label

        if self.mode == 'val':
            # i.e) DataAugumention
            label = np.eye(self.cls_number, dtype=np.float)[
                self.label_list[idx]].sum(axis=0)
            image = np.asarray(image).astype(np.float32) / 255
            image = image.transpose(2, 0, 1)
            return image, label

        if self.mode == 'test':
            # i.e) DataAugumention or TTA
            image = np.asarray(image).astype(np.float32) / 255
            image = image.transpose(2, 0, 1)
            return image
