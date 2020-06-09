import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import numpy as np
import os
import json
from skimage.feature import canny
from skimage.color import rgb2gray
from PIL import Image
import matplotlib.pyplot as plt
from .prepare_images import Prepare_img
from .utils import Get_gradient


class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_folder, mode, sigma, crop_size, scale):
        super().__init__()
        self.list_folder = list_folder
        self.mode = mode
        self.sigma = sigma
        self.crop_size = crop_size
        self.scale = scale

        if mode == "train":
            #load original HR image for training
            with open(os.path.join(self.list_folder, "train_images.json"), "r") as j:
                self.ori_imgs = json.load(j)
        elif mode == "test":
            #load original HR image for testing
            with open(os.path.join(self.list_folder, "test_images.json"), "r") as j:
                self.ori_imgs = json.load(j)
        elif mode == "eval":
            #load original HR image for eval
            with open(os.path.join(self.list_folder, "eval_images.json"), "r") as j:
                self.ori_imgs = json.load(j)

        self.prepare_img = Prepare_img(self.mode, self.crop_size, self.scale)

        self.get_grad = Get_gradient()

    def __getitem__(self, index):
        """
        Required method for DataLoader

        """
        ori_img = Image.open(self.ori_imgs[index], "r")
        
        lr_img, hr_img = self.prepare_img(ori_img)

        #lr_edge = self.load_edge(np.array(lr_img))
        #hr_edge = self.load_edge(np.array(hr_img))

        lr_img_tensor = F.to_tensor(lr_img).float()
        hr_img_tensor = F.to_tensor(hr_img).float()

        lr_gradient_tensor = self.get_grad(lr_img_tensor).detach()
        hr_gradient_tensor = self.get_grad(hr_img_tensor).detach()

        #lr_edge_tensor = F.to_tensor(lr_edge).float()
        #hr_edge_tensor = F.to_tensor(hr_edge).float()

        return lr_img_tensor, hr_img_tensor, lr_gradient_tensor, hr_gradient_tensor

    def __len__(self):
        """
        Required method for DataLoader
        """
        return len(self.ori_imgs)

    def load_name(self, index):
        name = self.ori_imgs[index]
        return os.path.basename(name)

    def load_edge(self, img):
        return canny(rgb2gray(img), sigma=self.sigma).astype(np.float)

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


