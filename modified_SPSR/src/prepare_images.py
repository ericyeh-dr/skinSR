from PIL import Image
import numpy as np
import random


class Prepare_img():
    def __init__(self, mode, crop_size, scale):
        self.mode = mode
        self.crop_size = crop_size
        self.scale = scale

    def __call__(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the PIL Image format

        """
        
        if self.mode == "train" or self.mode == "eval":
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # For saving memory, if width or height > 1024, crop it to 1024
            #if img.width > 900:
            #    img = img.crop((0, 0, 900, img.height))
            #if img.height > 600:
            #    img = img.crop((0, 0, img.width, 600))
            
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scale
            y_remainder = img.height % self.scale
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))
        
        lr_img = hr_img.resize((int(hr_img.width / self.scale), int(hr_img.height / self.scale)),
                               Image.BICUBIC)

        return lr_img, hr_img
