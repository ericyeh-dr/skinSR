import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import Dataset
from .loss import ContentLoss, StyleLoss
from .utils import Get_gradient

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.relu(out)

        return out

class VGGBlock_oneconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu(out)

        return out

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super().__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.unet_weights_path = os.path.join(config.PATH, name+".pth")

    def load(self):
        if os.path.exists(self.unet_weights_path):
            print("Loading ...{}".format(self.name))

            if torch.cuda.is_available():
                data = torch.load(self.unet_weights_path)
            else:
                data = torch.load(self.unet_weights_path, map_location = lambda storage, loc:storage)

            self.unet.load_state_dict(data["unet"])
            self.iteration = data["iteration"]

    def save(self):
        print("Saving...{}...".format(self.name))

        torch.save({
            "iteration": self.iteration,
            "unet": self.unet.state_dict()
            }, self.unet_weights_path)

class BaseModel2(nn.Module):
    def __init__(self, name, config):
        super().__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.unet_weights_path = os.path.join(config.PATH, name+".pth")

    def load(self):
        if os.path.exists(self.unet_weights_path):
            print("Loading ...{}".format(self.name))

            if torch.cuda.is_available():
                data = torch.load(self.unet_weights_path)
            else:
                data = torch.load(self.unet_weights_path, map_location = lambda storage, loc:storage)

            self.unet2.load_state_dict(data["unet"])
            self.iteration = data["iteration"]

    def save(self):
        print("Saving...{}...".format(self.name))

        torch.save({
            "iteration": self.iteration,
            "unet": self.unet2.state_dict()
            }, self.unet_weights_path)

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self, init_type = "normal", gain = 0.02):

        def init_func(module):
            classname = module.__class__.__name__
                
            if hasattr(module, "weight") and (classname.find("Linear") != -1 or classname.find("Conv") != -1):
                if init_type == "normal":
                    nn.init.normal_(module.weight.data, mean = 0.0, std = gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(module.weight.data, gain = gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(module.weight.data, a = 0, mode = "fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(module.weight.data, gain = gain)
                
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)

            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(module.weight.data, 1.0, std = gain)
                nn.init.constant_(module.bias.data, 0.0)

        self.apply(init_func)


class UNet(BaseNet):
    def __init__(self, config, input_channels=3):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024, 16, 32]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.convb0_5 = VGGBlock_oneconv(input_channels, nb_filter[-1])
        self.convb0_6 = VGGBlock_oneconv(nb_filter[-1], nb_filter[-2])

        self.conv0_0 = VGGBlock_oneconv(input_channels, nb_filter[0])
        self.conv1_0 = VGGBlock_oneconv(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock_oneconv(nb_filter[1], nb_filter[2])
        self.conv3_0 = VGGBlock_oneconv(nb_filter[2], nb_filter[3])
        self.conv4_0 = VGGBlock_oneconv(nb_filter[3], nb_filter[4])

        self.conv3_1 = VGGBlock_oneconv(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2 = VGGBlock_oneconv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3 = VGGBlock_oneconv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4 = VGGBlock_oneconv(nb_filter[0]+nb_filter[1], nb_filter[0])

        self.conv0_5 = VGGBlock_oneconv(nb_filter[-1]+nb_filter[0], nb_filter[-1])
        self.conv0_6 = VGGBlock_oneconv(nb_filter[-2]+nb_filter[-1], nb_filter[-2])

        self.final = nn.Conv2d(nb_filter[-2], input_channels, kernel_size=1)

        self.init_weights()

    def forward(self, input):
        xb0_5 = self.convb0_5(self.up(input))
        xb0_6 = self.convb0_6(self.up(xb0_5))
        
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        x0_5 = self.conv0_5(torch.cat([xb0_5, self.up(x0_4)], 1))
        x0_6 = self.conv0_6(torch.cat([xb0_6, self.up(x0_5)], 1))

        output = self.final(x0_6)

        return output


class NestedUNet(BaseNet):
    def __init__(self, config, input_channels = 3):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512, 8, 16]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.convb0_5 = VGGBlock(input_channels, nb_filter[-1], nb_filter[-1])
        self.convb0_6 = VGGBlock(nb_filter[-1], nb_filter[-2], nb_filter[-2])

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv0_5 = VGGBlock(nb_filter[-1]+nb_filter[0], nb_filter[-1], nb_filter[-1])
        self.conv0_6 = VGGBlock(nb_filter[-2]+nb_filter[-1], nb_filter[-2], nb_filter[-2])

        self.final = nn.Conv2d(nb_filter[-2], input_channels, kernel_size=1)
        
        self.init_weights()

    def forward(self, input):
        xb0_5 = self.convb0_5(self.up(input))
        xb0_6 = self.convb0_6(self.up(xb0_5))
        
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x0_5 = self.conv0_5(torch.cat([xb0_5, self.up(x0_4)], 1))
        x0_6 = self.conv0_6(torch.cat([xb0_6, self.up(x0_5)], 1))

        output = self.final(x0_6)

        return output


class UnetModel(BaseModel):
    def __init__(self, config):
        super().__init__("unet", config)

        self.config = config

        #components
        self.unet = NestedUNet(config)

        #loss functions
        L1_loss = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()
        content_loss = ContentLoss()
        style_loss = StyleLoss()

        #gradient
        self.get_grad = Get_gradient()

        self.add_module('unet', self.unet)

        self.add_module("L1_loss", L1_loss)
        self.add_module("MSE_loss", self.MSE_loss)
        self.add_module('content_loss', content_loss)
        self.add_module('style_loss', style_loss)
        
        self.add_module("get_grad", self.get_grad)

        self.optimizer = optim.Adam(
            params = self.unet.parameters(),
            lr = float(config.LR),
            betas = (config.BETA1, config.BETA2)
            )

    def forward(self, lr_images):
        outputs = self.unet.forward(lr_images)
        return outputs

    def backward(self, mix_loss):
        mix_loss.backward()
        self.optimizer.step()

    def process(self, lr_images, hr_images):
        self.iteration += 1

        # zero optimizers
        self.optimizer.zero_grad()

        # process outputs
        outputs = self.forward(lr_images)

        #mse_loss
        #mse_loss = self.MSE_loss(outputs, hr_images)

        l_loss = self.config.L1_LOSS_WEIGHT*self.L1_loss(outputs, hr_images)

        #mge_loss
        fake_grads = self.get_grad.forward(outputs)
        hr_grads = self.get_grad.forward(hr_images)
        mge_loss = self.MSE_loss(fake_grads, hr_grads)

        #content loss
        c_loss = self.config.CONTENT_LOSS_WEIGHT*self.content_loss(outputs, hr_images)

        #style loss
        s_loss = self.config.STYLE_LOSS_WEIGHT*self.style_loss(outputs, hr_images)

        #mix_loss
        mix_loss = l_loss + self.config.MGE_LOSS_WEIGHT*mge_loss + c_loss + s_loss

        # L1 loss
        #gen_l1_loss = self.L1_loss(outputs, hr_images) * self.config.L1_LOSS_WEIGHT
        #gen_loss += gen_l1_loss


        # content loss
        #gen_content_loss = self.content_loss(outputs, hr_images)
        #gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        #gen_loss += gen_content_loss

        # style loss
        #gen_style_loss = self.style_loss(outputs, hr_images)
        #gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        #gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_mge", mge_loss.item()),
            ("l_mix", mix_loss.item())
            ]

        return outputs, hr_grads, mix_loss, logs

class UnetModel2(BaseModel2):
    def __init__(self, config):
        super().__init__("unet2", config)

        self.config = config

        #components
        self.unet2 = UNet(config)

        #loss functions
        L1_loss = nn.L1Loss()
        MSE_loss = nn.MSELoss()
        content_loss = ContentLoss()
        style_loss = StyleLoss()

        #gradient
        self.get_grad = Get_gradient()

        self.add_module('unet2', self.unet2)

        self.add_module("L1_loss", L1_loss)
        self.add_module("MSE_loss", MSE_loss)
        self.add_module('content_loss', content_loss)
        self.add_module('style_loss', style_loss)
        
        self.add_module("get_grad", self.get_grad)

        self.optimizer = optim.Adam(
            params = self.unet2.parameters(),
            lr = float(config.LR),
            betas = (config.BETA1, config.BETA2)
            )

    def forward(self, lr_images):
        outputs = self.unet2.forward(lr_images)
        return outputs

    def backward(self, mix_loss):
        mix_loss.backward()
        self.optimizer.step()

    def process(self, lr_images, hr_images):
        self.iteration += 1

        # zero optimizers
        self.optimizer.zero_grad()

        # process outputs
        outputs = self.forward(lr_images)

        #mse_loss
        #mse_loss = self.MSE_loss(outputs, hr_images)

        l_loss = self.config.L1_LOSS_WEIGHT*self.L1_loss(outputs, hr_images)

        #mge_loss
        fake_grads = self.get_grad.forward(outputs)
        hr_grads = self.get_grad.forward(hr_images)
        #mge_loss = self.MSE_loss(fake_grads, hr_grads)
        l_grad_loss = self.L1_loss(fake_grads, hr_grads)

        #content loss
        #c_loss = self.config.CONTENT_LOSS_WEIGHT*self.content_loss(outputs, hr_images)

        #style loss
        #s_loss = self.config.STYLE_LOSS_WEIGHT*self.style_loss(outputs, hr_images)

        #mix_loss
        mix_loss = l_loss + self.config.MGE_LOSS_WEIGHT*l_grad_loss 

        # L1 loss
        #gen_l1_loss = self.L1_loss(outputs, hr_images) * self.config.L1_LOSS_WEIGHT
        #gen_loss += gen_l1_loss


        # content loss
        #gen_content_loss = self.content_loss(outputs, hr_images)
        #gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        #gen_loss += gen_content_loss

        # style loss
        #gen_style_loss = self.style_loss(outputs, hr_images)
        #gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        #gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_loss", l_loss.item()),
            ("l_grad_loss", l_grad_loss.item()),
            ("l_mix", mix_loss.item())
            ]

        return outputs, hr_grads, mix_loss, logs



