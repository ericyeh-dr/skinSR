import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .components import SRGenerator2, GradDiscriminator, SRDiscriminator
from .dataset import Dataset
from .loss import AdversarialLoss, ContentLoss, StyleLoss, SimpleContentLoss
from .utils import Get_gradient
import os

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super().__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name+"_gen.pth")
        self.dis_grad_weights_path = os.path.join(config.PATH, name+"_dis_grad.pth")
        self.dis_sr_weights_path = os.path.join(config.PATH, name+"_dis_sr.pth")

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print("Loading generator...{}".format(self.name))

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location = lambda storage, loc:storage)

            self.generator.load_state_dict(data["generator"])
            self.iteration = data["iteration"]

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_grad_weights_path):
            print("loading gradient discriminator...{}".format(self.name))
            
            if torch.cuda.is_available():
                data = torch.load(self.dis_grad_weights_path)
            else:
                data = torch.load(self.dis_grad_weights_path, map_location = lambda storage, loc:storage)

            self.grad_discriminator.load_state_dict(data["graddiscriminator"])

        if self.config.MODE == 1 and os.path.exists(self.dis_sr_weights_path):
            print("loading sr discriminator...{}".format(self.name))
            
            if torch.cuda.is_available():
                data = torch.load(self.dis_sr_weights_path)
            else:
                data = torch.load(self.dis_sr_weights_path, map_location = lambda storage, loc:storage)

            self.sr_discriminator.load_state_dict(data["srdiscriminator"])

    def save(self):
        print("Saving...{}...".format(self.name))

        torch.save({
            "iteration": self.iteration,
            "generator": self.generator.state_dict()
            }, self.gen_weights_path)

        torch.save({
            "graddiscriminator": self.grad_discriminator.state_dict()
            }, self.dis_grad_weights_path)

        torch.save({
            "srdiscriminator": self.sr_discriminator.state_dict()
            }, self.dis_sr_weights_path)

class SRModel2(BaseModel):
    def __init__(self, config):
        super().__init__("SRModel2", config)

        self.config = config
        # generator input: [rgb(3) + edge(1)]
        # discriminator input: (rgb(3) 
        self.generator = SRGenerator2()
        self.grad_discriminator = GradDiscriminator(in_channels = 3, use_sigmoid = config.GAN_LOSS != "hinge")
        self.sr_discriminator = SRDiscriminator(in_channels = 3, use_sigmoid = config.GAN_LOSS != "hinge")

        if len(config.GPU) > 1:
            self.generator = nn.DataParallel(self.generator, config.GPU)
            self.grad_discriminator = nn.DataParallel(self.grad_discriminator, config.GPU)
            self.sr_discriminator = nn.DataParallel(self.sr_discriminator, config.GPU)

        self.L1_loss = nn.L1Loss()
        self.content_loss = SimpleContentLoss()
        self.style_loss = StyleLoss()
        self.adversarial_loss = AdversarialLoss(type = config.GAN_LOSS)
        self.get_grad = Get_gradient()

        self.add_module('generator', self.generator)
        self.add_module('grad_discriminator', self.grad_discriminator)
        self.add_module('sr_discriminator', self.sr_discriminator)

        self.add_module("L1_loss", self.L1_loss)
        self.add_module("content_loss", self.content_loss)
        self.add_module("style_loss", self.style_loss)
        self.add_module("adversarial_loss", self.adversarial_loss)

        self.add_module("get_grad", self.get_grad)

        self.gen_optimizer = optim.Adam(
            params = self.generator.parameters(),
            lr = float(config.LR),
            betas = (config.BETA1, config.BETA2)
            )

        self.dis_grad_optimizer = optim.Adam(
            params = self.grad_discriminator.parameters(),
            lr = float(config.LR),
            betas = (config.BETA1, config.BETA2)
            )

        self.dis_sr_optimizer = optim.Adam(
            params = self.sr_discriminator.parameters(),
            lr = float(config.LR),
            betas = (config.BETA1, config.BETA2)
            )

    def forward(self, lr_images):
        interp_hr_images = F.interpolate(lr_images, scale_factor = self.config.SCALE).detach()
        lr_grads = self.get_grad(lr_images).detach()
        interp_hr_grads = F.interpolate(lr_grads, scale_factor = self.config.SCALE).detach()

        outputs, gen_grads = self.generator.forward(interp_hr_images, interp_hr_grads)
        sr_grads = self.get_grad(outputs)
        
        return outputs, sr_grads, gen_grads

    def backward(self, gen_loss, dis_grad_loss, dis_sr_loss):
        dis_grad_loss.backward()
        self.dis_grad_optimizer.step()

        dis_sr_loss.backward()
        self.dis_sr_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()

    def process(self, lr_images, hr_images):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_grad_optimizer.zero_grad()
        self.dis_sr_optimizer.zero_grad()

        # process outputs
        outputs, sr_grads, gen_grads = self.forward(lr_images)
        gen_loss = 0
        dis_grad_loss = 0
        dis_sr_loss = 0

        #grad_discriminator loss
        hr_grads = self.get_grad(hr_images).detach()
        dis_grad_real = hr_grads
        dis_grad_gen = gen_grads.detach()
        dis_grad_sr = sr_grads.detach()

        dis_grad_real_outputs = self.grad_discriminator.forward(dis_grad_real)                    
        dis_grad_sr_outputs = self.grad_discriminator.forward(dis_grad_sr)

        dis_real_grad_loss = self.adversarial_loss(dis_grad_real_outputs, True, True)
        dis_fake_grad_loss = self.adversarial_loss(dis_grad_sr_outputs, False, True)
        dis_grad_loss = (dis_real_grad_loss + dis_fake_grad_loss) / 2

        #sr_discriminator loss
        dis_sr_real = hr_images
        dis_sr_gen = outputs.detach()

        dis_sr_real_outputs = self.sr_discriminator.forward(dis_sr_real)
        dis_sr_gen_outputs = self.sr_discriminator.forward(dis_sr_gen)

        dis_real_sr_loss = self.adversarial_loss(dis_sr_real_outputs, True, True)
        dis_fake_sr_loss = self.adversarial_loss(dis_sr_gen_outputs, False, True)
        dis_sr_loss = (dis_real_sr_loss + dis_fake_sr_loss) /2



        # generator adversarial loss
        sr_grads_from_dis = self.grad_discriminator.forward(sr_grads)
        grad_sr_adv_loss = self.adversarial_loss(sr_grads_from_dis, True, False)*self.config.GRAD_SR_ADV_LOSS_WEIGHT

        sr_from_dis = self.sr_discriminator.forward(outputs)
        sr_adv_loss = self.adversarial_loss(sr_from_dis, True, False)*self.config.SR_ADV_LOSS_WEIGHT
        gen_adv_loss = grad_sr_adv_loss + sr_adv_loss
        gen_loss = gen_loss + gen_adv_loss

        # generator L1 loss
        L1_grad_gen_pix_loss = self.L1_loss(hr_grads, gen_grads)*self.config.L1_GRAD_GEN_PIX_LOSS_WEIGHT
        L1_grad_sr_pix_loss = self.L1_loss(hr_grads, sr_grads)*self.config.L1_GRAD_SR_PIX_LOSS_WEIGHT
        L1_sr_pix_loss = self.L1_loss(hr_images, outputs)*self.config.L1_SR_PIX_LOSS_WEIGHT
        gen_L1_loss = L1_grad_gen_pix_loss + L1_grad_sr_pix_loss + L1_sr_pix_loss
        gen_loss = gen_loss + gen_L1_loss


        # generator content loss
        gen_content_loss = self.content_loss(outputs, hr_images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT

        # generator style loss
        #gen_style_loss = self.style_loss(outputs, hr_images)
        #gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT

        gen_perceptual_loss = gen_content_loss
        #+ gen_style_loss

        gen_loss = gen_loss + gen_perceptual_loss

        # create logs
        logs = [
            ("l_dis_grad", dis_grad_loss.item()),
            ("l_dis_sr", dis_sr_loss.item()),
            ("l_gen_adv", gen_adv_loss.item()),
            ("l_gen_l1", gen_L1_loss.item()),
            ("l_gen_perceptual", gen_perceptual_loss.item())
            ]

        return outputs, gen_loss, dis_grad_loss, dis_sr_loss, logs
