
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import SRModel
from .metrics import PSNR, EdgeAccuracy
from .utils import Progbar, create_dir, stitch_images, imsave

class EdgeMatch():
    def __init__(self, config):
        self.config = config

        self.model_name = "spsr"

        self.debug = False
        self.sr_model = SRModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)

        self.train_dataset = Dataset(list_folder = config.LIST_FOLDER, 
                                     mode = "train", 
                                     sigma = config.SIGMA, 
                                     crop_size = config.CROP_SIZE, 
                                     scale = config.SCALE
                                     )
        
        self.test_dataset = Dataset(list_folder = config.LIST_FOLDER, 
                                     mode = "test", 
                                     sigma = config.SIGMA, 
                                     crop_size = config.CROP_SIZE, 
                                     scale = config.SCALE
                                     )

        self.val_dataset = Dataset(list_folder = config.LIST_FOLDER, 
                                     mode = "eval", 
                                     sigma = config.SIGMA, 
                                     crop_size = config.CROP_SIZE, 
                                     scale = config.SCALE
                                     )

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')
    
    def load(self):
        self.sr_model.load()

    def save(self):
        self.sr_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
            )
        
        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check relevant path in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.sr_model.train()

                lr_images, hr_images = self.cuda(*items)

                # sr model
                hr_images_pred, gen_loss, dis_grad_loss, dis_sr_loss, logs = self.sr_model.process(lr_images, hr_images)

                # metrics
                psnr = self.psnr(self.postprocess(hr_images), self.postprocess(hr_images_pred))
                mae = (torch.sum(torch.abs(hr_images - hr_images_pred)) / torch.sum(hr_images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

                # backward
                self.sr_model.backward(gen_loss, dis_grad_loss, dis_sr_loss)
                iteration = self.sr_model.iteration

                if iteration > max_iteration:
                    keep_training = False
                    print('Maximum number of iterations reached!')
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(hr_images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('\nEnd training....')
    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.sr_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['iter'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            lr_images, hr_images = self.cuda(*items)

            hr_images_pred, gen_loss, dis_grad_loss, dis_sr_loss, logs = self.sr_model.process(lr_images, hr_images)

            # metrics
            psnr = self.psnr(self.postprocess(hr_images), self.postprocess(hr_images_pred))
            mae = (torch.sum(torch.abs(hr_images - hr_images_pred)) / torch.sum(hr_images)).float()
            logs.append(('psnr', psnr.item()))
            logs.append(('mae', mae.item()))

            logs = [("iter", iteration), ] + logs
            progbar.add(len(hr_images), values=logs)

    def test(self):
        self.sr_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        with torch.no_grad():
            for items in test_loader:
                name = self.test_dataset.load_name(index)
                lr_images, hr_images = self.cuda(*items)
                index += 1
            
                print(index, name)
                outputs, sr_grads, gen_grads = self.sr_model.forward(lr_images)

                output = self.postprocess(outputs)[0]
                path = os.path.join(self.results_path, name)
            
                imsave(output, path)

        print('\nEnd test....')

    def sample(self):
        if len(self.val_dataset) == 0:
            return

        self.sr_model.eval()

        model = self.config.MODEL
        items = next(self.val_dataset.create_iterator(self.config.SAMPLE_SIZE))
        lr_images, hr_images = self.cuda(*items)

        iteration = self.sr_model.iteration
        outputs, sr_grads, gen_grads = self.sr_model.forward(lr_images)

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(self.scale(lr_images)),
            self.postprocess(hr_images),
            self.postprocess(gen_grads),
            self.postprocess(sr_grads),
            self.postprocess(outputs),
            img_per_row=image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def scale(self, tensor):
        return F.interpolate(tensor, scale_factor = self.config.SCALE)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
