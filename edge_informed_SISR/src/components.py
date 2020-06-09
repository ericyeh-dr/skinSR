import torch
import torch.nn as nn

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
                nn.init.normal_(module.weight.data, 1.0, gain = gain)
                nn.init.constant_(module.bias.data, 0.0)

        self.apply(init_func)

class EdgeGenerator(BaseNet):
    def __init__(self, scale = 4, residual_blocks = 8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = 7, padding = 0)),
            nn.InstanceNorm2d(64, track_running_stats = False),
            nn.ReLU(True),

            nn.utils.spectral_norm(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(128, track_running_stats = False),
            nn.ReLU(True),

            nn.utils.spectral_norm(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(256, track_running_stats = False),
            nn.ReLU(True)
            )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(128, track_running_stats = False),
            nn.ReLU(True),

            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(64, track_running_stats = False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 7, padding = 0)
            )

        self.init_weights()
    
    def forward(self, input):
        output = self.encoder(input)
        output = self.middle(output)
        output = self.decoder(output)
        output = torch.sigmoid(output)
        return output

class GradientGenerator(BaseNet):
    def __init__(self, scale = 4, residual_blocks = 8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = 7, padding = 0)),
            nn.InstanceNorm2d(64, track_running_stats = False),
            nn.ReLU(True),

            nn.utils.spectral_norm(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(128, track_running_stats = False),
            nn.ReLU(True),

            nn.utils.spectral_norm(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(256, track_running_stats = False),
            nn.ReLU(True)
            )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(128, track_running_stats = False),
            nn.ReLU(True),

            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(64, track_running_stats = False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 7, padding = 0)
            )

        self.init_weights()
    
    def forward(self, input):
        output = self.encoder(input)
        output = self.middle(output)
        output = self.decoder(output)
        output = torch.sigmoid(output)
        return output

class SRGenerator(BaseNet):
    def __init__(self, scale = 4, residual_blocks = 8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = 7, padding = 0),
            nn.InstanceNorm2d(64, track_running_stats = False),
            nn.ReLU(True),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(128, track_running_stats = False),
            nn.ReLU(True),

            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(256, track_running_stats = False),
            nn.ReLU(True)
            )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(128, track_running_stats = False),
            nn.ReLU(True),

            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(64, track_running_stats = False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 7, padding = 0)
            )

        self.init_weights()
    
    def forward(self, input):
        output = self.encoder(input)
        output = self.middle(output)
        output = self.decoder(output)
        output = torch.sigmoid(output)
        return output

class Discriminator(BaseNet):
    def __init__(self, in_channels, use_sigmoid = True):
        super().__init__()

        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, True)
            )

        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, True)
            )

        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, True)
            )

        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 1, padding = 1, bias = False)),
            )

        self.conv5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, True)
            )

        self.init_weights()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        output = conv5
        if self.use_sigmoid:
            output = nn.sigmoid(conv5)

        return output, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation = 1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.utils.spectral_norm(nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 3, padding = 0, dilation = dilation, bias = False)),
            nn.InstanceNorm2d(dim, track_running_stats = False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 3, padding = 0, dilation = 1, bias = False)),
            nn.InstanceNorm2d(dim, track_running_stats = False)
            )

    def forward(self, input):
        output = input + self.conv_block(input)

        return output


