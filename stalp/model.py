import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Standard convolutional layer

    Derived from:
    https://github.com/dxyang/StyleTransfer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x


class UpsampleConvLayer(nn.Module):
    """
    Convolutional layer, but it applies nearest neighbour upscaling beforehand.

    Derived from:
    https://github.com/dxyang/StyleTransfer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample_factor):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode="nearest")

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.upsample(x)
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x


class ResidualBlock(nn.Module):
    """
    Derived from:
    https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer
    """

    def __init__(self, channels: int):
        super().__init__()

        self.relu = nn.LeakyReLU()

        self.conv1 = nn.Sequential(
            ConvLayer(channels, channels, 3, 1),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.conv2 = nn.Sequential(
            ConvLayer(channels, channels, 3, 1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        input_x = x
        x = self.relu(self.conv1(x))
        # TODO: Unsure if input_x should be inside LeakyReLU or added outside it
        # i.e. `LeakyReLU(a + b)` or `LeakyReLU(a) + b`?
        x = self.relu(self.conv2(x) + input_x)
        return x


class STALPNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # a max pool for halving the resolution of tensors
        self.maxpool = nn.MaxPool2d(2)

        self.conv1 = nn.Sequential(
            ConvLayer(3, 32, 7, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(),
        )
        # downsize by 2
        self.conv2 = nn.Sequential(
            ConvLayer(32, 64, 3, 2),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(),
        )
        # downsize by 2
        self.conv3 = nn.Sequential(
            ConvLayer(64, 128, 3, 2),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(),
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # upsize by 2
        # input channels is sum of res_blocks + conv3 + conv2
        self.deconv1 = nn.Sequential(
            UpsampleConvLayer(128 + 128 + 64, 64, 3, 1, 2),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(),
        )
        # upsize by 2
        # input channels is sum of deconv1 + conv1 + input
        self.deconv2 = nn.Sequential(
            UpsampleConvLayer(64 + 32 + 3, 32, 3, 1, 2),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(),
        )
        # this is the output, limited by tanh to (-1, 1)
        self.conv4 = nn.Sequential(
            ConvLayer(32, 3, 7, 1),
            # nn.Tanh(),
        )

    def forward(self, x):
        # early input validation to avoid size mismatch issues when upsizing later
        if x.shape[-1] < 8 or x.shape[-2] < 8:
            raise ValueError("Input image dimensions must be at least 8x8")
        if x.shape[-1] % 4 != 0 or x.shape[-2] % 4 != 0:
            raise ValueError("Input image dimensions must be divisible by 4")

        input_x = x
        x = self.conv1(x)
        conv1_x = x
        x = self.conv2(x)
        conv2_x = x
        x = self.conv3(x)
        conv3_x = x
        x = self.res_blocks(x)
        # concatenated skip connection
        x = torch.cat([x, conv3_x, self.maxpool(conv2_x)], dim=1)
        x = self.deconv1(x)
        # concatenated skip connection
        x = torch.cat([x, self.maxpool(conv1_x), self.maxpool(input_x)], dim=1)
        x = self.deconv2(x)
        x = self.conv4(x)

        return x
