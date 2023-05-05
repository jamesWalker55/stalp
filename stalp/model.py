import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Derived from:
    https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        pad="reflect",
        activation="relu",
        normalization="instance",
    ):
        super().__init__()

        # padding
        if pad == "reflect":
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        elif pad == "zero":
            self.pad = nn.ZeroPad2d(kernel_size // 2)
        else:
            raise NotImplementedError(f"Unexpected pad flag: {pad}")

        # convolution
        self.conv_layer = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride
        )

        # normalization
        if normalization == "instance":
            self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
        else:
            raise NotImplementedError(f"Unexpected normalization flag: {normalization}")

        # activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "linear":
            self.activation = lambda x: x
        else:
            raise NotImplementedError(f"Unexpected activation flag: {activation}")

    def forward(self, x):
        x = self.pad(x)
        x = self.conv_layer(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class ResidualLayer(nn.Module):
    """
    Derived from:
    https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        pad="reflect",
        normalization="instance",
    ):
        super().__init__()

        self.conv1 = ConvLayer(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            pad,
            activation="relu",
            normalization=normalization,
        )

        self.conv2 = ConvLayer(
            out_ch,
            out_ch,
            kernel_size,
            stride,
            pad,
            activation="linear",
            normalization=normalization,
        )

    def forward(self, x):
        y = self.conv1(x)
        return self.conv2(y) + x


class DeconvLayer(nn.Module):
    """
    Derived from:
    https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        pad="reflect",
        activation="relu",
        normalization="instance",
        upsample="nearest",
    ):
        super().__init__()

        # upsample
        self.upsample = upsample

        # pad
        if pad == "reflect":
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        elif pad == "zero":
            self.pad = nn.ZeroPad2d(kernel_size // 2)
        else:
            raise NotImplementedError(f"Unexpected pad flag: {pad}")

        # conv
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)

        # normalization
        if normalization == "instance":
            self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
        else:
            raise NotImplementedError(f"Unexpected normalization flag: {normalization}")

        # activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "linear":
            self.activation = lambda x: x
        else:
            raise NotImplementedError(f"Unexpected activation flag: {activation}")

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample)
        x = self.pad(x)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class STALPNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvLayer(3, 32, 7, 1)
        # downsize by 2
        self.conv2 = ConvLayer(32, 64, 3, 2)
        # downsize by 2
        self.conv3 = ConvLayer(64, 128, 3, 2)

        self.res_blocks = nn.Sequential(
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
        )

        # upsize by 2
        self.deconv1 = DeconvLayer(128, 64, 3, 1)
        # upsize by 2
        self.deconv2 = DeconvLayer(64, 32, 3, 1)
        self.conv4 = ConvLayer(32, 3, 7, 1, activation="relu")

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
        x = self.res_blocks(x) + conv3_x
        x = self.deconv1(x) + conv2_x
        x = self.deconv2(x) + conv1_x
        x = self.conv4(x) + input_x

        return x
