# Code copied from:
# https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49

import torch
import torchvision


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        features = torchvision.models.vgg16(pretrained=True).features
        blocks = []
        blocks.append(features[:4].eval())
        blocks.append(features[4:9].eval())
        blocks.append(features[9:16].eval())
        blocks.append(features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x, y, feature_layers=[], style_layers=[0, 1, 2, 3]):
        if x.shape != y.shape:
            raise ValueError(
                f"Input and target have different shapes: {x.shape} != {y.shape}"
            )

        total_pixels = x.shape[-2] * x.shape[-1]

        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] != 3:
            y = y.repeat(1, 3, 1, 1)

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        loss = 0.0
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            # "All layers used for Gram matrix computation are post-activated with ReLU to better incorporate non-linearity"
            x = torch.nn.functional.relu(x)
            y = torch.nn.functional.relu(y)
            if i in feature_layers:
                loss += torch.nn.functional.mse_loss(x, y) ** 2
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.mse_loss(gram_x, gram_y) ** 2

        # the larger the image, the larger the loss
        # normalise it by dividing by the number of pixels
        # this normalisation isn't perfect, loss goes down with higher resolutions
        # but it's good enough
        return loss / total_pixels
