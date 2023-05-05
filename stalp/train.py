import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from stalp import STALPNet
from stalp.lib.vgg_perceptual_loss import VGGPerceptualLoss

IMAGE_EXT = tuple(f".{ext}" for ext in "png jpg jpeg bmp".split())


def get_image_paths(dir_path: str):
    paths = []
    for entry in os.scandir(dir_path):
        if not entry.is_file():
            continue
        name = entry.name.lower()
        if any(name.endswith(ext) for ext in IMAGE_EXT):
            paths.append(entry.path)

    return paths


class KeyframesDataset(Dataset):
    def __init__(self, paired_paths: list[tuple[str, str]], transform) -> None:
        super().__init__()
        self.pairs = paired_paths
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair_paths = self.pairs[index]
        x = Image.open(pair_paths[0]).convert("RGB")
        y = Image.open(pair_paths[1]).convert("RGB")
        return (self.transform(x), self.transform(y))


class UnpairedDataset(Dataset):
    def __init__(self, unpaired_paths: list[str], transform) -> None:
        super().__init__()
        self.paths = unpaired_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


# mean and std of ImageNet to use pre-trained VGG
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_ebsynth_dataset(video_frames_dir: str, keyframes_dir: str):
    keyframes: list[tuple[str, str]] = []
    unpaired: list[str] = []

    for entry in os.scandir(video_frames_dir):
        # skip if non-file
        if not entry.is_file():
            continue

        # skip if non-image
        name = entry.name.lower()
        if not any(name.endswith(ext) for ext in IMAGE_EXT):
            continue

        # check if image is in keyframes folder
        stylised_path = os.path.join(keyframes_dir, name)
        if os.path.exists(stylised_path):
            # image is indeed in keyframes, this is a keyframe pair
            keyframes.append((entry.path, stylised_path))
            # also add to unpaired
            unpaired.append(entry.path)
        else:
            # not in keyframes, this is unpaired
            unpaired.append(entry.path)

    tr = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    key_dataset = KeyframesDataset(keyframes, tr)
    frm_dataset = UnpairedDataset(unpaired, tr)

    if len(key_dataset) == 0:
        raise FileNotFoundError("No keyframes found")
    if len(frm_dataset) == 0:
        raise FileNotFoundError("No unpaired images found")

    return key_dataset, frm_dataset


def train(
    net: STALPNet,
    key_dataset: KeyframesDataset,
    frm_dataset: UnpairedDataset,
    lr: float,
    epochs: int,
    device,
):
    l1_loss = nn.L1Loss().to(device)
    vgg_loss = VGGPerceptualLoss().to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, betas=(0.9, 0.999))

    loss_logs = {
        "l1_loss": [],
        "vgg_loss": [],
    }

    key_loader = DataLoader(key_dataset)
    frm_loader = DataLoader(frm_dataset)

    # constants for calculating the loss
    vgg_layers_count = 4
    frm_count = len(frm_dataset)

    for epoch in range(1, epochs + 1):
        keyframe_loss = 0.0
        unpaired_loss = 0.0
        for x, y in iter(key_loader):
            y_hat = net(x)
            keyframe_loss += l1_loss(y_hat, y)
            for frame in iter(frm_loader):
                frame_hat = net(frame)
                unpaired_loss += vgg_loss(frame_hat, y)
        unpaired_loss *= 100 / (frm_count * vgg_layers_count)
        total_loss = keyframe_loss + unpaired_loss

        print(f"{epoch}\tloss: {total_loss}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
