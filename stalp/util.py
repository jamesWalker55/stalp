import math
from PIL import Image


def conv2d_length(length: int, kernel_length: int, padding: int = 0, stride: int = 1):
    """
    Formula described in:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """

    return math.floor((length + 2 * padding - (kernel_length - 1) - 1) / stride + 1)


def load_image(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
