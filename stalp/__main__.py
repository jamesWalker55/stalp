import torch
from .model import STALPNet


def main():
    device = torch.device("cuda")
    net = STALPNet().to(device)

    # x = torch.rand(1, 3, 16, 16).to(device)
    # print(f"{x.shape = }")
    # y = net(x)
    # print(f"{y.shape = }")

    for i in range(0, 101):
        x = torch.rand(1, 3, i, i).to(device)
        try:
            y = net(x)
        except RuntimeError as e:
            print(i, e)
        except ValueError as e:
            print(i, "x", e)
        else:
            print(i, x.shape, "->", y.shape)

    # print(f"{conv2d_length(3, 3, 0, 2) = }")
    # print(f"{conv2d_length(4, 3, 0, 2) = }")
    # print(f"{conv2d_length(5, 3, 0, 2) = }")
    # print(f"{conv2d_length(6, 3, 0, 2) = }")
    # print(f"{conv2d_length(7, 3, 0, 2) = }")


if __name__ == "__main__":
    main()
