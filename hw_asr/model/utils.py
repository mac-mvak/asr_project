import torch


def _conv_shape_transform(L_in, kernel_size,
                           padding=0, dilation=1, stride=1, **batch):
    ans = torch.floor((L_in + 2 * padding - dilation * (kernel_size - 1)-1)/stride + 1).int()
    return ans





