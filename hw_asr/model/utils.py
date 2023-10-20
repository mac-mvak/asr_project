import torch


def _conv_shape_transform(L_in, kernel_size,
                            padding=torch.tensor([0,0]),
                            dilation=torch.tensor([1, 1]), 
                            stride=torch.tensor([1, 1]), dim=None, **batch):
    if dim is not None:
        kernel_size = kernel_size[dim]
        padding = padding[dim]
        dilation = dilation[dim]
        stride = stride[dim]
    ans = torch.floor((L_in + 2 * padding - dilation * (kernel_size - 1)-1)/stride + 1).int()
    return ans




