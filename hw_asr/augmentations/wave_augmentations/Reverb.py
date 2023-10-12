import random
import torchaudio_augmentations
import torch
import torch.nn.functional as F
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class Reverb(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.reverb = torchaudio_augmentations.Reverb(**kwargs)

    def __call__(self, data: Tensor):
        return self.reverb(data)
