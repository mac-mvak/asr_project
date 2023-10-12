import random
import torchaudio
import librosa
import random

import torch_audiomentations

from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class AddNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        
        filename = librosa.ex('pistachio')
        self.noise, _ = torchaudio.load(filename)
        self.snr = Tensor([kwargs['snr']])
        self.p = kwargs['p']

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            return torchaudio.functional.add_noise(data[:,:self.noise.shape[1]],
                                                    self.noise[:, :data.shape[1]], self.snr)
        else:
            return data

