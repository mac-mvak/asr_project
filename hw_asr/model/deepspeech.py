from torch import nn
from hw_asr.model.utils import _conv_shape_transform
import torch

from hw_asr.base import BaseModel


class DeepSpeech(BaseModel):
    def __init__(self, n_feats, n_class, conv_type, convs_params, grus_params,
                **batch):
        super().__init__(n_feats, n_class, **batch)
        self.conv_type = conv_type
        self.convs_params = convs_params
        self.grus_len = len(grus_params)
        self.convs = nn.Sequential()
        for conv_params in self.convs_params:
            self.convs.append(nn.Conv1d(**conv_params['convolution']))
            self.convs.append(nn.BatchNorm1d(**conv_params['batch_norm']))
            self.convs.append(nn.ReLU())
        self.grus = nn.ModuleList()
        self.bnorms = nn.ModuleList()
        for gru_params in grus_params:
            self.grus.append(nn.GRU(**gru_params['gru']))
            self.bnorms.append(nn.BatchNorm1d(**gru_params['batch_norm']))
        self.fc = nn.Linear(in_features=2*grus_params[-1]['gru']['hidden_size'], 
                            out_features=n_class)


    def forward(self, spectrogram, **batch):
        conv_spec = self.convs(spectrogram)
        out = conv_spec
        for i, tup in enumerate(zip(self.grus, self.bnorms)):
            gru, bnorm = tup
            out, _ = gru(out.transpose(1, 2))
            out = bnorm(out.transpose(1, 2))
            if i != self.grus_len - 1:
                out = nn.functional.relu(out)
        return {"logits": self.fc(out.transpose(1, 2))}

    def transform_input_lengths(self, input_lengths):
        for conv_params in self.convs_params:
            input_lengths = _conv_shape_transform(input_lengths, **conv_params['convolution'])
        return input_lengths 



