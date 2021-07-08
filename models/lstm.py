from abc import ABC

import torch
import torch.nn as nn


class Attention(nn.Module, ABC):
    def __init__(self, feature_dim, step_dim, bias=True):
        super(Attention, self).__init__()

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class TemporalLSTM(nn.Module, ABC):
    def __init__(self, n_features=2048, hidden_size=32, n_class=7, num_layers=2, drop=0.4):
        super(TemporalLSTM, self).__init__()

        self.bi_lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, bidirectional=False, batch_first=True, dropout=drop
            , num_layers=num_layers
        )
        # self.bi_gru = nn.GRU(
        #     input_size=n_features, hidden_size=hidden_size, bidirectional=True, batch_first=True, dropout=drop
        # )
        self.lstm_attention = Attention(hidden_size * 2, 8)
        self.fc = nn.Sequential(
            nn.Linear(5, n_class),
            # nn.Linear(hidden_size , n_class)
        )

    def forward(self, x):
        self.bi_lstm.flatten_parameters()
        lstm_out, _ = self.bi_lstm(x)
        # h_lstm_atten = self.lstm_attention(lstm_out)
        #
        # avg_pool_g = torch.mean(lstm_out, 1)
        # max_pool_g, _ = torch.max(lstm_out, 1)
        #
        # pool = torch.cat([avg_pool_g, max_pool_g, h_lstm_atten], 1)
        # return pool
        # import pdb; pdb.set_trace()
        lstm_out = lstm_out.reshape(lstm_out.size(0), -1)
        return self.fc(lstm_out)

    def freeze(self):
        pass

    def unfreeze(self):
        pass
