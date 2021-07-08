import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(nn.Linear(in_planes, in_planes // 2),
                                nn.ReLU(),
                                nn.Linear(in_planes // 2, in_planes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # import pdb ; pdb.set_trace()
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.fc(avg_out.view(avg_out.size(0), -1))
        max_out = self.fc(max_out.view(max_out.size(0), -1))
        out = (avg_out + max_out) * y
        att_weight = out / 2.0 + y * 2
        return self.sigmoid(out).unsqueeze(dim=2), att_weight


class ChannelAttentionv2(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttentionv2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Sequential(nn.Linear(in_planes, in_planes // 2),
                                 nn.ReLU(),
                                 nn.Linear(in_planes // 2, in_planes))
        self.fc2 = nn.Sequential(nn.Linear(in_planes, in_planes // 2),
                                 nn.ReLU(),
                                 nn.Linear(in_planes // 2, in_planes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # import pdb ; pdb.set_trace()
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.fc1(avg_out.view(avg_out.size(0), -1))
        max_out = self.fc2(max_out.view(max_out.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(dim=2)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SKConv(nn.Module):
    def __init__(self, features, M=2, G=2, r=16, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv1d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                nn.BatchNorm1d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Conv1d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm1d(d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv1d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()

    def forward(self, x, y):

        residual = x
        batch_size = x.shape[0]

        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2])

        feats_U = torch.sum(feats, dim=1)

        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1)
        attention_vectors = self.softmax(attention_vectors)

        y = y.unsqueeze(dim=2)
        y = y.unsqueeze(dim=1)
        attention_vectors = attention_vectors

        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return self.relu(feats_V + self.shortcut(residual)), attention_vectors
