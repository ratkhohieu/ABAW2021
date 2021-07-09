from utils import prepare_model
import torch
import torch.nn as nn

from models import *
# from utils.prepare_model import prepare_model_relation
import pickle


class Baseline(nn.Module):
    def __init__(self, num_classes):
        super(Baseline, self).__init__()
        # self.backbone = prepare_model.prepare_model_relation()
        self.backbone = models.densenet169(pretrained=True)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        out = self.backbone(x)
        return out


class Lstm(nn.Module):
    def __init__(self, n_features=2048, hidden_size=32, n_class=7, num_layers=2, drop=0.4):
        super(Lstm, self).__init__()
        self.backbone = prepare_model.prepare_model_relation()
        self.backbone.fc = nn.Identity()
        self.lstm = TemporalLSTM(n_features=n_features, hidden_size=hidden_size, n_class=n_class, num_layers=num_layers,
                                 drop=drop)
        self.n_features = n_features

    def forward(self, x):
        batch_samples = torch.Tensor().cuda()
        for i in range(x.shape[0]):
            out = self.backbone(x[0])
            batch_samples = torch.cat((batch_samples, out.unsqueeze(dim=0)))
        # import pdb; pdb.set_trace()
        out_lstm = self.lstm(batch_samples)
        return out_lstm


class Multitask(nn.Module):
    def __init__(self, num_classes_ex, num_classes_au):
        super(Multitask, self).__init__()
        # self.backbone = prepare_model.prepare_model_relation()

        self.backbone = models.resnet.resnet50(pretrained=False)
        pretrained_vggface2 = './weight/resnet50_ft_weight.pkl'
        with open(pretrained_vggface2, 'rb') as f:
            pretrained_data = pickle.load(f)
        for k, v in pretrained_data.items():
            pretrained_data[k] = torch.tensor(v)

        self.backbone.fc = nn.Identity()
        self.backbone.load_state_dict(pretrained_data, strict=False)

        self.fc1_1 = nn.Linear(in_features=2048, out_features=512)
        self.fc1_2 = nn.Linear(in_features=512, out_features=num_classes_ex)
        self.fc2_1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2_2 = nn.Linear(in_features=512, out_features=num_classes_au)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_b = self.backbone(x)

        out_ex = self.fc1_1(out_b)
        out_ex = self.fc1_2(self.dropout(self.relu(out_ex)))

        out_au = self.fc2_1(out_b)
        out_au = self.fc2_2(self.dropout(self.relu(out_au)))

        return out_ex, out_au


class Mlp_ex_au(nn.Module):
    def __init__(self):
        super(Mlp_ex_au, self).__init__()
        self.fc1 = nn.Linear(in_features=12, out_features=19)
        self.fc2 = nn.Linear(in_features=19, out_features=19)
        self.fc_ex = nn.Linear(in_features=19, out_features=7)
        self.fc_au = nn.Linear(in_features=19, out_features=12)

    def forward(self, x):
        out_b = self.fc1(x)
        out_b = self.fc2(out_b)
        out_ex = self.fc_ex(out_b)
        out_au = self.fc_au(out_b)
        return out_ex, out_au

#
# class Multitask_ex(nn.Module):
#     def __init__(self, num_classes_ex):
#         super(Multitask_ex, self).__init__()
#         self.backbone = prepare_model.prepare_model_relation()
#         self.backbone.fc = nn.Identity()
#         self.fc1 = nn.Linear(in_features=2048, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=num_classes_ex)
#
#     def forward(self, x):
#         out_b = self.backbone(x)
#         out_b = self.fc1(out_b)
#         out_ex = self.fc2(out_b)
#         return out_ex
#
#
# class Multitask_au(nn.Module):
#     def __init__(self, num_classes_au, share_fc1):
#         super(Multitask_au, self).__init__()
#         self.backbone = prepare_model.prepare_model_relation()
#         self.backbone.fc = nn.Identity()
#         self.fc1 = share_fc1
#         self.fc2 = nn.Linear(in_features=512, out_features=num_classes_au)
#
#     def forward(self, x):
#         out_b = self.backbone(x)
#         out_b = self.fc1(out_b)
#         out_au = self.fc2(out_b)
#         return out_au


class Resnet_Multitask(nn.Module):
    def __init__(self):
        super(Resnet_Multitask, self).__init__()
        self.backbone = resnet50(pretrained=False)

        self.fc_expr = nn.Sequential(
            nn.Linear(in_features=self.backbone.fc.in_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )
        self.fc_au = nn.Sequential(
            nn.Linear(in_features=self.backbone.fc.in_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 12)
        )
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        out_expr = self.fc_expr(x)
        out_au = self.fc_au(x)

        return out_expr, out_au



class Multitask_ex(nn.Module):
    def __init__(self):
        super(Multitask_ex, self).__init__()
        self.backbone = resnet50(pretrained=False)
        pretrained_vggface2 = './weight/resnet50_ft_weight.pkl'
        with open(pretrained_vggface2,'rb') as f:
            pretrained_data = pickle.load(f)
        for k,v in pretrained_data.items():
            pretrained_data[k] = torch.tensor(v)
        self.backbone.fc = nn.Identity()

        self.fc_expr = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512,7)
        )

    def forward(self, x):
        x = self.backbone(x)
        out_expr = self.fc_expr(x)

        return out_expr


class Multitask_au(nn.Module):
    def __init__(self, share_bb):
        super(Multitask_au, self).__init__()
        self.backbone = share_bb
        self.backbone.fc = nn.Identity()

        self.fc_au = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512,12)
        )
    def forward(self, x):
        x = self.backbone(x)
        out_au = self.fc_au(x)

        return out_au