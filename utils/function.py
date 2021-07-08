import glob
import os
import random

import numpy as np
import pandas as pd
import torch


def seed_everything(seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def unfreeze(model, percent=0.25):
    l = int(np.ceil(len(model._modules.keys()) * percent))
    l = list(model._modules.keys())[-l:]
    print(f"unfreezing these layer {l}", )
    for name in l:
        for params in model._modules[name].parameters():
            params.requires_grad_(True)


def pad_if_need(df, length):
    df = df.sort_values(by=['labels_ex', 'image_id'])
    df.labels_ex.value_counts()
    out_df = pd.DataFrame()
    for i in df.labels_ex.unique():
        df1 = df[df.labels_ex == i]
        df1 = df1.append(df1.iloc[[-1] * (length - len(df1) % length)]).reset_index(drop=True)
        out_df = pd.concat([out_df, df1], axis=0).reset_index(drop=True)
    return out_df


def ensemble_ex(root):
    list_np = glob.glob(root + '/*')
    results = []
    for i in list_np:
        result = np.load(i)
        results.append(result)
    results = np.stack(results, axis=0)
    # import pdb; pdb.set_trace()
    results = np.mean(results, axis=0)
    results = np.argmax(results, axis=1)
    return results


def to_onehot_ex(label):
    arr = torch.zeros(7)
    arr[label] = 1
    return arr


