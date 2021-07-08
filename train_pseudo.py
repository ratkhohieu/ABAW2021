import argparse
import warnings

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import trange

from data_loader import *
from metrics import *
from models.models_abaw import *
from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_epochs', default=10, type=int, help='number epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'


def main():
    seed_everything()

    train_dataset_all = Aff2_Dataset_pseudo(df=pd.read_csv('../data/labels_save/multitask/inner_ex_au.csv'),
                                            type_partition='2type')
    train_loader_all = DataLoader(dataset=train_dataset_all,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=True,
                                  drop_last=False)

    valid_dataset_all = Aff2_Dataset_pseudo(df=pd.read_csv('../data/labels_save/multitask/valid_inner_ex_au.csv'),
                                            type_partition='2type')

    valid_loader_all = DataLoader(dataset=valid_dataset_all,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=False,
                                  drop_last=False)

    model = Mlp_ex_au()

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model.to(device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-4, last_epoch=-1)

    optimizer.zero_grad()
    optimizer.step()
    best_scores = 0
    # for name, p in model.named_parameters():
    #     if name == 'fc.weight' or name == 'fc.bias':
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False

    with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
        for epoch in t:
            # if epoch == 0:
            #     unfreeze(model.backbone, percent=0.2)
            # else:
            #     unfreeze(model.backbone, percent=1)
            t.set_description('Epoch %i' % epoch)
            cost_list = 0
            scheduler_steplr.step(epoch)
            model.train()

            for batch_idx, samples in tqdm(enumerate(train_loader_all), total=len(train_loader_all)):
                # images = samples['images'].to(device).float()
                labels_ex = samples['labels_ex'].to(device).long()
                labels_au = samples['labels_au'].to(device).float()
                # import pdb; pdb.set_trace()
                pred_ex, pred_au = model(labels_au)

                loss = criterion1(pred_ex, labels_ex)
                cost_list += loss.item()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                t.set_postfix(Loss=f'{cost_list / (batch_idx + 1):04f}',
                              Batch=f'{batch_idx + 1:03d}/{len(train_loader_all):03d}',
                              Lr=optimizer.param_groups[0]['lr'])

            # model.eval()
            # with torch.no_grad():
            #     cat_preds = []
            #     cat_labels = []
            #     for batch_idx, samples in tqdm(enumerate(valid_loader_all), total=len(valid_loader_all),
            #                                    desc='Valid_mode'):
            #         labels_ex = samples['labels_ex'].to(device).float()
            #         labels_au = samples['labels_au'].to(device).float()
            #
            #         pred_ex, pred_au = model(labels_ex)
            #
            #         cat_preds.append(pred_au.detach().cpu().numpy())
            #         cat_labels.append(labels_au.detach().cpu().numpy())
            #
            #     cat_preds = np.concatenate(cat_preds, axis=0)
            #     cat_labels = np.concatenate(cat_labels, axis=0)
            #     cat_preds = (cat_preds > 0.5).astype(int)
            #     # import pdb; pdb.set_trace();
            #     f1, acc, total = AU_metric(cat_preds, cat_labels)
            #     print(f'f1 = {f1} \n'
            #           f'acc = {acc} \n'
            #           f'total = {total} \n')

            model.eval()
            with torch.no_grad():
                cat_preds = []
                cat_labels = []
                for batch_idx, samples in tqdm(enumerate(valid_loader_all), total=len(valid_loader_all),
                                               desc='Valid_mode'):
                    labels_ex = samples['labels_ex'].to(device).long()
                    labels_au = samples['labels_au'].to(device).float()

                    pred_ex, pred_au = model(labels_au)
                    pred_cat = F.softmax(pred_ex)
                    pred_cat = torch.argmax(pred_cat, dim=1)

                    cat_preds.append(pred_cat.detach().cpu().numpy())
                    cat_labels.append(labels_ex.detach().cpu().numpy())

                cat_preds = np.concatenate(cat_preds, axis=0)
                cat_labels = np.concatenate(cat_labels, axis=0)

                f1, acc, total = EXPR_metric(cat_preds, cat_labels)
                print(f'f1 = {f1} \n'
                      f'acc = {acc} \n'
                      f'total = {total} \n')


if __name__ == '__main__':
    main()
