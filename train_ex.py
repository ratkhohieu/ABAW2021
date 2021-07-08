import argparse
import warnings

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import trange

from data_loader import *
from metrics import *
from models import *
from utils import *

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_epochs', default=12, type=int, help='number epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'


def main():
    seed_everything()
    list_csv = glob.glob('../data/labels_save/expression/Train_Set/' + '*')
    # self.list_csv = [i for i in self.list_csv if len(pd.read_csv(i)) != 0]
    df = pd.DataFrame()
    for i in tqdm(list_csv, total=len(list_csv)):
        df = pd.concat((df, pd.read_csv(i)), axis=0).reset_index(drop=True)

    # df = creat_balance_data_ex()

    df_train, df_valid = split_data(df_data1=df, df_data2=None, number_fold=5)

    for fold in tqdm(range(len(df_train)), total=len(df_train)):


        train_dataset = Aff2_Dataset_static_shuffle(df=df_train[fold], root=False,
                                                    transform=train_transform, type_partition='ex')
        # valid_dataset = Aff2_Dataset_static_shuffle(root='../data/labels_save/expression/Validation_Set/',
        #                                             transform=test_transform, type_partition='ex')
        # train_dataset = Aff2_Dataset_static_shuffle(root=False, df=df,
        #                                             transform=train_transform, type_partition='ex')
        #
        # train_dataset_affect = AFFECTNET_Dataset(root='../../data_emotion/',
        #                                          df_data='../data/labels_save/expression/affectnet_train.csv',
        #                                          transforms=train_transform)

        valid_dataset = Aff2_Dataset_static_shuffle(root='../data/labels_save/expression/Validation_Set/',
                                                    transform=test_transform, type_partition='ex')

        # train_dataset = ConcatDataset([train_dataset_affect, train_dataset_aff2])

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=True,
                                  drop_last=False)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=False,
                                  drop_last=False)

        model = Baseline(num_classes=7)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(device)
        criterion1 = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-4, last_epoch=-1)

        optimizer.zero_grad()
        optimizer.step()
        best_scores = 0
        # for name, p in model.named_parameters():
        #     if name == 'module.fc.weight' or name == 'module.fc.bias':
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False

        with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
            for epoch in t:
                # if epoch == 0:
                #     unfreeze(model.module.backbone, percent=0.2)
                # else:
                #     unfreeze(model.module.backbone, percent=1)
                t.set_description('Epoch %i' % epoch)
                cost_list = 0
                scheduler_steplr.step(epoch)

                model.train()
                cat_preds = []
                cat_labels = []
                for batch_idx, samples in tqdm(enumerate(train_loader), total=len(train_loader)):
                    images = samples['images'].to(device).float()
                    labels_cat = samples['labels'].to(device).long()
                    # import pdb; pdb.set_trace()
                    pred_cat = model(images)

                    loss = criterion1(pred_cat, labels_cat)
                    cost_list += loss.item()
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    pred_cat = F.softmax(pred_cat)
                    pred_cat = torch.argmax(pred_cat, dim=1)

                    cat_preds.append(pred_cat.detach().cpu().numpy())
                    cat_labels.append(labels_cat.detach().cpu().numpy())

                    t.set_postfix(Loss=f'{cost_list / (batch_idx + 1):04f}',
                                  Batch=f'{batch_idx + 1:03d}/{len(train_loader):03d}',
                                  Lr=optimizer.param_groups[0]['lr'])

                cat_preds = np.concatenate(cat_preds, axis=0)
                cat_labels = np.concatenate(cat_labels, axis=0)
                f1, acc, total = EXPR_metric(cat_preds, cat_labels)
                print(f'f1 = {f1} \n'
                      f'acc = {acc} \n'
                      f'total = {total} \n')

                model.eval()
                with torch.no_grad():
                    cat_preds = []
                    cat_labels = []
                    pred_fold_cat = []
                    for batch_idx, samples in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid_mode'):
                        images = samples['images'].to(device).float()
                        labels_cat = samples['labels'].to(device).long()

                        pred_cat = model(images)
                        pred_cat = F.softmax(pred_cat)
                        pred_fold_cat.append(pred_cat.detach().cpu())
                        pred_cat = torch.argmax(pred_cat, dim=1)

                        cat_preds.append(pred_cat.detach().cpu().numpy())
                        cat_labels.append(labels_cat.detach().cpu().numpy())

                    cat_preds = np.concatenate(cat_preds, axis=0)
                    cat_labels = np.concatenate(cat_labels, axis=0)
                    pred_fold_cat = np.concatenate(pred_fold_cat, axis=0)

                    f1, acc, total = EXPR_metric(cat_preds, cat_labels)
                    print(f'f1 = {f1} \n'
                          f'acc = {acc} \n'
                          f'total = {total} \n')
                if best_scores < total:
                    os.makedirs('./results/ex', exist_ok=True)
                    np.save(f'./results/ex/{fold}.npy', pred_fold_cat)

    cat_preds = ensemble_ex('./results/ex')
    f1, acc, total = EXPR_metric(cat_preds, cat_labels)

    print(f'f1 = {f1} \n'
          f'acc = {acc} \n'
          f'total = {total} \n')


if __name__ == '__main__':
    main()
