import argparse
import warnings

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import trange

from data_loader import *
from metrics import *
from models.models_abaw import *
from utils import *
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_epochs', default=10, type=int, help='number epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'


# def train(model, optimizer, criterion1, criterion2, train_loader_ex, train_loader_au, train_loader_all):
#     cost_list = 0
#     model.train()
#     # pid = os.getpid()
#     for batch_idx, samples in tqdm(enumerate(train_loader_ex), total=len(train_loader_ex)):
#         images = samples['images'].to(device).float()
#         labels_cat = samples['labels'].to(device).long()
#         # import pdb; pdb.set_trace()
#         pred_cat, _ = model(images)
#
#         loss = criterion1(pred_cat, labels_cat)
#         cost_list += loss.item()
#         optimizer.zero_grad()
#
#         loss.backward()
#         optimizer.step()
#
#     cost_list = 0
#     for batch_idx, samples in tqdm(enumerate(train_loader_au), total=len(train_loader_au)):
#         images = samples['images'].to(device).float()
#         labels_cat = samples['labels'].to(device).float()
#         # import pdb; pdb.set_trace()
#         _, pred_cat = model(images)
#
#         loss = criterion2(pred_cat, labels_cat)
#         cost_list += loss.item()
#         optimizer.zero_grad()
#
#         loss.backward()
#         optimizer.step()
#
#     cost_list = 0
#     for batch_idx, samples in tqdm(enumerate(train_loader_all), total=len(train_loader_all)):
#         images = samples['images'].to(device).float()
#         labels_ex = samples['labels_ex'].to(device).long()
#         labels_au = samples['labels_au'].to(device).float()
#         # import pdb; pdb.set_trace()
#         pred_ex, pred_au = model(images)
#
#         loss = criterion1(pred_ex, labels_ex) + criterion2(pred_au, labels_au)
#         cost_list += loss.item()
#         optimizer.zero_grad()
#
#         loss.backward()
#         optimizer.step()



def train(rank, model, train_loader_ex, criterion1):
    torch.manual_seed(123+ rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.num_epochs + 1):
        train_epoch(model, train_loader_ex, optimizer, criterion1)



def train_epoch(model, train_loader_ex, optimizer, criterion1):
    model.train()
    cost_list=0
    for batch_idx, samples in tqdm(enumerate(train_loader_ex), total=len(train_loader_ex)):
        images = samples['images'].to(device).float()
        labels_cat = samples['labels'].to(device).long()
        # import pdb; pdb.set_trace()
        pred_cat, _ = model(images)

        loss = criterion1(pred_cat, labels_cat)
        cost_list += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()





def main():
    seed_everything()

    train_dataset_all = Aff2_Dataset_static_multitask(df=pd.read_csv('../data/labels_save/multitask/inner_ex_au.csv'),
                                                      transform=train_transform, type_partition='2type')
    train_loader_all = DataLoader(dataset=train_dataset_all,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=True,
                                  drop_last=False)

    train_dataset_ex = Aff2_Dataset_static_shuffle(root='../data/labels_save/expression/Train_Set/',
                                                   transform=train_transform, type_partition='ex')
    train_loader_ex = DataLoader(dataset=train_dataset_ex,
                                 batch_size=args.batch_size,
                                 num_workers=4,
                                 shuffle=True,
                                 drop_last=False)

    train_dataset_au = Aff2_Dataset_static_shuffle(root='../data/labels_save/action_unit/Train_Set/',
                                                   transform=train_transform, type_partition='au')
    train_loader_au = DataLoader(dataset=train_dataset_au,
                                 batch_size=args.batch_size,
                                 num_workers=4,
                                 shuffle=True,
                                 drop_last=False)

    # import pdb; pdb.set_trace()
    valid_dataset_au = Aff2_Dataset_static_shuffle(root='../data/labels_save/action_unit/Validation_Set/',
                                                   transform=test_transform, type_partition='au')
    valid_dataset_ex = Aff2_Dataset_static_shuffle(root='../data/labels_save/expression/Validation_Set/',
                                                   transform=test_transform, type_partition='ex')

    valid_loader_au = DataLoader(dataset=valid_dataset_au,
                                 batch_size=args.batch_size,
                                 num_workers=4,
                                 shuffle=False,
                                 drop_last=False)

    valid_loader_ex = DataLoader(dataset=valid_dataset_ex,
                                 batch_size=args.batch_size,
                                 num_workers=4,
                                 shuffle=False,
                                 drop_last=False)

    model = Multitask(num_classes_ex=7, num_classes_au=12)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    model.share_memory()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-4, last_epoch=-1)

    optimizer.zero_grad()
    optimizer.step()
    best_scores_ex = 0
    best_scores_au = 0
    # for name, p in model.named_parameters():
    #     if name == 'fc.weight' or name == 'fc.bias':
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False
    processes = []
    for rank in range(4):
        p = mp.Process(target=train, args=(rank, model, train_loader_ex, criterion1))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


    # with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
    #     for epoch in t:
    #
    #         t.set_description('Epoch %i' % epoch)
    #         scheduler_steplr.step(epoch)
    #
    #         # mp.set_start_method('spawn')
    #
    #
    #         model.eval()
    #         with torch.no_grad():
    #             cat_preds = []
    #             cat_labels = []
    #             for batch_idx, samples in tqdm(enumerate(valid_loader_au), total=len(valid_loader_au),
    #                                            desc='Valid_mode'):
    #                 images = samples['images'].to(device).float()
    #                 labels_cat = samples['labels'].to(device).float()
    #
    #                 _, pred_cat = model(images)
    #                 pred_cat = F.sigmoid(pred_cat)
    #
    #                 cat_preds.append(pred_cat.detach().cpu().numpy())
    #                 cat_labels.append(labels_cat.detach().cpu().numpy())
    #
    #             cat_preds = np.concatenate(cat_preds, axis=0)
    #             cat_labels = np.concatenate(cat_labels, axis=0)
    #             cat_preds = (cat_preds > 0.5).astype(int)
    #             # import pdb; pdb.set_trace();
    #             f1, acc, total = AU_metric(cat_preds, cat_labels)
    #             print(f'f1_au = {f1} \n'
    #                   f'acc_au = {acc} \n'
    #                   f'total_au = {total} \n')
    #             if best_scores_au < total:
    #                 best_scores_au = total
    #                 os.makedirs('./weight', exist_ok=True)
    #                 torch.save(model, f'./weight/multitask_best_au_2.pth')
    #
    #             cat_preds = []
    #             cat_labels = []
    #             for batch_idx, samples in tqdm(enumerate(valid_loader_ex), total=len(valid_loader_ex),
    #                                            desc='Valid_mode'):
    #                 images = samples['images'].to(device).float()
    #                 labels_cat = samples['labels'].to(device).long()
    #
    #                 pred_cat, _ = model(images)
    #                 pred_cat = F.softmax(pred_cat)
    #                 pred_cat = torch.argmax(pred_cat, dim=1)
    #
    #                 cat_preds.append(pred_cat.detach().cpu().numpy())
    #                 cat_labels.append(labels_cat.detach().cpu().numpy())
    #
    #             cat_preds = np.concatenate(cat_preds, axis=0)
    #             cat_labels = np.concatenate(cat_labels, axis=0)
    #
    #             f1, acc, total = EXPR_metric(cat_preds, cat_labels)
    #             print(f'f1_ex = {f1} \n'
    #                   f'acc_ex = {acc} \n'
    #                   f'total_ex = {total} \n')
    #
    #             if best_scores_ex < total:
    #                 best_scores_ex = total
    #                 os.makedirs('./weight', exist_ok=True)
    #                 torch.save(model, f'./weight/multitask_best_ex_2.pth')


if __name__ == '__main__':
    main()
