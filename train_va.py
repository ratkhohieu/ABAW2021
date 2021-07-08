import argparse
import warnings

from torch.nn import functional as F
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
    train_dataset = Aff2_Dataset_static_shuffle(root='../data/labels_save/valence_arousal/Train_Set/',
                                                transform=train_transform, type_partition='va')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True,
                              drop_last=False)

    valid_dataset = Aff2_Dataset_static_shuffle(root='../data/labels_save/valence_arousal/Validation_Set/',
                                                transform=test_transform, type_partition='va')
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=False,
                              drop_last=False)

    model = Baseline(num_classes=2)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model.to(device)
    criterion1 = CCCLoss()
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

            for batch_idx, samples in tqdm(enumerate(train_loader), total=len(train_loader)):
                images = samples['images'].to(device).float()
                labels_cat = samples['labels'].to(device).float()
                # import pdb; pdb.set_trace()
                pred_cat = model(images)

                loss = criterion1(pred_cat, labels_cat)
                cost_list += loss.item()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                t.set_postfix(Loss=f'{cost_list / (batch_idx + 1):04f}',
                              Batch=f'{batch_idx + 1:03d}/{len(train_loader):03d}',
                              Lr=optimizer.param_groups[0]['lr'])

            model.eval()
            with torch.no_grad():
                cat_preds = []
                cat_labels = []
                for batch_idx, samples in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid_mode'):
                    images = samples['images'].to(device).float()
                    labels_cat = samples['labels'].to(device).float()

                    pred_cat = model(images)
                    pred_cat = F.tanh(pred_cat)
                    # pred_cat = torch.argmax(pred_cat, dim=1)

                    cat_preds.append(pred_cat.detach().cpu().numpy())
                    cat_labels.append(labels_cat.detach().cpu().numpy())

                cat_preds = np.concatenate(cat_preds, axis=0)
                cat_labels = np.concatenate(cat_labels, axis=0)
                # cat_preds = (cat_preds*2-1).astype(float)

                items, total = VA_metric(cat_preds, cat_labels)
                print(f'items = {items} \n'
                      f'total = {total} \n')


if __name__ == '__main__':
    main()
