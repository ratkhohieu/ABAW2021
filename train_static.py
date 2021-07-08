import time

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from models import *
import warnings
import argparse
from utils import *
from data_loader import *
from metrics import *


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--batch_image', default=64, type=int, help='batch image')
parser.add_argument('--num_epochs', default=10, type=int, help='number epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'


def main():
    seed_everything()
    train_dataset = Aff2_Dataset_static(root='../data/labels_save/expression/Train_Set/', transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              shuffle=True,
                              drop_last=False)
    # import pdb; pdb.set_trace()
    valid_dataset = Aff2_Dataset_static(root='../data/labels_save/expression/Validation_Set/', transform=test_transform)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True,
                              drop_last=False)

    model = resnet50(num_classes=7, pretrained=True)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model.to(device)

    criterion1 = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_time = time.time()
    scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-4, last_epoch=-1)

    optimizer.zero_grad()
    optimizer.step()
    best_scores = 0

    with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
        for epoch in t:
            t.set_description('Epoch %i' % epoch)
            cost_list = 0
            scheduler_steplr.step(epoch)

            model.train()

            for batch_idx, samples in tqdm(enumerate(train_loader), total=len(train_loader)):
                images = samples['images'].to(device).float().squeeze(dim=0)
                labels_cat = samples['labels'].to(device).long().squeeze(dim=0)

                for i in range(0, images.size(0), args.batch_image):
                    if i+args.batch_image <= images.size(0):
                        # import pdb; pdb.set_trace()
                        batch_images = images[i:i+args.batch_image]
                        batch_labels = labels_cat[i:i+args.batch_image]
                    else:
                        batch_images = images[i:]
                        batch_labels = labels_cat[i:]

                    assert len(batch_labels) == len(batch_images)
                    # import pdb; pdb.set_trace()
                    pred_cat = model(batch_images)

                    loss = criterion1(pred_cat, batch_labels)

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
                    images = samples['images'].to(device).float().squeeze(dim=0)
                    labels_cat = samples['labels'].to(device).long().squeeze(dim=0)

                    for i in range(0, images.size(0), args.batch_image):
                        if i + 5 <= images.size(0):
                            # import pdb; pdb.set_trace()
                            batch_images = images[i:i + args.batch_image]
                            batch_labels = labels_cat[i:i + args.batch_image]
                        else:
                            batch_images = images[i:]
                            batch_labels = labels_cat[i:]

                        assert len(batch_labels) == len(batch_images)
                        # import pdb; pdb.set_trace()
                        pred_cat = model(batch_images)
                        pred_cat = F.softmax(pred_cat)
                        pred_cat = torch.argmax(pred_cat, dim=1)

                        cat_preds.append(pred_cat.detach().cpu().numpy())
                        cat_labels.append(batch_labels.detach().cpu().numpy())

                cat_preds = np.concatenate(cat_preds, axis=0)
                cat_labels = np.concatenate(cat_labels, axis=0)
                f1, acc, total = EXPR_metric(cat_preds, cat_labels)
                print(f'f1 = {f1} \n'
                      f'acc = {acc} \n'
                      f'total = {total} \n')

if __name__ == '__main__':
    main()
