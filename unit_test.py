import argparse
import warnings

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import trange

from data_loader import *
from metrics import *
import torch.multiprocessing as mp
from multiprocessing import Pool
from models.models_abaw import *
import torchvision.models.video as vd
from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=2048, type=int, help='batch size')
parser.add_argument('--num_epochs', default=10, type=int, help='number epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'


train_dataset = Aff2_Dataset_static_multitask(df=pd.read_csv('../data/labels_save/multitask/inner_ex_au.csv'),
                                              transform=train_transform, type_partition='2type')

# import pdb; pdb.set_trace()
valid_dataset_au = Aff2_Dataset_static_shuffle(root='../data/labels_save/action_unit/Train_Set/',
                                               transform=test_transform, type_partition='au')
valid_dataset_ex = Aff2_Dataset_static_shuffle(root='../data/labels_save/expression/Train_Set/',
                                               transform=test_transform, type_partition='ex')
for i in tqdm(train_dataset):
    0
for i in tqdm(valid_dataset_ex):
    0
for i in tqdm(valid_dataset_au):
    0
# import pdb; pdb.set_trace()
#
# create_test_df('../test/test_set_AU_Challenge.txt', out_path='./weight/au_test_len.csv')
# create_test_df('../test/test_set_Expr_Challenge.txt', out_path='./weight/ex_test_len.csv')
# create_test_df('../test/test_set_VA_Challenge.txt', out_path='./weight/va_test_len.csv')
