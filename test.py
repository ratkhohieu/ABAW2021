import argparse
import warnings

from torch.utils.data import DataLoader

from data_loader import *
from models.models_abaw import *
from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')

parser.add_argument('-b', '--batch_size', default=512, type=int, help='batch size')
parser.add_argument('-p', '--partition', default='ex', type=str, help='partition')

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'


def main():
    # seed_everything()
    df_video = pd.read_csv('./weight/ex_test_len.csv')
    # list_csv = glob.glob('../data/labels_save/expression/Test_Set/*')
    # model = torch.load('./weight/multitask_best_ex_3.pth')
    model = Resnet_Multitask()
    model.load_state_dict(torch.load('../model_2anno_trainall_withFocal_BOTH_epoch_4.pth'))
    model.to(device)

    for ind, name_video in tqdm(enumerate(df_video['name_videos']), total=len(df_video)):
        df = pd.read_csv('../data/labels_save/expression/Test_Set/' + name_video + '.csv', index_col=0)
        test_dataset = Aff2_Dataset_static_multitask_test(df=df,
                                                          transform=test_transform)
        test_loader_ex = DataLoader(dataset=test_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=4,
                                    shuffle=True,
                                    drop_last=False)
        model.eval()
        with torch.no_grad():
            cat_preds = []

            for batch_idx, samples in tqdm(enumerate(test_loader_ex), total=len(test_loader_ex),
                                           desc='Test_mode'):
                images = samples['images'].to(device).float()

                pred_cat, _ = model(images)
                pred_cat = F.softmax(pred_cat)
                pred_cat = torch.argmax(pred_cat, dim=1)

                cat_preds.append(pred_cat.detach().cpu().numpy())

            cat_preds = np.concatenate(cat_preds, axis=0)

        df['result'] = cat_preds
        df['image_id'] = df.image_id.apply(lambda x: int(os.path.split(x)[1].split('.')[0]))
        df1 = pd.DataFrame(columns=['image_id'],
                           data=list(set(range(1, df_video['num_frames'][ind])).difference(set(df.image_id))))
        df1['result'] = -1
        # print(name_video)
        df2 = pd.concat([df, df1])
        df2 = df2.sort_values(by=['image_id']).reset_index(drop=True)
        path = 'results/expression/'
        os.makedirs(path, exist_ok=True)

        file = open(path + name_video + '.txt', "a")
        file.write("Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise \n")
        file.close()
        # import pdb; pdb.set_trace()
        df2 = df2.drop(columns='image_id')
        df2.to_csv(path + name_video + '.txt', header=None, index=None, sep=',', mode='a')

        # print(len((df2)))
        # break


def main1():
    # seed_everything()
    df_video = pd.read_csv('./weight/au_test_len.csv')
    # list_csv = glob.glob('../data/labels_save/expression/Test_Set/*')
    # model = torch.load('./weight/multitask_best_au_3.pth')
    model = Resnet_Multitask()
    model.load_state_dict(torch.load('../model_2anno_trainall_withoutFocal_AU_epoch_4.pth'))
    model.to(device)
    for ind, name_video in tqdm(enumerate(df_video['name_videos']), total=len(df_video)):
        df = pd.read_csv('../data/labels_save/action_unit/Test_Set/' + name_video + '.csv', index_col=0)
        test_dataset = Aff2_Dataset_static_multitask_test(df=df,
                                                          transform=test_transform)
        test_loader_au = DataLoader(dataset=test_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=4,
                                    shuffle=True,
                                    drop_last=False)
        model.eval()
        with torch.no_grad():
            cat_preds = []

            for batch_idx, samples in tqdm(enumerate(test_loader_au), total=len(test_loader_au),
                                           desc='Test_mode'):
                images = samples['images'].to(device).float()

                _, pred_cat = model(images)
                # pred_cat = F.softmax(pred_cat)
                pred_cat = torch.sigmoid(pred_cat)

                cat_preds.append(pred_cat.detach().cpu().numpy())

            cat_preds = np.concatenate(cat_preds, axis=0)
            cat_preds = (cat_preds > 0.5).astype(int)

        df[['AU' + str(i) for i in range(1, 13)]] = cat_preds.astype(int)
        df['image_id'] = df.image_id.apply(lambda x: int(os.path.split(x)[1].split('.')[0]))
        df1 = pd.DataFrame(columns=['image_id'],
                           data=list(set(range(1, df_video['num_frames'][ind])).difference(set(df.image_id))))
        df1[['AU' + str(i) for i in range(1, 13)]] = [[-1] * 12] * len(df1)
        df2 = pd.concat([df, df1])
        df2 = df2.sort_values(by=['image_id']).reset_index(drop=True)
        path = 'results/action_unit/'
        os.makedirs(path, exist_ok=True)

        file = open(path + name_video + '.txt', "a")
        file.write("AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26 \n")
        file.close()

        df2 = df2.drop(columns=['image_id', 'result'])
        df2.to_csv(path + name_video + '.txt', header=None, index=None, sep=',', mode='a')

        # print(len((df2)))


if __name__ == '__main__':
    if args.partition == 'ex':
        main()
    else:
        main1()
