import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import cv2
import os


def creat_balance_data_ex():
    list_csv_ex = glob.glob('../data/labels_save/expression/Train_Set/*')

    df = pd.DataFrame()
    for i in tqdm(list_csv_ex):
        df = pd.concat((df, pd.read_csv(i, index_col=0)), axis=0).reset_index(drop=True)
    # df2 = pd.read_csv('../data/labels_save/expression/train_7emotion.csv')
    # df2.face_files = df2.face_files.apply(lambda x: '../data/' + x)
    # categories = {'Neutral': 0, 'Angry': 1, 'Disgust': 2, 'Fear': 3, 'Happy': 4, 'Sad': 5, 'Surprise': 6}
    # df2.emotion = df2.emotion.apply(lambda x: categories[x])
    #
    # df2 = df2.drop(columns=['video_name', 'frame_files'])
    # df2 = df2.rename(columns={'face_files': 'image_id', 'emotion': 'labels_ex'})
    #
    # df3 = pd.concat([df, df2], axis=0)
    df3 = df
    # TODO maybe change the rate of sample
    weights = df3.labels_ex.value_counts().sort_values()[0] / df3.labels_ex.value_counts().sort_values()

    df4 = pd.DataFrame()
    for k, v in dict(weights).items():
        df4 = pd.concat([df4, df3[df3['labels_ex'] == k].sample(frac=v, random_state=1, replace=True)],
                        axis=0).reset_index(drop=True)

    return df4


def create_test_df(path_txt, out_path):
    au = open(path_txt, 'r')
    lines = au.readlines()
    au_video = []
    for line in lines:
        au_video.append(line.strip())

    k = []
    v = []
    for video_dir in list(au_video):
        k.append(video_dir)
        if 'left' in video_dir:
            video_dir = video_dir.replace('_left', '')
        elif 'right' in video_dir:
            video_dir = video_dir.replace('_right', '')
        if os.path.exists('../all_videos/' + video_dir + '.mp4'):
            path = '../all_videos/' + video_dir + '.mp4'
        else:
            path = '../all_videos/' + video_dir + '.avi'

        v_cap = cv2.VideoCapture(path)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v.append(v_len)

    df1 = pd.DataFrame(columns=['name_videos'], data=k)
    df1['len_images'] = v

    au_video = ['../data/cropped_aligned/' + x for x in au_video]
    len_images = []
    for i in au_video:
        a = glob.glob(i + '/*')
        len_images.append(len(a))

    df = pd.DataFrame(columns=['folder_dir'], data=au_video)
    df['len_images'] = len_images
    df['name_videos'] = df.folder_dir.apply(lambda x: x.replace('../data/cropped_aligned/', ''))

    df3 = df.merge(df1, how='outer', on='name_videos')
    df3 = df3.rename(columns={'len_images_x': 'num_images', 'len_images_y': 'num_frames'})

    df3.to_csv(out_path)