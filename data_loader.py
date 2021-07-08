import ast

from torch.utils.data import Dataset
from torchvision import transforms

from utils import *

train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize(size=(224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      # transforms.RandomErasing(),
                                      ])

test_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ])


class Aff2_Dataset_static(Dataset):
    def __init__(self, root, transform):
        super(Aff2_Dataset_static, self).__init__()
        self.list_csv = glob.glob(root + '*')
        self.list_csv = [i for i in self.list_csv if len(pd.read_csv(i)) != 0]
        self.transform = transform

    def __getitem__(self, index):
        df = pd.read_csv(self.list_csv[index])
        images = []
        labels = np.array(df['labels_ex'])
        for image_dir in df['image_id']:
            image = cv2.imread(image_dir)[..., ::-1]
            image = self.transform(image)
            images.append(np.array(image))
        images = torch.tensor(images)
        sample = {
            'images': images,
            'labels': labels
        }
        assert images.shape[0] == labels.shape[0]
        # print(images.shape[0], labels.shape[0])
        return sample

    def __len__(self):
        return len(self.list_csv)


class Aff2_Dataset_static_shuffle(Dataset):
    def __init__(self, root, transform, type_partition, df=None):
        super(Aff2_Dataset_static_shuffle, self).__init__()
        if root:
            self.list_csv = glob.glob(root + '*')
            # self.list_csv = [i for i in self.list_csv if len(pd.read_csv(i)) != 0]
            self.df = pd.DataFrame()

            for i in tqdm(self.list_csv, total=len(self.list_csv)):
                self.df = pd.concat((self.df, pd.read_csv(i)), axis=0).reset_index(drop=True)
        else:
            self.df = df
        self.transform = transform
        self.type_partition = type_partition
        if self.type_partition == 'ex':
            self.list_labels_ex = np.array(self.df['labels_ex'].values)
        elif self.type_partition == 'au':
            self.list_labels_au = np.array(self.df['labels_au'].values)
        else:
            self.list_labels_va = np.array(self.df['labels_va'].values)

        self.list_image_id = np.array(self.df['image_id'].values)

    def __getitem__(self, index):
        if self.type_partition == 'ex':
            label = self.list_labels_ex[index]
        elif self.type_partition == 'au':
            label = ast.literal_eval(self.list_labels_au[index])
        else:
            label = ast.literal_eval(self.list_labels_va[index])
            # label = (label+1)/2.0
        image = cv2.imread(self.list_image_id[index])[..., ::-1]
        sample = {
            'images': self.transform(image),
            'labels': torch.tensor(label)
        }
        return sample

    def __len__(self):
        return len(self.df)


class Aff2_Dataset_series_shuffle(Dataset):
    def __init__(self, root, transform, type_partition, length_seq):
        super(Aff2_Dataset_series_shuffle, self).__init__()
        self.list_csv = glob.glob(root + '*')
        # import pdb; pdb.set_trace()
        self.df = pd.DataFrame()
        for i in tqdm(self.list_csv, total=len(self.list_csv)):
            self.one_df = pd.read_csv(i)
            self.one_df = pad_if_need(self.one_df, length_seq)
            self.df = pd.concat((self.df, self.one_df), axis=0).reset_index(drop=True)
        self.transform = transform
        self.type_partition = type_partition
        if self.type_partition == 'ex':
            self.list_labels_ex = np.array(self.df['labels_ex'].values)
        # elif self.type_partition == 'au':
        #     self.list_labels_au = np.array(self.df['labels_au'].values)
        # else:
        #     self.list_labels_va = np.array(self.df['labels_va'].values)

        self.list_image_id = np.array(self.df['image_id'].values)
        self.length = length_seq

    def __getitem__(self, index):
        images = []
        labels = []
        for i in range(index * self.length, index * self.length + self.length):
            if self.type_partition == 'ex':
                label = self.list_labels_ex[i]
                # elif self.type_partition == 'au':
                #     label = ast.literal_eval(self.list_labels_au[index])
                # else:
                #     label = ast.literal_eval(self.list_labels_va[index])
                #     # label = (label+1)/2.0

                image = cv2.imread(self.list_image_id[i])[..., ::-1]
                images.append(self.transform(image))
                labels.append(label)
        # import pdb; pdb.set_trace()
        images = np.stack(images)
        labels = np.array(labels).mean()

        assert labels.is_integer()
        # print(images.shape)
        # print(labels.shape)
        sample = {
            'images': torch.tensor(images),
            'labels': torch.tensor(labels)
        }
        return sample

    def __len__(self):
        return len(self.df) // self.length


class AFFECTNET_Dataset(Dataset):
    def __init__(self, df_data, transforms, root):
        if not isinstance(df_data, pd.DataFrame):
            self.df = pd.read_csv(df_data)
        else:
            self.df = df_data
        self.transforms = transforms
        self.root = root
        self.list_filepath = self.df['subDirectory_filePath'].values
        self.arousal = self.df['arousal'].values
        self.valence = self.df['valence'].values

        self.labels_ex = self.df['labels_ex'].values
        self.landmark = self.df['facial_landmarks'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.root + self.list_filepath[idx])
        image = cv2.imread(path)[..., ::-1]

        labels_conti = [(self.arousal[idx] + 10) / 20.0,
                        (self.valence[idx] + 10) / 20.0]
        labels_conti = np.asarray(labels_conti)
        expression = np.asarray(self.labels_ex[idx])

        # x = self.df.face_x[idx]
        # y = self.df.face_y[idx]
        # w = self.df.face_width[idx]
        # h = self.df.face_height[idx]

        # even = np.array(self.landmark[idx].split(';'), float)[0::2]
        # odd = np.array(self.landmark[idx].split(';'), float)[1::2]
        #
        # even = (even - x) / w
        # odd = (odd - y) / h
        # landmark = [[x, y] for [x, y] in zip(even, odd)]
        # landmark = np.concatenate(landmark, axis=0)

        sample = {'images': self.transforms(image),
                  # 'labels_conti': labels_conti,
                  # 'landmark': landmark,
                  'labels': expression}

        return sample


class Aff2_Dataset_static_multitask(Dataset):
    def __init__(self, transform, type_partition, df=None):
        super(Aff2_Dataset_static_multitask, self).__init__()
        self.df = df
        self.transform = transform
        self.type_partition = type_partition
        # TODO adjust the types of data
        if self.type_partition == 'ex':
            self.list_labels_ex = np.array(self.df['labels_ex'].values)
        elif self.type_partition == 'au':
            self.list_labels_au = np.array(self.df['labels_au'].values)
        elif self.type_partition == '2type':
            self.list_labels_ex = np.array(self.df['labels_ex'].values)
            self.list_labels_au = np.array(self.df['labels_au'].values)

        self.list_image_id = np.array(self.df['image_id'].values)

    def __getitem__(self, index):
        image = cv2.imread(self.list_image_id[index])[..., ::-1]
        if self.type_partition == 'ex':
            label = self.list_labels_ex[index]
            return {
                'images': self.transform(image),
                'labels': torch.tensor(label),
            }
        elif self.type_partition == 'au':
            label = ast.literal_eval(self.list_labels_au[index])
            return {
                'images': self.transform(image),
                'labels': torch.tensor(label),
            }

        if self.type_partition == '2type':
            label_ex = self.list_labels_ex[index]
            label_au = ast.literal_eval(self.list_labels_au[index])
            sample = {
                'images': self.transform(image),
                'labels_ex': torch.tensor(label_ex),
                'labels_au': torch.tensor(label_au),
            }
            return sample

    def __len__(self):
        return len(self.df)


class Aff2_Dataset_pseudo(Dataset):
    def __init__(self, type_partition, df=None):
        super(Aff2_Dataset_pseudo, self).__init__()
        self.df = df
        self.type_partition = type_partition
        # TODO adjust the types of data
        if self.type_partition == 'ex':
            self.list_labels_ex = np.array(self.df['labels_ex'].values)
        elif self.type_partition == 'au':
            self.list_labels_au = np.array(self.df['labels_au'].values)
        elif self.type_partition == '2type':
            self.list_labels_ex = np.array(self.df['labels_ex'].values)
            self.list_labels_au = np.array(self.df['labels_au'].values)

    def __getitem__(self, index):

        if self.type_partition == 'ex':
            label = self.list_labels_ex[index]
            # label = to_onehot_ex(label)
            return {
                'labels': torch.tensor(label),
            }
        elif self.type_partition == 'au':
            label = ast.literal_eval(self.list_labels_au[index])
            return {
                'labels': torch.tensor(label),
            }

        if self.type_partition == '2type':
            label_ex = self.list_labels_ex[index]
            # label_ex = to_onehot_ex(label_ex)
            label_au = ast.literal_eval(self.list_labels_au[index])
            sample = {
                'labels_ex': torch.tensor(label_ex),
                'labels_au': torch.tensor(label_au),
            }
            return sample

    def __len__(self):
        return len(self.df)


class Aff2_Dataset_static_multitask_test(Dataset):
    def __init__(self, transform, df=None):
        super(Aff2_Dataset_static_multitask_test, self).__init__()
        self.df = df
        self.transform = transform

        self.list_image_id = np.array(self.df['image_id'].values)

    def __getitem__(self, index):
        image = cv2.imread(self.list_image_id[index])[..., ::-1]

        sample = {
            'images': self.transform(image),
        }
        return sample

    def __len__(self):
        return len(self.df)

