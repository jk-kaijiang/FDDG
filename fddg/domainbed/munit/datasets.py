import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import json
import pandas as pd

class SensitiveImageFolder(ImageFolder):
    def __init__(self,
        root,
        transform = None,
        ):
        super().__init__(root,
                         transform,
                         )
        path_list = root.split('/')
        path_list.pop()
        dict_path = "/".join(path_list)
        with open(dict_path + '/data.json') as f:
            self.dict = json.load(f)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        file_name = path.split('/')[-1]

        z = self.dict[file_name][2]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, z

    def __len__(self):
        return  len(self.samples)



class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)



class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        self.datasets = []
        for i, environment in enumerate(environments):
            if environment != environments[test_envs]:
                env_transform = transform

                path = os.path.join(root, environment)
                env_dataset = SensitiveImageFolder(path,
                    transform=env_transform)

                self.datasets.append(env_dataset)


        self.ori_dataset = ConcatDataset(self.datasets)
        self.shuffle = torch.randperm(len(self.ori_dataset))
        self.dataset = Subset(self.ori_dataset, self.shuffle)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class CelebA_5(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["0", "1", "2", "3", "4"]
    def __init__(self, root, test_envs):
        self.dir = os.path.join(root, "CelebA_5/")
        super().__init__(self.dir, test_envs)

class CelebA_9(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ['00', '11', '12', '21', '22', '31', '32', '41', '42']
    def __init__(self, root, test_envs):
        self.dir = os.path.join(root, "CelebA_9/")
        super().__init__(self.dir, test_envs)



class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs):
        self.dir = os.path.join("/home/YOUR_PATH/Testing/mbdg-main/domainbed/data/", "PACS/")
        super().__init__(self.dir, test_envs)

class CCMNIST1(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = [0, 1, 2]
    def __init__(self, root, test_envs):
        self.dir = os.path.join("/home/YOUR_PATH/data/CCMNIST1/")
        super().__init__(self.dir, test_envs)

class FairFace(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ['0', '1', '2', '3', '4', '5', '6']
    def __init__(self, root, test_envs):
        self.dir = os.path.join("/home/YOUR_PATH/data/FairFace/")
        super().__init__(self.dir, test_envs)

class YFCC(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ['0', '1', '2']
    def __init__(self, root, test_envs):
        self.dir = os.path.join("/home/YOUR_PATH/data/YFCC/")
        super().__init__(self.dir, test_envs)

class NYPD(Dataset):
    def __init__(self, env):
        df = pd.read_csv("/home/YOUR_PATH/data/NYPD/" + str(env) + ".csv", encoding='latin-1', low_memory=False)

        self.x, self.y, self.z = self.df2tensor(df)

    def df2tensor(self, initial_data):
        y = initial_data['frisked'].values
        others = initial_data.drop('frisked', axis=1)

        z = others['race_B'].values
        x = others.drop('race_B', axis=1).values

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y)
        z = torch.tensor(z, dtype=torch.float32)
        return x, y, z

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]

def get_CelebA5_loaders(env):

    env_dataset = CelebA_5('/home/YOUR_PATH/data/', env).dataset
    loader = DataLoader(env_dataset, batch_size=1, num_workers=4, pin_memory=True)
    return loader, loader, loader, loader

def get_CCMNIST1_loaders(env, batch_size):

    env_dataset = CCMNIST1('/home/YOUR_PATH/data/', env).dataset
    loader = DataLoader(env_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    return loader, loader, loader, loader

def get_FairFace_loaders(env, batch_size):

    env_dataset = FairFace('/home/YOUR_PATH/data/', env).dataset
    loader = DataLoader(env_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    return loader, loader, loader, loader

def get_YFCC_loaders(env, batch_size):
    env_dataset = YFCC('/home/YOUR_PATH/data/', env).dataset
    loader = DataLoader(env_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    return loader, loader, loader, loader

def get_NYPD_loaders(env, batch_size):
    env_list = []
    for i in range(5):
        if i != env:
            env_list.append(NYPD(i))
    env_dataset = ConcatDataset(env_list)
    loader = DataLoader(env_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    return loader, loader, loader, loader