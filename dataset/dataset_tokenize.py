import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from dataset.config import get_dataset_spec, dataset_mean_std_paths


def normalize_and_pad_motion(motion, mean, std, unit_length):
    motion = ((motion - mean) / std).astype(np.float32)
    padded_length = (
        (len(motion) + unit_length - 1) // unit_length
    ) * unit_length
    if padded_length > len(motion):
        motion = np.pad(
            motion,
            ((0, padded_length - len(motion)), (0, 0)),
            mode="constant",
        )
    return motion



class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, window_size = 64, unit_length = 8):
        self.window_size = window_size
        self.unit_length = unit_length
        self.feat_bias = feat_bias

        self.dataset_name = dataset_name
        spec = get_dataset_spec(dataset_name)
        min_motion_len = spec.min_motion_len
        self.data_root = str(spec.root)
        self.motion_dir = str(spec.motion_dir)
        self.text_dir = str(spec.text_dir)
        self.joints_num = spec.joints
        self.max_motion_length = 196
        
        joints_num = self.joints_num

        mean_path, std_path = dataset_mean_std_paths(dataset_name)
        mean = np.load(mean_path)
        std = np.load(std_path)
        
        split_file = pjoin(self.data_root, 'train.txt')
        
        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len:
                    continue
                if not spec.variable_length and len(motion) >= 200:
                    continue

                data_dict[name] = {'motion': motion,
                                   'length': len(motion),
                                   'name': name}
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        if get_dataset_spec(self.dataset_name).variable_length:
            motion = normalize_and_pad_motion(
                motion, self.mean, self.std, self.unit_length)
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]
            "Z Normalization"
            motion = (motion - self.mean) / self.std

        return motion, name

def DATALoader(dataset_name,
                batch_size = 1,
                num_workers=8, unit_length=4, shuffle=True):
    
    train_loader = torch.utils.data.DataLoader(VQMotionDataset(dataset_name, unit_length=unit_length),
                                              batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
