import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from functools import partial
from dataset.config import get_dataset_spec, dataset_mean_std_paths
from dataset.variable_motion import (
    DistributedLengthBucketBatchSampler,
    MotionSequenceDataset,
    collate_motion_sequences,
)



class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64, unit_length = 4):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        spec = get_dataset_spec(dataset_name)
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

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

            
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        
        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx+self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4,
               distributed=False,
               seed=123):
    spec = get_dataset_spec(dataset_name)
    if spec.variable_length:
        train_set = MotionSequenceDataset(dataset_name, split="train")
        world_size = torch.distributed.get_world_size() if distributed else 1
        rank = torch.distributed.get_rank() if distributed else 0
        batch_sampler = DistributedLengthBucketBatchSampler(
            train_set.lengths,
            batch_size=batch_size,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            shuffle=True,
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=partial(collate_motion_sequences, multiple=unit_length),
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    
    trainSet = VQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            trainSet, shuffle=True, seed=seed, drop_last=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=sampler is None,
                                              #sampler=sampler,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    epoch = 0
    while True:
        if hasattr(iterable.batch_sampler, "set_epoch"):
            iterable.batch_sampler.set_epoch(epoch)
        elif hasattr(iterable.sampler, "set_epoch"):
            iterable.sampler.set_epoch(epoch)
        for x in iterable:
            yield x
        epoch += 1
