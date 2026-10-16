import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate
from functools import partial
from dataset.config import get_dataset_spec, dataset_max_token_length
from dataset.variable_motion import DistributedLengthBucketBatchSampler


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def collate_variable_tokens(batch, pad_idx):
    captions, token_rows, lengths = zip(*batch)
    max_length = max(len(tokens) for tokens in token_rows)
    padded = np.full((len(token_rows), max_length), pad_idx, dtype=np.int64)
    for index, tokens in enumerate(token_rows):
        padded[index, :len(tokens)] = tokens
    return list(captions), torch.from_numpy(padded), torch.as_tensor(lengths)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name=None):
        
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name

        self.unit_length = unit_length
        # self.mot_start_idx = codebook_size
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        spec = get_dataset_spec(dataset_name)
        self.variable_length = spec.variable_length
        self.data_root = str(spec.root)
        self.motion_dir = str(spec.motion_dir)
        self.text_dir = str(spec.text_dir)
        self.joints_num = spec.joints
        fps = spec.fps
        self.max_motion_length = (
            dataset_max_token_length(dataset_name, unit_length)
            if spec.variable_length
            else (26 if unit_length == 8 else 51)
        )

        split_file = pjoin(self.data_root, 'train.txt')


        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s.npy'%name))

                # Read text
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]

                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'text':text_data}
                    new_name_list.append(name)
            except:
                pass
        self.data_dict = data_dict
        self.name_list = new_name_list
        self.lengths = [
            max(len(tokens) for tokens in data_dict[name]['m_token_list'])
            for name in self.name_list
        ]

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)

        text_data = random.choice(text_list)
        caption= text_data['caption']

        
        coin = np.random.choice([False, False, True])
        # print(len(m_tokens))
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]

        if self.variable_length:
            m_tokens = np.concatenate([
                m_tokens, np.ones((1), dtype=int) * self.mot_end_idx
            ], axis=0)
        elif m_tokens_len+1 < self.max_motion_length:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        else:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        return caption, m_tokens.reshape(-1), m_tokens_len




def DATALoader(dataset_name,
                batch_size, codebook_size, tokenizer_name, unit_length=4,
                num_workers=8, distributed=False, seed=123):

    train_set = Text2MotionDataset(dataset_name, codebook_size=codebook_size,
                                   tokenizer_name=tokenizer_name, unit_length=unit_length)
    if train_set.variable_length:
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
            collate_fn=partial(
                collate_variable_tokens,
                pad_idx=train_set.mot_pad_idx,
            ),
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, shuffle=True, seed=seed, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size,
                                              shuffle=sampler is None,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              collate_fn=(
                                                  partial(
                                                      collate_variable_tokens,
                                                      pad_idx=train_set.mot_pad_idx,
                                                  )
                                                  if train_set.variable_length
                                                  else None
                                              ),
                                              drop_last = True)
    

    return train_loader


def cycle(iterable):
    epoch = 0
    while True:
        epoch_sampler = getattr(iterable, "batch_sampler", None)
        if not hasattr(epoch_sampler, "set_epoch"):
            epoch_sampler = getattr(iterable, "sampler", None)
        if hasattr(epoch_sampler, "set_epoch"):
            epoch_sampler.set_epoch(epoch)
        for x in iterable:
            yield x
        epoch += 1
