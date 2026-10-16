"""
Text2Motion training dataset with pre-built perturbed text augmentation.

Loads preserving pairs from preserving_pairs_full.jsonl. At each __getitem__,
randomly chooses between original caption or a perturbed (paraphrase) text.
Only uses 'preserve' relation pairs — no altering pairs.
"""

import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import json
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate
from dataset.config import get_dataset_spec


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def _load_preserve_pairs(pair_path):
    """Load preserving pairs from JSONL, build lookup: (motion_id, text_orig) -> [text_pair, ...]"""
    pair_dict = {}
    with open(pair_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if d.get('relation') != 'preserve':
                continue
            key = (d['motion_id'], d['text_orig'])
            if key not in pair_dict:
                pair_dict[key] = []
            pair_dict[key].append(d['text_pair'])
    return pair_dict


class Text2MotionDatasetAug(data.Dataset):
    """Text2Motion dataset with pre-built perturbed text augmentation.

    At each sample, with probability aug_prob, returns a randomly chosen
    perturbed (paraphrase) caption instead of the original.
    """

    def __init__(self, dataset_name, feat_bias=5, unit_length=4, codebook_size=1024,
                 tokenizer_name=None, aug_prob=0.5, pair_data_path=None):
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name
        self.aug_prob = aug_prob

        self.unit_length = unit_length
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1

        spec = get_dataset_spec(dataset_name)
        self.data_root = str(spec.root)
        self.motion_dir = str(spec.motion_dir)
        self.text_dir = str(spec.text_dir)
        self.joints_num = spec.joints
        fps = spec.fps
        self.max_motion_length = 26 if unit_length == 8 else 51

        split_file = pjoin(self.data_root, 'train.txt')

        # Load preserving pairs
        self.pair_dict = {}
        if pair_data_path:
            print(f'[AugDataset] Loading preserve pairs from {pair_data_path}')
            self.pair_dict = _load_preserve_pairs(pair_data_path)
            print(f'[AugDataset] Loaded {len(self.pair_dict)} (motion_id, caption) combos with perturbed texts')

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list, desc='Loading motion data'):
            try:
                m_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s.npy' % name))

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
                                m_token_list_new = [tokens[int(f_tag*fps/unit_length):int(to_tag*fps/unit_length)]
                                                    for tokens in m_token_list
                                                    if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]
                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = '%s_%f_%f' % (name, f_tag, to_tag)
                                data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                        except:
                            pass

                    if flag:
                        data_dict[name] = {'m_token_list': m_token_list,
                                           'text': text_data}
                        new_name_list.append(name)
            except:
                pass
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)

        text_data = random.choice(text_list)
        caption = text_data['caption']

        # Random augmentation: use pre-built perturbed text
        if self.pair_dict and random.random() < self.aug_prob:
            name = self.name_list[item]
            key = (name, caption)
            pairs = self.pair_dict.get(key)
            if pairs:
                caption = random.choice(pairs)

        # Random token dropping (same as original)
        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]

        if m_tokens_len + 1 < self.max_motion_length:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx,
                                       np.ones((self.max_motion_length - 1 - m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        else:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        return caption, m_tokens.reshape(-1), m_tokens_len


def DATALoader(dataset_name, batch_size, codebook_size, tokenizer_name,
               unit_length=4, num_workers=8, aug_prob=0.5, pair_data_path=None):
    train_loader = torch.utils.data.DataLoader(
        Text2MotionDatasetAug(dataset_name, codebook_size=codebook_size,
                              tokenizer_name=tokenizer_name, unit_length=unit_length,
                              aug_prob=aug_prob, pair_data_path=pair_data_path),
        batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
