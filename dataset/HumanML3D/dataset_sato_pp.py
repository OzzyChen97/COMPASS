"""
SATO++ Dataset Loader

加载 preserving/altering pair 数据，与原始 T2M-GPT motion tokens 配对。

输入: 原始 T2M-GPT 数据 + preserving_pairs JSONL 文件
输出: (caption_orig, caption_pair, m_tokens, m_tokens_len, relation, edit_type, slot_mask)
"""

import os
import json
import codecs as cs
import random
import numpy as np
import torch
from torch.utils import data
from os.path import join as pjoin
from tqdm import tqdm
import utils.paramUtil as paramUtil
from dataset.slot_parser import SlotParser


class SATOPPDataset(data.Dataset):
    """
    SATO++ 训练数据集。

    每条样本返回:
        caption_orig: str — 原始文本
        caption_pair: str — preserving/altering pair 文本
        m_tokens: np.array (max_motion_len,) — 离散 motion tokens
        m_tokens_len: int — 有效长度
        relation: str — 'preserve' 或 'alter'
        edit_type: str|None — altering 的编辑类型
        edited_mask: np.array (max_text_len,) bool — 被 CLIP tokenize 后哪些 token 位置被编辑
    """

    def __init__(self, dataset_name, pair_data_path=None, codebook_size=512,
                 tokenizer_name=None, unit_length=4, preserve_ratio=0.7):
        self.max_text_len = 77  # CLIP max context length
        self.preserve_ratio = preserve_ratio
        self.slot_parser = SlotParser()

        # ---- 基础数据加载 (与 dataset_TM_train 相同) ----
        self.max_length = 64
        self.dataset_name = dataset_name
        self.unit_length = unit_length
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.joints_num = 22
            self.max_motion_length = 26 if unit_length == 8 else 51
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.joints_num = 21
            self.max_motion_length = 26 if unit_length == 8 else 51

        self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        self.text_dir = pjoin(self.data_root, 'texts')

        # ---- 加载 motion tokens ----
        split_file = pjoin(self.data_root, 'train.txt')
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        self.data_dict = {}
        self.name_list = []
        for name in tqdm(id_list, desc='Loading motion tokens'):
            try:
                m_token_path = pjoin(self.data_root, tokenizer_name, f'{name}.npy')
                m_token_list = np.load(m_token_path)

                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    for line in f.readlines():
                        try:
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict = {'caption': caption}
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                fps = 20 if dataset_name == 't2m' else 12.5
                                m_token_list_new = [
                                    tokens[int(f_tag * fps / unit_length):
                                           int(to_tag * fps / unit_length)]
                                    for tokens in m_token_list
                                    if int(f_tag * fps / unit_length) < int(to_tag * fps / unit_length)
                                ]
                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = f'{name}_{f_tag}_{to_tag}'
                                self.data_dict[new_name] = {
                                    'm_token_list': m_token_list_new,
                                    'text': [text_dict]
                                }
                                self.name_list.append(new_name)
                        except:
                            pass

                    if flag:
                        self.data_dict[name] = {
                            'm_token_list': m_token_list,
                            'text': text_data
                        }
                        self.name_list.append(name)
            except:
                pass

        # ---- 加载 pair 数据 ----
        # Primary index: motion_id -> {text_orig: [pair_info, ...]}
        self.pair_dict = {}
        # Fallback index: text_orig -> [pair_info, ...] (for motion_ids without pair data)
        self.caption_pairs = {}
        if pair_data_path and os.path.exists(pair_data_path):
            with open(pair_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    p = json.loads(line.strip())
                    mid = p.get('motion_id', '')
                    orig = p.get('text_orig', '')
                    if mid not in self.pair_dict:
                        self.pair_dict[mid] = {}
                    if orig not in self.pair_dict[mid]:
                        self.pair_dict[mid][orig] = []
                    self.pair_dict[mid][orig].append(p)
                    if orig not in self.caption_pairs:
                        self.caption_pairs[orig] = []
                    self.caption_pairs[orig].append(p)
            total_pairs = sum(
                len(pairs) for mid_dict in self.pair_dict.values()
                for pairs in mid_dict.values()
            )
            print(f"Loaded {total_pairs} pairs for {len(self.pair_dict)} motions")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption_orig = text_data['caption']

        # ---- 查找 pair: motion_id 优先，caption fallback ----
        # Extract base motion_id (name can be '010962' or '010962_0.0_2.0')
        base_mid = name.split('_')[0] if not name.split('_')[0].isalpha() else name
        # Try primary: motion_id + caption match
        mid_pairs = self.pair_dict.get(base_mid, {})
        available_pairs = mid_pairs.get(caption_orig, [])
        # Fallback: caption-only match (for motion_ids without pair data)
        if not available_pairs:
            available_pairs = self.caption_pairs.get(caption_orig, [])

        # 按比例选择 preserving/altering
        if available_pairs:
            # 按类型分组
            preserve_pairs = [p for p in available_pairs if p.get('relation') == 'preserve']
            alter_pairs = [p for p in available_pairs if p.get('relation') == 'alter']

            if preserve_pairs and (not alter_pairs or random.random() < self.preserve_ratio):
                pair = random.choice(preserve_pairs)
                relation = 'preserve'
            elif alter_pairs:
                pair = random.choice(alter_pairs)
                relation = 'alter'
            else:
                pair = None
        else:
            pair = None

        if pair is not None:
            caption_pair = pair['text_pair']
            edit_type = pair.get('edit_type') or ''
            # 计算 edited token mask (convert to numpy for collate consistency)
            edited_mask = self.slot_parser.get_edited_token_mask(
                caption_orig, caption_pair, self.max_text_len).numpy()
        else:
            caption_pair = caption_orig
            relation = 'preserve'
            edit_type = ''
            edited_mask = np.zeros(self.max_text_len, dtype=bool)

        # ---- Motion token 处理 (与原 T2M-GPT 一致) ----
        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]

        m_tokens_len = m_tokens.shape[0]
        if m_tokens_len + 1 < self.max_motion_length:
            m_tokens = np.concatenate([
                m_tokens,
                np.ones((1), dtype=int) * self.mot_end_idx,
                np.ones((self.max_motion_length - 1 - m_tokens_len), dtype=int) * self.mot_pad_idx
            ], axis=0)
        else:
            m_tokens = np.concatenate([
                m_tokens,
                np.ones((1), dtype=int) * self.mot_end_idx
            ], axis=0)

        return (caption_orig, caption_pair, m_tokens.reshape(-1),
                m_tokens_len, relation, edit_type, edited_mask)


def collate_fn_sato_pp(batch):
    """自定义 collate: 按 m_tokens_len 排序"""
    batch.sort(key=lambda x: x[3], reverse=True)
    return torch.utils.data.dataloader.default_collate(batch)


def DATALoader(dataset_name, batch_size, codebook_size, tokenizer_name,
               pair_data_path=None, unit_length=4, preserve_ratio=0.7,
               num_workers=8):
    train_loader = torch.utils.data.DataLoader(
        SATOPPDataset(
            dataset_name, pair_data_path=pair_data_path,
            codebook_size=codebook_size, tokenizer_name=tokenizer_name,
            unit_length=unit_length, preserve_ratio=preserve_ratio
        ),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
