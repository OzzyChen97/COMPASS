import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from collections import Counter
from tqdm import tqdm
from functools import partial

import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate
from dataset.config import (
    dataset_max_motion_length,
    dataset_mean_std_paths,
    get_dataset_spec,
)
from dataset.variable_motion import pad_motion_arrays


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def collate_fn_variable(batch, unit_length=4):
    batch.sort(key=lambda item: item[3], reverse=True)
    word, pos, captions, sent_lengths, motions, _, tokens, names = zip(*batch)
    padded, motion_lengths, _ = pad_motion_arrays(motions, multiple=unit_length)
    return (
        torch.as_tensor(np.stack(word)),
        torch.as_tensor(np.stack(pos)),
        list(captions),
        torch.as_tensor(sent_lengths),
        padded,
        motion_lengths,
        list(tokens),
        list(names),
    )


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, is_test, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4):
        
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        spec = get_dataset_spec(dataset_name)
        self.variable_length = spec.variable_length
        self.data_root = str(spec.root)
        self.motion_dir = str(spec.motion_dir)
        self.text_dir = str(spec.text_dir)
        self.joints_num = spec.joints
        fps = spec.fps
        self.max_motion_length = dataset_max_motion_length(dataset_name)

        mean_path, std_path = dataset_mean_std_paths(dataset_name)
        mean = np.load(mean_path)
        std = np.load(std_path)
        
        if is_test:
            split_file = pjoin(self.data_root, 'test.txt')
        else:
            split_file = pjoin(self.data_root, 'val.txt')

        min_motion_len = spec.min_motion_len
        # min_motion_len = 64

        joints_num = self.joints_num

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
                if len(motion) < min_motion_len:
                    continue
                if not self.variable_length and len(motion) >= 200:
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*fps) : int(to_tag*fps)]
                                if len(n_motion) < min_motion_len:
                                    continue
                                if not self.variable_length and len(n_motion) >= 200:
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        if self.variable_length:
            self.pointer = 0
        else:
            self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        # data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.variable_length:
            coin2 = 'full'
        elif self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'full':
            m_length = len(motion)
        elif coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        if self.variable_length:
            motion = motion[:m_length]
        else:
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if not self.variable_length and m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name


def _normalize_caption_key(caption):
    return " ".join(str(caption).strip().lower().split())


def _parse_time_tag(value):
    value = float(value)
    return 0.0 if np.isnan(value) else value


def _resample_motion_frames(motion, target_length):
    """Linearly resample a motion feature sequence to an explicit frame count."""
    source_length = len(motion)
    if source_length < 1:
        raise ValueError('cannot resample an empty motion')
    if source_length == target_length:
        return motion.copy()
    positions = np.linspace(0.0, source_length - 1, target_length)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, source_length - 1)
    weight = (positions - lower).astype(motion.dtype, copy=False)[:, None]
    return motion[lower] * (1.0 - weight) + motion[upper] * weight


class CaptionLockedText2MotionDataset(data.Dataset):
    """Deterministic perturbation-eval samples resolved from pair annotations.

    Every retained row is resolved by the exact normalized
    ``(motion_id, text_orig)`` key. Unlike ``Text2MotionDataset``, this dataset
    does not randomly select captions, motion lengths, or crop offsets.
    """

    def __init__(self, dataset_name, is_test, w_vectorizer, pair_data,
                 max_text_len=20, unit_length=4, length_policy='strict'):
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        if length_policy not in {'strict', 'repair-all'}:
            raise ValueError('unknown caption-locked length policy: %s' % length_policy)
        self.length_policy = length_policy

        spec = get_dataset_spec(dataset_name)
        self.variable_length = spec.variable_length
        self.max_motion_length = dataset_max_motion_length(dataset_name)
        self.data_root = str(spec.root)
        self.joints_num = spec.joints
        self.fps = spec.fps
        self.min_motion_len = spec.min_motion_len

        self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        self.text_dir = pjoin(self.data_root, 'texts')
        mean_path, std_path = dataset_mean_std_paths(dataset_name)
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

        split_name = 'test.txt' if is_test else 'val.txt'
        with cs.open(pjoin(self.data_root, split_name), 'r') as handle:
            split_ids = {line.strip() for line in handle if line.strip()}

        self.entries = []
        skipped = Counter()
        repairs = Counter()
        annotation_cache = {}
        motion_cache = {}
        segment_count = 0

        for pair_idx, pair in enumerate(pair_data or []):
            motion_id = str(pair.get('motion_id', '')).strip()
            text_orig = str(pair.get('text_orig', '')).strip()
            if not motion_id or not text_orig:
                skipped['missing_pair_key'] += 1
                continue
            if motion_id not in split_ids:
                skipped['outside_split'] += 1
                continue

            try:
                annotations = annotation_cache.get(motion_id)
                if annotations is None:
                    annotations = self._load_annotations(motion_id)
                    annotation_cache[motion_id] = annotations
                key = _normalize_caption_key(text_orig)
                matches = [row for row in annotations if row['key'] == key]
                if not matches:
                    skipped['caption_not_found'] += 1
                    continue

                # Prefer the full-motion annotation when duplicate captions exist.
                annotation = next(
                    (row for row in matches
                     if row['from_tag'] == 0.0 and row['to_tag'] == 0.0),
                    matches[0])

                full_motion = motion_cache.get(motion_id)
                if full_motion is None:
                    full_motion = np.load(pjoin(self.motion_dir, motion_id + '.npy'))
                    motion_cache[motion_id] = full_motion

                from_tag = annotation['from_tag']
                to_tag = annotation['to_tag']
                if from_tag == 0.0 and to_tag == 0.0:
                    start, end = 0, len(full_motion)
                else:
                    start = max(0, min(int(from_tag * self.fps), len(full_motion)))
                    end = max(0, min(int(to_tag * self.fps), len(full_motion)))
                    segment_count += 1

                motion = full_motion[start:end]
                original_start, original_end = start, end
                original_length = len(motion)
                repair_type = 'none'

                if self.length_policy == 'repair-all':
                    if original_length == 0:
                        # A zero-duration annotation is unusable. The source motion is
                        # still caption-locked and is the least lossy recoverable input.
                        start, end = 0, len(full_motion)
                        motion = full_motion
                        repair_type = 'invalid-segment-use-full-motion'
                    if len(motion) < self.min_motion_len:
                        if len(full_motion) >= self.min_motion_len and len(motion) > 0:
                            center = (start + end) // 2
                            start = max(0, min(
                                center - self.min_motion_len // 2,
                                len(full_motion) - self.min_motion_len))
                            end = start + self.min_motion_len
                            motion = full_motion[start:end]
                            repair_type = 'expand-short-segment-to-minimum'
                        else:
                            # The TM2T evaluator can encode native short motions
                            # (its effective length is floor(frames / 4), clamped
                            # to at least one). Keeping source frames is less
                            # distortive than interpolating 263-D rotations,
                            # velocities, and contacts merely to reach 40 frames.
                            repair_type = 'keep-native-short'
                    if not self.variable_length and len(motion) >= 200:
                        motion = _resample_motion_frames(
                            motion, self.max_motion_length)
                        repair_type = 'resample-long-to-maximum'

                repairs[repair_type] += 1

                if len(motion) < self.min_motion_len and self.length_policy == 'strict':
                    skipped['motion_too_short'] += 1
                    continue
                if not self.variable_length and len(motion) >= 200:
                    skipped['motion_too_long'] += 1
                    continue

                self.entries.append({
                    'pair_idx': pair_idx,
                    'motion_id': motion_id,
                    'caption': text_orig,
                    'caption_key': key,
                    'tokens': annotation['tokens'],
                    'motion': motion,
                    'length_repair': {
                        'policy': self.length_policy,
                        'repair_type': repair_type,
                        'original_start': original_start,
                        'original_end': original_end,
                        'original_length': original_length,
                        'repaired_start': start,
                        'repaired_end': end,
                        'repaired_length': len(motion),
                        'full_motion_length': len(full_motion),
                    },
                })
            except (OSError, ValueError, IndexError):
                skipped['invalid_source'] += 1

        self.resolution_stats = {
            'input_pairs': len(pair_data or []),
            'resolved_pairs': len(self.entries),
            'segment_pairs': segment_count,
            'skipped': dict(sorted(skipped.items())),
            'repairs': dict(sorted(repairs.items())),
        }
        if not self.entries:
            raise ValueError(
                'No caption-locked samples resolved from pair_data: %s'
                % self.resolution_stats)

    def _load_annotations(self, motion_id):
        annotations = []
        with cs.open(pjoin(self.text_dir, motion_id + '.txt'), 'r') as handle:
            for line in handle:
                fields = line.rstrip('\n').split('#')
                if len(fields) < 4:
                    continue
                caption = fields[0].strip()
                annotations.append({
                    'key': _normalize_caption_key(caption),
                    'tokens': fields[1].strip().split(),
                    'from_tag': _parse_time_tag(fields[2]),
                    'to_tag': _parse_time_tag(fields[3]),
                })
        return annotations

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, item):
        entry = self.entries[item]
        caption = entry['caption']
        tokens = list(entry['tokens'])

        if len(tokens) < self.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens += ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = ['sos/OTHER'] + tokens[:self.max_text_len] + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        motion = entry['motion']
        if self.variable_length:
            m_length = len(motion)
        else:
            m_length = (len(motion) // self.unit_length) * self.unit_length
            crop_start = (len(motion) - m_length) // 2
            motion = motion[crop_start:crop_start + m_length]
        motion = (motion - self.mean) / self.std
        if not self.variable_length and m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1])),
            ], axis=0)

        return (word_embeddings, pos_one_hots, caption, sent_len, motion,
                m_length, '_'.join(tokens), entry['motion_id'])




def DATALoader(dataset_name, is_test,
                batch_size, w_vectorizer,
                num_workers=8, unit_length=4, max_text_len=None):
    if max_text_len is None:
        max_text_len = 20
    
    dataset = Text2MotionDataset(
                                              dataset_name, is_test, w_vectorizer,
                                              max_text_len=max_text_len,
                                              unit_length=unit_length)
    val_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size,
                                              shuffle = True,
                                              num_workers=num_workers,
                                              collate_fn=(
                                                  partial(collate_fn_variable, unit_length=unit_length)
                                                  if dataset.variable_length
                                                  else collate_fn
                                              ),
                                              drop_last = not dataset.variable_length)
    return val_loader


def CaptionLockedDATALoader(dataset_name, is_test, batch_size, w_vectorizer,
                            pair_data, num_workers=8, unit_length=4,
                            max_text_len=None, length_policy='strict'):
    if max_text_len is None:
        max_text_len = 20
    dataset = CaptionLockedText2MotionDataset(
        dataset_name,
        is_test,
        w_vectorizer,
        pair_data,
        max_text_len=max_text_len,
        unit_length=unit_length,
        length_policy=length_policy)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=(
            partial(collate_fn_variable, unit_length=unit_length)
            if dataset.variable_length
            else collate_fn
        ),
        drop_last=False)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
