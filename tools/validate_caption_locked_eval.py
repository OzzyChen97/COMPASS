#!/usr/bin/env python
"""Read-only validation for deterministic caption-locked eval loading."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import SequentialSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import dataset_TM_eval
from utils.word_vectorizer import WordVectorizer


def load_pair_data(path):
    with open(path, 'r', encoding='utf-8') as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == '[':
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', choices=['t2m', 'kit'], required=True)
    parser.add_argument('--pair-data-path', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    pair_data = load_pair_data(args.pair_data_path)
    vectorizer = WordVectorizer('./glove', 'our_vab')
    loader = dataset_TM_eval.CaptionLockedDATALoader(
        args.dataname,
        True,
        args.batch_size,
        vectorizer,
        pair_data,
        num_workers=args.num_workers,
    )
    dataset = loader.dataset

    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.drop_last is False
    assert len(dataset) > 0

    check_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for index in check_indices:
        first = dataset[index]
        second = dataset[index]
        assert len(first) == 8
        assert first[2] == second[2]
        assert first[3] == second[3]
        assert first[5:] == second[5:]
        np.testing.assert_array_equal(first[4], second[4])

    seen = 0
    for batch in loader:
        assert len(batch) == 8
        seen += len(batch[2])
    assert seen == len(dataset)

    result = {
        **dataset.resolution_stats,
        'batch_size': args.batch_size,
        'num_batches': len(loader),
        'samples_seen': seen,
        'shuffle': False,
        'drop_last': loader.drop_last,
        'deterministic_samples_checked': len(check_indices),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
