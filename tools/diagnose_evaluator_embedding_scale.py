#!/usr/bin/env python
"""Compare real-motion embedding scale across evaluation datasets."""

import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import dataset_TM_eval
from dataset.config import configure_evaluator_source, evaluator_opt_path
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.sato_pp_eval import calculate_diversity
from utils.word_vectorizer import WordVectorizer


def main():
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    np.random.seed(123)
    random.seed(123)
    vectorizer = WordVectorizer("./glove", "our_vab")

    for dataset_name in ("t2m", "kit", "motionx"):
        device = torch.device("cuda")
        opt = get_opt(evaluator_opt_path(dataset_name), device)
        opt = configure_evaluator_source(opt, dataset_name)
        wrapper = EvaluatorModelWrapper(opt)
        loader = dataset_TM_eval.DATALoader(
            dataset_name, True, 32, vectorizer
        )

        embeddings = []
        for batch in loader:
            motions = batch[4].to(device)
            lengths = batch[5].to(device)
            with torch.no_grad():
                embedding = wrapper.get_motion_embeddings(motions, lengths)
            embeddings.append(embedding.cpu().numpy())

        values = np.concatenate(embeddings)
        covariance = np.cov(values, rowvar=False)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        effective_rank = eigenvalues.sum() ** 2 / (
            np.square(eigenvalues).sum() + 1e-12
        )
        diversity = calculate_diversity(
            values, 300 if len(values) > 300 else 100
        )
        print({
            "dataset": dataset_name,
            "samples": len(values),
            "mean_norm": float(np.linalg.norm(values, axis=1).mean()),
            "covariance_trace": float(np.trace(covariance)),
            "effective_rank": float(effective_rank),
            "diversity": float(diversity),
            "largest_eigenvalue": float(eigenvalues.max()),
        })

        del wrapper, loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
