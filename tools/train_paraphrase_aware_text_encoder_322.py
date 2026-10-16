"""Phase 2 (Motion-X-322): 重训 322 native matcher 的文本编码器（paraphrase-aware ruler）。

与 HumanML3D 版协议一致：
- 加载标准 matcher `checkpoints/motionx322/text_mot_match/model/finest.tar`
- 冻结 movement_encoder 与 motion_encoder，只训练 text_encoder (BiGRUCo)
- 训练对 = 干净 train 对 + P1/P2/P3 preserve 改写对（改写文本 -> 原动作），
  batch 内 clean:P1:P2:P3 = 1:1:1:1
- 损失: 标准 match 阶段 contrastive（margin 10, roll 负样本）
- 验收门（held-out 尺子验收集）:
  (a) 干净文本 R@1 相对标准尺子下降 < 1 点
  (b) P3 query R@1 相对标准尺子明显改善
- 保存到 `checkpoints/motionx322/text_mot_match_para_aware/model/finest.tar`

Motion-X-322 为变长运动（smplx322, 322 维, drop_tail=0, unit_length=4）。
文本预处理与 322 评测管线一致: TM2TTextProcessor(alphabetic_only=False)。

用法:
    PY=/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python
    CUDA_VISIBLE_DEVICES=3 $PY tools/train_paraphrase_aware_text_encoder_322.py
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.config import get_dataset_spec
from dataset.variable_motion import downsample_lengths, pad_motion_arrays
from train_motionx_evaluator_ddp import (
    MatcherModel,
    TextMotionDataset,
    collate_text_motion_variable,
    contrastive_losses,
)
from utils.sato_pp_eval import TM2TTextProcessor, calculate_R_precision
from utils.word_vectorizer import WordVectorizer

DATA_ROOT = Path("dataset/Motion-X-322")
BASE_CKPT = "checkpoints/motionx322/text_mot_match/model/finest.tar"
OUT_DIR = "checkpoints/motionx322/text_mot_match_para_aware"
MEAN_PATH = DATA_ROOT / "Mean.npy"
STD_PATH = DATA_ROOT / "Std.npy"
MAX_TEXT_LEN = 20
UNIT_LENGTH = 4


def tokenize_to_inputs(vectorizer, tokens, max_text_len):
    tokens = [t if t.count("/") == 1 else "unk/OTHER" for t in tokens]
    if len(tokens) < max_text_len:
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens += ["unk/OTHER"] * (max_text_len + 2 - sent_len)
    else:
        tokens = ["sos/OTHER"] + tokens[:max_text_len] + ["eos/OTHER"]
        sent_len = len(tokens)
    word_embeddings, pos_one_hots = zip(*(vectorizer[t] for t in tokens))
    return (np.stack(word_embeddings).astype(np.float32),
            np.stack(pos_one_hots).astype(np.float32),
            sent_len)


class ParaTextMotionDataset322(torch.utils.data.Dataset):
    """改写对: (P1/P2/P3 text_pair -> 原 motion)，变长运动。"""

    def __init__(self, pair_path, level, mean, std, processor, vectorizer,
                 max_text_len=MAX_TEXT_LEN):
        self.vectorizer = vectorizer
        self.processor = processor
        self.max_text_len = max_text_len
        self.mean, self.std = mean, std
        rows = []
        for row in json.loads(Path(pair_path).read_text(encoding="utf-8")):
            if row.get("level") != level:
                continue
            motion_id = str(row["motion_id"])
            try:
                motion = np.load(DATA_ROOT / "new_joint_vecs" / f"{motion_id}.npy",
                                 allow_pickle=False).astype(np.float32)
                if motion.ndim != 2 or motion.shape[1] != 322:
                    continue
                rows.append((((motion - mean) / std).astype(np.float32),
                             str(row["text_pair"])))
            except (OSError, ValueError):
                continue
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        motion, text_pair = self.rows[index]
        tokens = self.processor.tokenize(text_pair)
        word_embs, pos_ohots, sent_len = tokenize_to_inputs(
            self.vectorizer, tokens, self.max_text_len)
        return (word_embs, pos_ohots, text_pair, sent_len, motion,
                len(motion), "_".join(tokens))


def load_standard_ckpt(device):
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model = MatcherModel(device, dim_pose=322, drop_tail=0,
                         variable_length=True, unit_length=UNIT_LENGTH)
    model.movement_encoder.load_state_dict(ckpt["movement_encoder"])
    model.text_encoder.load_state_dict(ckpt["text_encoder"])
    model.motion_encoder.load_state_dict(ckpt["motion_encoder"])
    return model.to(device), ckpt


def val_retrieval(model, val_rows, mean, std, processor, vectorizer,
                  batch_size=32, device="cuda"):
    """held-out 尺子验收集上的 caption-locked 批内检索（变长）。"""
    import codecs as cs

    ann_cache = {}
    motion_cache = {}

    def get_motion(motion_id):
        if motion_id not in motion_cache:
            motion = np.load(DATA_ROOT / "new_joint_vecs" / f"{motion_id}.npy",
                             allow_pickle=False).astype(np.float32)
            motion_cache[motion_id] = ((motion - mean) / std).astype(np.float32)
        return motion_cache[motion_id]

    def ann_tokens(motion_id, text_orig):
        if motion_id not in ann_cache:
            captions = {}
            with cs.open(DATA_ROOT / "texts" / f"{motion_id}.txt") as f:
                for line in f:
                    parts = line.rstrip("\n").split("#")
                    if len(parts) >= 2:
                        captions[" ".join(parts[0].strip().lower().split())] = [
                            t for t in parts[1].split() if t]
            ann_cache[motion_id] = captions
        key = " ".join(str(text_orig).strip().lower().split())
        return ann_cache[motion_id].get(key)

    rows = sorted(val_rows, key=lambda r: len(r["text_pair"].split()),
                  reverse=True)
    clean_counts = np.zeros(3, dtype=np.float64)
    pert_counts = {"P1": np.zeros(3), "P2": np.zeros(3), "P3": np.zeros(3)}
    clean_n = 0
    pert_n = {"P1": 0, "P2": 0, "P3": 0}
    for start in tqdm(range(0, len(rows), batch_size), desc="val", leave=False):
        batch = rows[start:start + batch_size]
        if len(batch) < 2:
            continue
        entries = []
        for row in batch:
            motion = get_motion(row["motion_id"])
            tokens = ann_tokens(row["motion_id"], row["text_orig"])
            if tokens is None:
                continue
            word_embs, pos_ohots, sent_len = tokenize_to_inputs(
                vectorizer, tokens, MAX_TEXT_LEN)
            entries.append({
                "sent_len": sent_len,
                "word": word_embs,
                "pos": pos_ohots,
                "motion": motion,
                "pert_tokens": processor.tokenize(row["text_pair"]),
                "level": row["level"],
            })
        if not entries:
            continue
        entries.sort(key=lambda e: e["sent_len"], reverse=True)
        word = torch.from_numpy(np.stack([e["word"] for e in entries])).to(device)
        pos = torch.from_numpy(np.stack([e["pos"] for e in entries])).to(device)
        sent = torch.tensor([e["sent_len"] for e in entries], dtype=torch.long,
                            device=device)
        motions, motion_lens, _ = pad_motion_arrays(
            [e["motion"] for e in entries], multiple=UNIT_LENGTH)
        motions = torch.from_numpy(np.stack(motions)).to(device)
        mlens = torch.tensor(motion_lens, dtype=torch.long, device=device)
        with torch.no_grad():
            movements = model.movement_encoder(motions)
            move_lens = downsample_lengths(mlens, UNIT_LENGTH,
                                           include_partial=True).to(device)
            motion_emb = model.motion_encoder(movements.detach(), move_lens)
            text_emb = model.text_encoder(word, pos, sent)
        clean_r, _ = calculate_R_precision(
            text_emb.cpu().numpy(), motion_emb.cpu().numpy(), top_k=3, sum_all=True)
        clean_counts += clean_r
        clean_n += len(entries)
        p_entries = []
        for idx, e in enumerate(entries):
            p_word, p_pos, p_sent = tokenize_to_inputs(
                vectorizer, e["pert_tokens"], MAX_TEXT_LEN)
            p_entries.append((idx, p_word, p_pos, p_sent, e["level"]))
        if len(p_entries) >= 2:
            p_entries.sort(key=lambda t: t[3], reverse=True)
            p_word = torch.from_numpy(
                np.stack([t[1] for t in p_entries])).to(device)
            p_pos = torch.from_numpy(
                np.stack([t[2] for t in p_entries])).to(device)
            p_sent = torch.tensor([t[3] for t in p_entries], dtype=torch.long,
                                  device=device)
            with torch.no_grad():
                p_text_emb = model.text_encoder(p_word, p_pos, p_sent)
            pool = motion_emb[torch.tensor(
                [t[0] for t in p_entries], device=motion_emb.device)]
            p_r_mat, _ = calculate_R_precision(
                p_text_emb.cpu().numpy(), pool.cpu().numpy(),
                top_k=3, sum_all=False)
            for j, t in enumerate(p_entries):
                pert_counts[t[4]] += p_r_mat[j]
                pert_n[t[4]] += 1
    out = {
        "clean_R1": clean_counts[0] / max(clean_n, 1),
        "clean_R2": clean_counts[1] / max(clean_n, 1),
        "clean_R3": clean_counts[2] / max(clean_n, 1),
        "clean_n": clean_n,
    }
    for level in ("P1", "P2", "P3"):
        out[f"{level}_R1"] = pert_counts[level][0] / max(pert_n[level], 1)
        out[f"{level}_n"] = pert_n[level]
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clean-ratio", type=float, default=0.25,
                        help="batch 中干净样本占比（其余均分给 P1/P2/P3）")
    parser.add_argument("--milestones", nargs="+", type=int, default=[20, 40])
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--train-pairs",
                        default="dataset/Motion-X-322/train_ruler_perb_p123.json")
    parser.add_argument("--val-pairs",
                        default="dataset/Motion-X-322/val_ruler_perb_p123.json")
    parser.add_argument("--report-md", default="ablation/ruler_retrain_322_report.md")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    vectorizer = WordVectorizer("glove", "our_vab")
    processor = TM2TTextProcessor(vectorizer, max_text_len=MAX_TEXT_LEN,
                                  alphabetic_only=False)

    train_ids = [l.strip() for l in
                 (DATA_ROOT / "train.txt").read_text().splitlines() if l.strip()]
    val_ruler_rows = json.loads(Path(args.val_pairs).read_text(encoding="utf-8"))
    # 注意: 322 的 val.txt 与 train.txt 不相交，且池不覆盖 val.txt；
    # 尺子验收集是 train 的 deterministic held-out（每 10 取 1），
    # 因此训练必须排除这些 held-out id。
    held_out = {str(r["motion_id"]) for r in val_ruler_rows}
    train_ruler_ids = [m for m in train_ids if m not in held_out]
    train_ruler_file = DATA_ROOT / "train_ruler.txt"
    train_ruler_file.write_text("\n".join(train_ruler_ids) + "\n",
                                encoding="utf-8")
    print(f"train_ruler: {len(train_ruler_ids)} motions -> {train_ruler_file}")

    clean_train = TextMotionDataset(
        DATA_ROOT, "train_ruler", mean, std, vectorizer,
        max_text_len=MAX_TEXT_LEN, max_motion_length=400,
        unit_length=UNIT_LENGTH, limit=0, dataset_key="motionx322")
    para_train = {level: ParaTextMotionDataset322(
        args.train_pairs, level, mean, std, processor, vectorizer)
        for level in ("P1", "P2", "P3")}
    print("clean_train:", len(clean_train),
          {k: len(v) for k, v in para_train.items()})

    val_rows = json.loads(Path(args.val_pairs).read_text(encoding="utf-8"))

    model, std_ckpt = load_standard_ckpt(device)
    for param in model.movement_encoder.parameters():
        param.requires_grad = False
    for param in model.motion_encoder.parameters():
        param.requires_grad = False
    model.eval()

    baseline_val = val_retrieval(model, val_rows, mean, std, processor,
                                 vectorizer, device=device)
    print("baseline (standard ruler) val:", baseline_val)

    per_group = max(args.batch_size // 4, 1)
    clean_count = max(int(round(args.batch_size * args.clean_ratio)), 1)
    para_per_level = max((args.batch_size - clean_count) // 3, 1)
    clean_count = args.batch_size - para_per_level * 3

    def sample_batch():
        batch = []
        batch += [clean_train[random.randrange(len(clean_train))]
                  for _ in range(clean_count)]
        for level in ("P1", "P2", "P3"):
            ds = para_train[level]
            batch += [ds[random.randrange(len(ds))]
                      for _ in range(para_per_level)]
        word, pos, captions, sent, motions, mlens, tokens = \
            collate_text_motion_variable(batch, unit_length=UNIT_LENGTH)
        return word, pos, sent, motions, mlens

    steps_per_epoch = args.steps_per_epoch
    if args.smoke:
        steps_per_epoch = 5
        args.epochs = args.smoke_epochs

    optimizer = torch.optim.Adam(model.text_encoder.parameters(), lr=args.lr)
    best_score = -math.inf
    best_score_state = None
    best_score_epoch = -1
    best_score_clean_drop = math.inf
    best_score_val = None
    best_gate_score = -math.inf
    best_gate_state = None
    best_gate_epoch = -1
    best_gate_clean_drop = math.inf
    best_gate_val = None
    stale = 0
    log_lines = ["# 改写感知尺子重训报告（Motion-X-322）\n",
                 f"- train motions: {len(train_ruler_ids)}, clean rows: "
                 f"{len(clean_train)}, para rows: "
                 f"{ {k: len(v) for k, v in para_train.items()} }",
                 f"- lr={args.lr} milestones={args.milestones} gamma={args.gamma} "
                 f"margin={args.margin} batch={args.batch_size}",
                 f"- baseline val: {baseline_val}\n",
                 "| epoch | clean R@1 | P1 R@1 | P2 R@1 | P3 R@1 | clean R@3 | loss |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]

    for epoch in range(args.epochs):
        decay = sum(epoch >= m for m in args.milestones)
        for group_ in optimizer.param_groups:
            group_["lr"] = args.lr * args.gamma ** decay
        model.train()
        total_loss = 0.0
        for _ in tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", leave=False):
            word, pos, sent, motions, mlens = sample_batch()
            word = word.to(device)
            pos = pos.to(device)
            sent = sent.to(device)
            motions = motions.to(device)
            mlens = mlens.to(device)
            optimizer.zero_grad()
            text_emb, motion_emb = model(word, pos, sent, motions, mlens)
            loss, _, _, _, _ = contrastive_losses(
                text_emb, motion_emb, args.margin, shift=1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / steps_per_epoch

        model.eval()
        val_metrics = val_retrieval(model, val_rows, mean, std, processor,
                                    vectorizer, device=device)
        clean_regression = baseline_val["clean_R1"] - val_metrics["clean_R1"]
        score = (val_metrics["P1_R1"] + val_metrics["P2_R1"]
                 + val_metrics["P3_R1"]) / 3
        log_lines.append(
            f"| {epoch} | {val_metrics['clean_R1']:.4f} | "
            f"{val_metrics['P1_R1']:.4f} | {val_metrics['P2_R1']:.4f} | "
            f"{val_metrics['P3_R1']:.4f} | {val_metrics['clean_R3']:.4f} | "
            f"{avg_loss:.4f} |")
        print(f"epoch {epoch}: loss={avg_loss:.4f} val={val_metrics} "
              f"clean_drop={clean_regression:.4f} score={score:.4f}")
        if score > best_score:
            best_score = score
            best_score_epoch = epoch
            best_score_clean_drop = clean_regression
            best_score_val = val_metrics
            best_score_state = {k: v.detach().cpu() for k, v in
                                model.state_dict().items()}
        gate_ok = clean_regression < 0.01
        if gate_ok and score > best_gate_score:
            best_gate_score = score
            best_gate_epoch = epoch
            best_gate_clean_drop = clean_regression
            best_gate_val = val_metrics
            best_gate_state = {k: v.detach().cpu() for k, v in
                               model.state_dict().items()}
        if gate_ok:
            stale = 0
        else:
            stale += 1
        if stale >= args.patience and epoch >= min(args.milestones or [0]):
            print(f"early stop at epoch {epoch} (patience {args.patience})")
            break

    out_dir = Path(OUT_DIR) / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_passed = best_gate_state is not None
    if gate_passed:
        state = best_gate_state
        epoch = best_gate_epoch
        clean_drop = best_gate_clean_drop
        val = best_gate_val
        score = best_gate_score
    else:
        state = best_score_state if best_score_state is not None else \
            {k: v.detach().cpu() for k, v in model.state_dict().items()}
        epoch = best_score_epoch
        clean_drop = best_score_clean_drop
        val = best_score_val
        score = best_score
    payload = {
        "epoch": epoch,
        "movement_encoder": {k[len("movement_encoder."):]: v for k, v in
                             state.items()
                             if k.startswith("movement_encoder.")},
        "motion_encoder": {k[len("motion_encoder."):]: v for k, v in
                           state.items()
                           if k.startswith("motion_encoder.")},
        "text_encoder": {k[len("text_encoder."):]: v for k, v in
                         state.items()
                         if k.startswith("text_encoder.")},
        "args": {
            "lr": args.lr, "milestones": args.milestones, "gamma": args.gamma,
            "margin": args.margin, "batch_size": args.batch_size,
            "seed": args.seed, "train_pairs": args.train_pairs,
            "val_pairs": args.val_pairs, "dataset": "motionx322",
            "clean_ratio": args.clean_ratio,
        },
    }
    if gate_passed:
        torch.save(payload, out_dir / "finest.tar")
    else:
        # 未通过验收门: 仍保留最优 score 模型供检查，但不生成 finest.tar
        torch.save(payload, out_dir / "best_score.tar")
    torch.save(payload, out_dir / "latest.tar")
    log_lines.append("")
    log_lines.append(f"- **best epoch {epoch}**, score={score:.4f}, "
                     f"clean_R1 drop={clean_drop:.4f}")
    if val:
        log_lines.append(f"- best epoch val: {val}")
    if gate_passed:
        log_lines.append(f"- saved to `{OUT_DIR}/model/finest.tar`")
    else:
        log_lines.append("- 验收门未通过（clean_R1 drop >= 0.01），"
                         "只保存 best_score.tar / latest.tar，不生成 finest.tar")
    Path(args.report_md).write_text("\n".join(log_lines), encoding="utf-8")
    print("\n".join(log_lines))
    if not gate_passed:
        raise SystemExit(
            f"验收门未通过: clean_R1 drop={clean_drop:.4f} >= 0.01")


if __name__ == "__main__":
    main()