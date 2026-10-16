"""Phase 2: 重训 TM2T 尺子的文本编码器（paraphrase-aware ruler）。

- 加载标准 matcher `checkpoints/t2m/text_mot_match/model/finest.tar`
- 冻结 movement_encoder 与 motion_encoder，只训练 text_encoder (BiGRUCo)
- 训练对: 干净 train 对 + P1/P2/P3 preserve 改写对（改写文本 -> 原动作），
  batch 内 clean:P1:P2:P3 = 1:1:1:1
- 损失: 与标准 match 阶段相同的 contrastive（margin 10, roll 负样本）
- 验收门（用不相交的官方 val）:
  (a) val 干净文本 R@1 相对标准尺子下降 < 1 点
  (b) val P3 query R@1 相对标准尺子明显改善
- 保存到 `checkpoints/t2m/text_mot_match_para_aware/model/finest.tar`
  （key 布局与标准 checkpoint 一致，供 EvaluatorModelWrapper 直接加载）

用法:
    PY=/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python
    CUDA_VISIBLE_DEVICES=3 $PY tools/train_paraphrase_aware_text_encoder.py
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.config import dataset_mean_std_paths, get_dataset_spec
from dataset.variable_motion import downsample_lengths
from models.modules import (MovementConvEncoder, TextEncoderBiGRUCo,
                            MotionEncoderBiGRUCo)
from utils.sato_pp_eval import TM2TTextProcessor, calculate_R_precision
from utils.word_vectorizer import WordVectorizer, POS_enumerator


# ============ Matcher（与 train_motionx_evaluator_ddp.py 的 MatcherModel 一致） ============

class MatcherModel(torch.nn.Module):
    def __init__(self, device, dim_pose=263, drop_tail=4, unit_length=4):
        super().__init__()
        self.drop_tail = int(drop_tail)
        self.unit_length = int(unit_length)
        self.movement_encoder = MovementConvEncoder(dim_pose - self.drop_tail, 512, 512)
        self.text_encoder = TextEncoderBiGRUCo(300, len(POS_enumerator), 512, 512, device)
        self.motion_encoder = MotionEncoderBiGRUCo(512, 1024, 512, device)

    def forward(self, word_embeddings, pos_one_hots, caption_lengths,
                motions, motion_lengths):
        align_idx = np.argsort(motion_lengths.tolist())[::-1].copy()
        motions = motions[align_idx]
        motion_lengths = motion_lengths[align_idx]
        with torch.no_grad():
            movement_input = motions if self.drop_tail == 0 else motions[..., :-self.drop_tail]
            movements = self.movement_encoder(movement_input)
        movement_lengths = downsample_lengths(
            motion_lengths, self.unit_length, include_partial=False
        ).to(motion_lengths.device)
        motion_embeddings = self.motion_encoder(movements.detach(), movement_lengths)
        text_embeddings = self.text_encoder(word_embeddings, pos_one_hots, caption_lengths)
        text_embeddings = text_embeddings[align_idx]
        return text_embeddings, motion_embeddings


def contrastive_losses(text_embeddings, motion_embeddings, margin, shift=1):
    positive_distance = F.pairwise_distance(text_embeddings, motion_embeddings)
    negative_motion = motion_embeddings.roll(int(shift), dims=0)
    negative_distance = F.pairwise_distance(text_embeddings, negative_motion)
    positive_loss = positive_distance.square().mean()
    negative_loss = torch.clamp(margin - negative_distance, min=0).square().mean()
    return positive_loss + negative_loss, positive_loss, negative_loss, \
        positive_distance.mean().item(), negative_distance.mean().item()


# ============ 数据集 ============

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


def load_motion_norm(root, motion_id, mean, std):
    motion = np.load(Path(root) / "new_joint_vecs" / f"{motion_id}.npy",
                     allow_pickle=False).astype(np.float32)
    return ((motion - mean) / std).astype(np.float32)


def crop_and_pad(motion, motion_length, max_motion_length, train=True):
    if train:
        start = random.randint(0, len(motion) - motion_length)
    else:
        start = (len(motion) - motion_length) // 2
    motion = motion[start:start + motion_length]
    if motion_length < max_motion_length:
        motion = np.concatenate([
            motion,
            np.zeros((max_motion_length - motion_length, motion.shape[1]),
                     dtype=np.float32),
        ])
    return motion


class CleanTextMotionDataset(Dataset):
    """干净 train/val 对；caption 与运动裁剪随机（训练）/中心（评测）。"""

    def __init__(self, root, split_file, mean, std, vectorizer, max_text_len=20,
                 max_motion_length=196, unit_length=4, train=True):
        self.vectorizer = vectorizer
        self.max_text_len = max_text_len
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.train = train
        self.mean, self.std = mean, std
        spec = get_dataset_spec("t2m")
        rows = []
        for motion_id in split_file.read_text(encoding="utf-8").splitlines():
            motion_id = motion_id.strip()
            if not motion_id:
                continue
            motion_path = Path(root) / "new_joint_vecs" / f"{motion_id}.npy"
            text_path = Path(root) / "texts" / f"{motion_id}.txt"
            try:
                motion = np.load(motion_path, allow_pickle=False).astype(np.float32)
                if len(motion) < spec.min_motion_len or len(motion) >= 200:
                    continue
                captions = []
                for raw_line in text_path.read_text(encoding="utf-8").splitlines():
                    parts = raw_line.strip().split("#")
                    if len(parts) < 4:
                        continue
                    from_tag, to_tag = float(parts[2]), float(parts[3])
                    from_tag = 0.0 if math.isnan(from_tag) else from_tag
                    to_tag = 0.0 if math.isnan(to_tag) else to_tag
                    if from_tag == 0.0 and to_tag == 0.0:
                        captions.append([t for t in parts[1].split() if t])
                if captions:
                    normalized = ((motion - mean) / std).astype(np.float32)
                    rows.append((normalized, captions))
            except (OSError, ValueError, UnicodeError):
                continue
        rows.sort(key=lambda row: len(row[0]))
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        motion, captions = self.rows[index]
        tokens = list(random.choice(captions))
        word_embs, pos_ohots, sent_len = tokenize_to_inputs(
            self.vectorizer, tokens, self.max_text_len)
        motion_length = (len(motion) // self.unit_length) * self.unit_length
        motion_length = max(self.unit_length, motion_length)
        motion = crop_and_pad(motion, motion_length, self.max_motion_length,
                              train=self.train)
        return word_embs, pos_ohots, sent_len, motion, motion_length


class ParaTextMotionDataset(Dataset):
    """改写对: (P1/P2/P3 text_pair -> 原 motion)。文本用 TM2T spaCy 预处理。"""

    def __init__(self, root, pair_path, level, mean, std, processor, vectorizer,
                 max_text_len=20, max_motion_length=196, unit_length=4, train=True):
        self.vectorizer = vectorizer
        self.processor = processor
        self.max_text_len = max_text_len
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.train = train
        self.mean, self.std = mean, std
        self.root = root
        rows = []
        for row in json.loads(Path(pair_path).read_text(encoding="utf-8")):
            if row.get("level") != level:
                continue
            motion_id = str(row["motion_id"])
            try:
                motion = np.load(Path(root) / "new_joint_vecs" / f"{motion_id}.npy",
                                 allow_pickle=False).astype(np.float32)
                if len(motion) < get_dataset_spec("t2m").min_motion_len or len(motion) >= 200:
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
        motion_length = (len(motion) // self.unit_length) * self.unit_length
        motion_length = max(self.unit_length, motion_length)
        motion = crop_and_pad(motion, motion_length, self.max_motion_length,
                              train=self.train)
        return word_embs, pos_ohots, sent_len, motion, motion_length


def collate_train(batch):
    batch.sort(key=lambda item: item[2], reverse=True)
    word, pos, sent_len, motions, m_lengths = zip(*batch)
    return (torch.from_numpy(np.stack(word)),
            torch.from_numpy(np.stack(pos)),
            torch.tensor(sent_len, dtype=torch.long),
            torch.from_numpy(np.stack(motions)),
            torch.tensor(m_lengths, dtype=torch.long))


# ============ 验收评测 ============

def val_retrieval(model, val_rows, root, mean, std, processor, vectorizer,
                  batch_size=32, max_text_len=20, max_motion_length=196,
                  unit_length=4, device="cuda"):
    """val 上 caption-locked 批内检索: clean query 与 P1/P2/P3 query 各算 R@。

    val_rows: list of (motion_id, text_orig, text_pair, level)
    clean 文本用标注文件的原始 tokens（与 eval loader 一致），
    扰动文本用 TM2T spaCy 预处理（与测试期扰动编码一致）。
    """
    import codecs as cs

    ann_cache = {}
    motion_cache = {}

    def get_motion(motion_id):
        if motion_id not in motion_cache:
            motion_cache[motion_id] = load_motion_norm(root, motion_id, mean, std)
        return motion_cache[motion_id]

    def ann_tokens(motion_id, text_orig):
        if motion_id not in ann_cache:
            captions = {}
            with cs.open(Path(root) / "texts" / f"{motion_id}.txt") as f:
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
            if len(motion) < get_dataset_spec("t2m").min_motion_len or len(motion) >= 200:
                continue
            motion_length = (len(motion) // unit_length) * unit_length
            motion = crop_and_pad(motion, motion_length, max_motion_length,
                                  train=False)
            tokens = ann_tokens(row["motion_id"], row["text_orig"])
            if tokens is None:
                continue
            word_embs, pos_ohots, sent_len = tokenize_to_inputs(
                vectorizer, tokens, max_text_len)
            entries.append({
                "sent_len": sent_len,
                "word": word_embs,
                "pos": pos_ohots,
                "motion": motion,
                "mlen": motion_length,
                "pert_tokens": processor.tokenize(row["text_pair"]),
                "level": row["level"],
            })
        if not entries:
            continue
        # BiGRU 需要 caption 长度降序
        entries.sort(key=lambda e: e["sent_len"], reverse=True)
        word = torch.from_numpy(np.stack([e["word"] for e in entries])).to(device)
        pos = torch.from_numpy(np.stack([e["pos"] for e in entries])).to(device)
        sent = torch.tensor([e["sent_len"] for e in entries], dtype=torch.long,
                            device=device)
        motions = torch.from_numpy(
            np.stack([e["motion"] for e in entries])).to(device)
        mlens = torch.tensor([e["mlen"] for e in entries], dtype=torch.long,
                             device=device)
        with torch.no_grad():
            # 直接按 entries 顺序编码（不走 MatcherModel.forward 的排序置换），
            # 保证 query 与 pool 的对角线对应正确
            movements = model.movement_encoder(motions[..., :-model.drop_tail])
            move_lens = downsample_lengths(mlens, model.unit_length,
                                           include_partial=False)
            move_lens = move_lens.to(device)
            motion_emb = model.motion_encoder(movements.detach(), move_lens)
            text_emb = model.text_encoder(word, pos, sent)
        clean_r, _ = calculate_R_precision(
            text_emb.cpu().numpy(), motion_emb.cpu().numpy(), top_k=3, sum_all=True)
        clean_counts += clean_r
        clean_n += len(entries)
        # 扰动 query 全部堆叠，pool 为整个 kept 批次的 GT 动作，对角线为正确匹配；
        # query 与 pool 必须按同一顺序（p_sent 降序满足 BiGRU）排列
        p_entries = []
        for idx, e in enumerate(entries):
            p_word, p_pos, p_sent = tokenize_to_inputs(
                vectorizer, e["pert_tokens"], max_text_len)
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


def load_standard_ckpt(device):
    path = Path("checkpoints/t2m/text_mot_match/model/finest.tar")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MatcherModel(device)
    model.movement_encoder.load_state_dict(ckpt["movement_encoder"])
    model.text_encoder.load_state_dict(ckpt["text_encoder"])
    model.motion_encoder.load_state_dict(ckpt["motion_encoder"])
    return model.to(device), ckpt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--milestones", nargs="+", type=int, default=[20, 40])
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--smoke", action="store_true", help="小数据快速验证")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--data-root", default="dataset/HumanML3D")
    parser.add_argument("--train-pairs", default="dataset/HumanML3D/train_ruler_perb_p123.json")
    parser.add_argument("--val-pairs", default="dataset/HumanML3D/val_ruler_perb_p123.json")
    parser.add_argument("--out-dir", default="checkpoints/t2m/text_mot_match_para_aware")
    parser.add_argument("--report-md", default="ablation/ruler_retrain_report.md")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    data_root = Path(args.data_root)
    mean_path, std_path = dataset_mean_std_paths("t2m")
    mean = np.load(mean_path)
    std = np.load(std_path)
    vectorizer = WordVectorizer("glove", "our_vab")
    processor = TM2TTextProcessor(vectorizer, max_text_len=20, alphabetic_only=True)

    # 不相交 split: train_ruler = train.txt - val.txt
    train_ids = [l.strip() for l in (data_root / "train.txt").read_text().splitlines() if l.strip()]
    val_ids = set(l.strip() for l in (data_root / "val.txt").read_text().splitlines() if l.strip())
    train_ruler_ids = [m for m in train_ids if m not in val_ids]
    train_ruler_file = data_root / "train_ruler.txt"
    train_ruler_file.write_text("\n".join(train_ruler_ids) + "\n", encoding="utf-8")
    print(f"train_ruler: {len(train_ruler_ids)} motions -> {train_ruler_file}")

    clean_train = CleanTextMotionDataset(
        data_root, train_ruler_file, mean, std, vectorizer, train=True)
    para_train = {level: ParaTextMotionDataset(
        data_root, args.train_pairs, level, mean, std, processor, vectorizer,
        train=True) for level in ("P1", "P2", "P3")}
    print("clean_train:", len(clean_train),
          {k: len(v) for k, v in para_train.items()})

    val_rows = json.loads(Path(args.val_pairs).read_text(encoding="utf-8"))

    model, std_ckpt = load_standard_ckpt(device)
    for param in model.movement_encoder.parameters():
        param.requires_grad = False
    for param in model.motion_encoder.parameters():
        param.requires_grad = False
    model.eval()

    # 基线: 标准尺子在 val 上的校准数字
    baseline_val = val_retrieval(
        model, val_rows, data_root, mean, std, processor, vectorizer,
        device=device)
    print("baseline (standard ruler) val:", baseline_val)

    # 采样器: 每 batch 8 clean + 8 P1 + 8 P2 + 8 P3
    per_group = max(args.batch_size // 4, 1)
    group = per_group * 4

    def sample_batch():
        batch = []
        batch += [clean_train[random.randrange(len(clean_train))]
                  for _ in range(per_group)]
        for level in ("P1", "P2", "P3"):
            ds = para_train[level]
            batch += [ds[random.randrange(len(ds))] for _ in range(per_group)]
        return collate_train(batch)

    steps_per_epoch = args.steps_per_epoch
    if args.smoke:
        steps_per_epoch = 5
        args.epochs = args.smoke_epochs

    optimizer = torch.optim.Adam(model.text_encoder.parameters(), lr=args.lr)
    best_score = -math.inf
    best_state = None
    best_epoch = -1
    best_clean_drop = math.inf
    best_val = None
    stale = 0
    log_lines = []
    log_lines.append(f"# 改写感知尺子重训报告\n")
    log_lines.append(f"- train motions: {len(train_ruler_ids)}, clean rows: {len(clean_train)}, "
                     f"para rows: { {k: len(v) for k, v in para_train.items()} }")
    log_lines.append(f"- lr={args.lr} milestones={args.milestones} gamma={args.gamma} "
                     f"margin={args.margin} batch={args.batch_size}")
    log_lines.append(f"- baseline val: {baseline_val}\n")
    log_lines.append("| epoch | clean R@1 | P1 R@1 | P2 R@1 | P3 R@1 | clean R@3 | loss |")
    log_lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for epoch in range(args.epochs):
        decay = sum(epoch >= m for m in args.milestones)
        for group_ in optimizer.param_groups:
            group_["lr"] = args.lr * args.gamma ** decay
        model.train()
        total_loss = 0.0
        for _ in tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", leave=False):
            word, pos, sent, motions, mlens = sample_batch()
            word, pos, sent = word.to(device), pos.to(device), sent.to(device)
            motions, mlens = motions.to(device), mlens.to(device)
            optimizer.zero_grad()
            text_emb, motion_emb = model(word, pos, sent, motions, mlens)
            loss, _, _, _, _ = contrastive_losses(
                text_emb, motion_emb, args.margin, shift=1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / steps_per_epoch

        model.eval()
        val_metrics = val_retrieval(
            model, val_rows, data_root, mean, std, processor, vectorizer,
            device=device)
        clean_regression = baseline_val["clean_R1"] - val_metrics["clean_R1"]
        score = (val_metrics["P1_R1"] + val_metrics["P2_R1"] + val_metrics["P3_R1"]) / 3
        log_lines.append(
            f"| {epoch} | {val_metrics['clean_R1']:.4f} | {val_metrics['P1_R1']:.4f} | "
            f"{val_metrics['P2_R1']:.4f} | {val_metrics['P3_R1']:.4f} | "
            f"{val_metrics['clean_R3']:.4f} | {avg_loss:.4f} |")
        print(f"epoch {epoch}: loss={avg_loss:.4f} val={val_metrics} "
              f"clean_drop={clean_regression:.4f} score={score:.4f}")
        if score > best_score and clean_regression < 0.01:
            best_score = score
            best_epoch = epoch
            best_clean_drop = clean_regression
            best_val = val_metrics
            best_state = {k: v.detach().cpu() for k, v in
                          model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience and epoch >= min(args.milestones or [0]):
            print(f"early stop at epoch {epoch} (patience {args.patience})")
            break

    if best_state is None:
        raise RuntimeError("没有任何 epoch 同时满足 val P 检索提升与 clean 不退化")

    out_dir = Path(args.out_dir) / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": best_epoch,
        "movement_encoder": {k[len("movement_encoder."):]: v for k, v in best_state.items()
                             if k.startswith("movement_encoder.")},
        "motion_encoder": {k[len("motion_encoder."):]: v for k, v in best_state.items()
                           if k.startswith("motion_encoder.")},
        "text_encoder": {k[len("text_encoder."):]: v for k, v in best_state.items()
                         if k.startswith("text_encoder.")},
        "args": {
            "lr": args.lr, "milestones": args.milestones, "gamma": args.gamma,
            "margin": args.margin, "batch_size": args.batch_size,
            "seed": args.seed, "train_pairs": args.train_pairs,
            "val_pairs": args.val_pairs,
        },
    }
    torch.save(payload, out_dir / "finest.tar")
    torch.save(payload, out_dir / "latest.tar")
    log_lines.append("")
    log_lines.append(f"- **best epoch {best_epoch}**, score={best_score:.4f}, "
                     f"clean_R1 drop={best_clean_drop:.4f}")
    if best_val:
        log_lines.append(f"- best epoch val: {best_val}")
    log_lines.append(f"- saved to `{args.out_dir}/model/finest.tar`")
    Path(args.report_md).write_text("\n".join(log_lines), encoding="utf-8")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()