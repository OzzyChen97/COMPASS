"""Phase 0 尺子诊断：量化 TM2T 文本编码器对 P1/P2/P3 扰动文本的敏感性。

三部分输出：
1. OOV 率：扰动文本 token 在 GloVe 'our_vab' 词表外的比例
   （按 token 和按句子两个口径，区分 BiGRU 是否"看不见"这些词）
2. 余弦相似度：每个 level 的 clean vs perturbed TM2T 文本嵌入平均余弦
   （尺子是否"感知到"改写；越接近 1 越不敏感）
3. 尺子校准 R@：caption-locked 加载 test 对，用 GT 动作在批内检索
   - clean query -> GT 动作（天花板）
   - P1/P2/P3 query -> 各自原动作（尺子"读懂改写"的能力）

判定规则（写进报告末尾）：
- 若 P3 query 的 R@ 显著低于 clean 天花板 -> 确认尺子是 P3 瓶颈，进入 Phase 1
- 若 OOV token 比例 > 15% -> BiGRU 重训救不回未登录词，需词表扩展或换 CLIP 文本侧

用法：
    PY=/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python
    CUDA_VISIBLE_DEVICES=3 $PY tools/diagnose_ruler_paraphrase_sensitivity.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.config import configure_evaluator_source, evaluator_opt_path
from dataset.dataset_TM_eval import CaptionLockedDATALoader
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.sato_pp_eval import TM2TTextProcessor, calculate_R_precision
from utils.word_vectorizer import WordVectorizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-path",
        default="dataset/HumanML3D/test_perb_deterministic.json",
        help="preserve 扰动对文件",
    )
    parser.add_argument("--levels", nargs="+", default=["P1", "P2", "P3"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="默认 cuda 可用则 cuda")
    parser.add_argument(
        "--out-md",
        default="ablation/ruler_diagnosis_humanml.md",
        help="诊断报告输出路径",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="可选：每个 level 只取前 N 对做 smoke 测试")
    parser.add_argument("--encode-chunk", type=int, default=256,
                        help="文本编码分块大小（控显存）")
    return parser.parse_args()


def load_pairs(pair_path, levels, limit):
    rows = json.loads(Path(pair_path).read_text(encoding="utf-8"))
    by_level = {}
    for level in levels:
        selected = [r for r in rows if r.get("level") == level]
        if limit:
            selected = selected[:limit]
        by_level[level] = selected
    return by_level


def build_evaluator(device):
    opt = get_opt(evaluator_opt_path("t2m"), device)
    opt = configure_evaluator_source(opt, "t2m")
    wrapper = EvaluatorModelWrapper(opt)
    return wrapper, opt


def oov_stats(pairs, processor, vectorizer):
    """按 level 统计扰动文本的 OOV 情况。

    Returns:
        dict: level -> {token_oov_rate, caption_oov_rate, token_total, oov_examples}
    """
    stats = {}
    for level, rows in pairs.items():
        total_tokens = 0
        oov_tokens = 0
        captions_with_oov = 0
        examples = []
        for row in rows:
            text = str(row.get("text_pair", ""))
            tokens = processor.tokenize(text)
            row_oov = 0
            for token in tokens:
                word = token.split("/")[0]
                if word in {"sos", "eos", "unk"}:
                    continue
                total_tokens += 1
                if word not in vectorizer.word2vec:
                    oov_tokens += 1
                    row_oov += 1
            if row_oov > 0:
                captions_with_oov += 1
                if len(examples) < 5:
                    examples.append((text, row_oov))
        stats[level] = {
            "token_total": total_tokens,
            "token_oov_rate": (oov_tokens / total_tokens) if total_tokens else 0.0,
            "caption_oov_rate": (captions_with_oov / len(rows)) if rows else 0.0,
            "pairs": len(rows),
            "oov_examples": examples,
        }
    return stats


def cosine_sims(pairs, processor, wrapper, chunk):
    """按 level 计算 clean vs perturbed TM2T 文本嵌入平均余弦。"""
    result = {}
    for level, rows in pairs.items():
        clean = [str(r["text_orig"]) for r in rows]
        pert = [str(r["text_pair"]) for r in rows]
        sims = []
        for start in range(0, len(clean), chunk):
            clean_emb = processor.encode(wrapper, clean[start:start + chunk])
            pert_emb = processor.encode(wrapper, pert[start:start + chunk])
            sims.append(F.cosine_similarity(clean_emb, pert_emb, dim=-1))
        sim = torch.cat(sims) if sims else torch.zeros(0)
        result[level] = {
            "mean": float(sim.mean()) if sim.numel() else float("nan"),
            "min": float(sim.min()) if sim.numel() else float("nan"),
        }
    return result


def calibration_retrieval(pairs, wrapper, vectorizer, processor,
                          batch_size, levels):
    """caption-locked GT 动作批内检索：clean query 与 perturbed query 各算 R@。

    每个 batch 内：query 编码（clean 或 perturbed）在 GT 动作池里检索，
    对角线为正确匹配（query 对应的原动作），与论文批内协议一致。
    """
    unit_length = int(getattr(wrapper.opt, "unit_length", 4))
    max_text_len = int(getattr(wrapper.opt, "max_text_len", 20))
    report = {}
    for level in levels:
        loader = CaptionLockedDATALoader(
            "t2m", True, batch_size, vectorizer,
            pair_data=pairs[level],
            num_workers=4,
            unit_length=unit_length,
            max_text_len=max_text_len,
            length_policy="strict",
        )
        pair_map = {str(r["motion_id"]): str(r["text_pair"]) for r in pairs[level]}
        clean_counts = np.zeros(3, dtype=np.float64)
        pert_counts = np.zeros(3, dtype=np.float64)
        clean_match = 0.0
        pert_match = 0.0
        samples = 0
        missing = 0
        for batch in tqdm(loader, desc=f"calibrate {level}", leave=False):
            word_embs, pos_ohot, captions, sent_len, motions, m_lens, _, names = batch
            names = [str(n) for n in names]
            text_emb, motion_emb = wrapper.get_co_embeddings(
                word_embs, pos_ohot, sent_len, motions, m_lens)
            # clean query（loader 的 text_orig 就是 clean caption）
            clean_r, clean_m = calculate_R_precision(
                text_emb.cpu().numpy(), motion_emb.cpu().numpy(),
                top_k=3, sum_all=True)
            clean_counts += clean_r
            clean_match += clean_m
            # perturbed query（按 motion_id 取 text_pair）
            pert_texts = []
            for name in names:
                if name in pair_map:
                    pert_texts.append(pair_map[name])
                else:
                    pert_texts.append(None)
                    missing += 1
            valid_idx = [i for i, t in enumerate(pert_texts) if t is not None]
            if valid_idx:
                pert_emb = processor.encode(
                    wrapper, [pert_texts[i] for i in valid_idx])
                pert_r, pert_m = calculate_R_precision(
                    pert_emb.cpu().numpy(),
                    motion_emb[valid_idx].cpu().numpy(),
                    top_k=3, sum_all=True)
                pert_counts += pert_r
                pert_match += pert_m
            samples += len(names)
        n = max(samples - missing, 1)
        report[level] = {
            "samples": samples,
            "missing_pairs": missing,
            "clean_R1": clean_counts[0] / samples,
            "clean_R2": clean_counts[1] / samples,
            "clean_R3": clean_counts[2] / samples,
            "pert_R1": pert_counts[0] / n,
            "pert_R2": pert_counts[1] / n,
            "pert_R3": pert_counts[2] / n,
        }
    return report


def write_report(args, oov, cos, calib, wrapper_opt):
    lines = [
        "# TM2T 尺子扰动敏感性诊断（Phase 0）",
        "",
        f"- Pair 文件: `{args.pair_path}`",
        f"- Levels: {args.levels}",
        f"- 尺子 checkpoint: `checkpoints/t2m/text_mot_match/model/finest.tar`",
        f"- 检索协议: caption-locked GT 动作批内检索，batch size {args.batch_size}",
        "",
        "## OOV 统计",
        "",
        "| Level | 样本对 | token OOV 率 | 含 OOV 句子比例 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for level in args.levels:
        s = oov[level]
        lines.append(
            f"| {level} | {s['pairs']} | {s['token_oov_rate']:.4f} | "
            f"{s['caption_oov_rate']:.4f} |")
    lines.append("")
    lines.append("## clean vs perturbed TM2T 文本嵌入余弦相似度")
    lines.append("")
    lines.append("| Level | mean | min |")
    lines.append("| --- | ---: | ---: |")
    for level in args.levels:
        c = cos[level]
        lines.append(f"| {level} | {c['mean']:.4f} | {c['min']:.4f} |")
    lines.append("")
    lines.append("## 尺子校准 R@（GT 动作批内检索）")
    lines.append("")
    lines.append("| Level | clean R@1 | clean R@2 | clean R@3 | perturbed R@1 | perturbed R@2 | perturbed R@3 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for level in args.levels:
        c = calib[level]
        lines.append(
            f"| {level} | {c['clean_R1']:.4f} | {c['clean_R2']:.4f} | "
            f"{c['clean_R3']:.4f} | {c['pert_R1']:.4f} | {c['pert_R2']:.4f} | "
            f"{c['pert_R3']:.4f} |")
    lines.append("")
    lines.append("## 判定")
    lines.append("")
    for level in args.levels:
        c = calib[level]
        drop = c["clean_R1"] - c["pert_R1"]
        lines.append(
            f"- {level}: perturbed R@1 相对 clean 天花板下降 {drop:.4f}")
    oov_max = max(oov[l]["token_oov_rate"] for l in args.levels)
    if oov_max > 0.15:
        lines.append(
            "- OOV token 率超过 15%：BiGRU 重训无法恢复未登录词，"
            "建议词表扩展或换 CLIP 文本侧（回到决策点）。")
    else:
        lines.append("- OOV token 率在 15% 以内：Phase 1/2 重训文本编码器可行。")
    lines.append("")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(args.pair_path, args.levels, args.limit)
    print("levels:", {k: len(v) for k, v in pairs.items()})

    wrapper, wrapper_opt = build_evaluator(device)
    vectorizer = WordVectorizer("glove", "our_vab")
    processor = TM2TTextProcessor(
        vectorizer,
        max_text_len=int(getattr(wrapper_opt, "max_text_len", 20)),
        alphabetic_only=True,
    )

    print("== OOV ==")
    oov = oov_stats(pairs, processor, vectorizer)
    for level, s in oov.items():
        print(level, f"token_oov={s['token_oov_rate']:.4f}",
              f"caption_oov={s['caption_oov_rate']:.4f}")

    print("== cosine sim ==")
    cos = cosine_sims(pairs, processor, wrapper, args.encode_chunk)
    for level, c in cos.items():
        print(level, f"mean={c['mean']:.4f} min={c['min']:.4f}")

    print("== calibration R@ ==")
    calib = calibration_retrieval(
        pairs, wrapper, vectorizer, processor, args.batch_size, args.levels)
    for level, c in calib.items():
        print(level,
              f"clean R@1={c['clean_R1']:.4f} -> pert R@1={c['pert_R1']:.4f}")

    write_report(args, oov, cos, calib, wrapper_opt)


if __name__ == "__main__":
    main()