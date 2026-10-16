"""
SATO++ Evaluation Module

新增评测指标:
- FID_P: 扰动文本生成 vs 真实 motion
- FID_D: 原始文本生成 vs 扰动文本生成
- RepCos: preserving pair 的表示余弦相似度
- JSD: 扰动前后注意力散度
- Slot Attribution Shift: altering 时被编辑 slot 的归因变化
"""

import hashlib
import json

import torch
import torch.nn.functional as F
import numpy as np
import random
import re
from collections import defaultdict
from pathlib import Path
from scipy import linalg
from tqdm import tqdm


def _require_clip_module():
    """Load OpenAI CLIP only for T2M-GPT generation paths that need it."""
    try:
        import clip
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI CLIP is required for SATO++ T2M-GPT motion generation"
        ) from exc
    return clip


# ========== FID 计算 ==========

def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


# ========== JSD ==========

def compute_jsd(p, q):
    """
    Jensen-Shannon Divergence between two probability distributions.

    Args:
        p, q: (bs, n) probability distributions (已经 softmax)

    Returns:
        scalar JSD
    """
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(torch.log(m + 1e-8), p, reduction='batchmean')
    kl_qm = F.kl_div(torch.log(m + 1e-8), q, reduction='batchmean')
    return 0.5 * (kl_pm + kl_qm)


# ========== R-precision / Diversity / Matching ==========

def euclidean_distance_matrix(matrix1, matrix2):
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    return np.sqrt(d1 + d2 + d3)


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    return np.concatenate(top_k_list, axis=1)


def calculate_R_precision(embedding1, embedding2, top_k=3, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    return top_k_mat, matching_score


class TM2TTextProcessor:
    """Convert raw captions into standard TM2T evaluator text embeddings.

    HumanML3D preprocessing mirrors the reference ``process_text`` contract:
    strip surrounding whitespace, remove hyphens, retain alphabetic tokens,
    and lemmatize nouns and verbs except the literal lowercase word ``left``.
    """

    def __init__(self, word_vectorizer, max_text_len=20,
                 spacy_model='en_core_web_sm', alphabetic_only=True, nlp=None):
        self.word_vectorizer = word_vectorizer
        self.max_text_len = int(max_text_len)
        self.spacy_model = spacy_model
        self.alphabetic_only = bool(alphabetic_only)
        self._token_cache = {}
        if nlp is None:
            try:
                import spacy
                nlp = spacy.load(spacy_model, disable=['parser', 'ner'])
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    f"spaCy model {spacy_model!r} is required for perturbed-text "
                    "TM2T retrieval evaluation"
                ) from exc
        self.nlp = nlp

    def tokenize(self, caption):
        caption = str(caption).strip().replace('-', '')
        cached = self._token_cache.get(caption)
        if cached is not None:
            return list(cached)

        tokens = []
        for token in self.nlp(caption):
            word = token.text
            if self.alphabetic_only:
                if not word.isalpha():
                    continue
            elif token.is_space or token.is_punct:
                continue
            if token.pos_ in {'NOUN', 'VERB'} and word != 'left' and token.lemma_:
                word = token.lemma_
            tokens.append(f"{word}/{token.pos_ or 'OTHER'}")
        self._token_cache[caption] = tuple(tokens)
        return list(tokens)

    def vectorize(self, captions):
        word_rows = []
        pos_rows = []
        lengths = []
        for caption in captions:
            tokens = self.tokenize(caption)
            if len(tokens) < self.max_text_len:
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                lengths.append(len(tokens))
                tokens += ['unk/OTHER'] * (self.max_text_len + 2 - len(tokens))
            else:
                tokens = [
                    'sos/OTHER', *tokens[:self.max_text_len], 'eos/OTHER'
                ]
                lengths.append(len(tokens))

            word_vectors = []
            pos_vectors = []
            for token in tokens:
                word_vector, pos_vector = self.word_vectorizer[token]
                word_vectors.append(word_vector)
                pos_vectors.append(pos_vector)
            word_rows.append(np.stack(word_vectors))
            pos_rows.append(np.stack(pos_vectors))

        return (
            torch.from_numpy(np.stack(word_rows)).float(),
            torch.from_numpy(np.stack(pos_rows)).float(),
            torch.tensor(lengths, dtype=torch.long),
        )

    def encode(self, eval_wrapper, captions):
        """Encode captions and restore their input order after length sorting."""
        if not captions:
            raise ValueError("TM2T text encoding requires at least one caption")
        word_embeddings, pos_one_hots, sent_lens = self.vectorize(captions)
        order = torch.tensor(
            sorted(
                range(len(captions)),
                key=lambda index: int(sent_lens[index]),
                reverse=True,
            ),
            dtype=torch.long,
        )
        device = eval_wrapper.device
        with torch.no_grad():
            sorted_embeddings = eval_wrapper.text_encoder(
                word_embeddings[order].to(device),
                pos_one_hots[order].to(device),
                sent_lens[order].to(device),
            )
        embeddings = torch.empty_like(sorted_embeddings)
        embeddings[order.to(sorted_embeddings.device)] = sorted_embeddings
        return embeddings


def calculate_diversity(activation, diversity_times):
    assert activation.shape[0] > diversity_times
    first = np.random.choice(activation.shape[0], diversity_times, replace=False)
    second = np.random.choice(activation.shape[0], diversity_times, replace=False)
    return np.linalg.norm(activation[first] - activation[second], axis=1).mean()


def _evaluation_diversity(activation):
    diversity_times = (
        300 if activation.shape[0] > 300
        else min(100, activation.shape[0] - 1)
    )
    if diversity_times < 1:
        raise ValueError("Diversity requires at least two motion samples")
    return calculate_diversity(activation, diversity_times)


# ========== RepCos ==========

def compute_repcos(u_orig, u_pair):
    """
    Preserving pair 的表示余弦相似度。

    Args:
        u_orig: (N, dim) 原始文本的主语义表示
        u_pair: (N, dim) pair 文本的主语义表示

    Returns:
        mean cosine similarity
    """
    return F.cosine_similarity(u_orig, u_pair, dim=-1).mean().item()


# ========== Slot Attribution Metrics ==========

def compute_slot_attribution_shift(alpha_orig, alpha_pert, edited_mask):
    """
    计算 altering pair 的 slot-level 归因变化指标。

    Args:
        alpha_orig: (N, L) 原句归因
        alpha_pert: (N, L) 扰动句归因
        edited_mask: (N, L) bool

    Returns:
        dict with:
        - edited_shift: 被编辑 slot 的平均归因变化
        - non_edited_shift: 未编辑 slot 的平均归因变化
        - specificity: edited_shift / (edited_shift + non_edited_shift + eps)
    """
    diff = (alpha_orig - alpha_pert).abs()

    edited_shift = diff[edited_mask].mean().item() if edited_mask.any() else 0.0
    non_edited_shift = diff[~edited_mask].mean().item() if (~edited_mask).any() else 0.0

    specificity = edited_shift / (edited_shift + non_edited_shift + 1e-8)

    return {
        'edited_shift': edited_shift,
        'non_edited_shift': non_edited_shift,
        'specificity': specificity,
    }


# ========== 综合评测函数 ==========

def _normalize_caption(text):
    return " ".join(str(text).strip().lower().split())


def _extract_motion_id(name):
    """Remove only the legacy random segment prefix; preserve arbitrary IDs."""
    value = str(name)
    # HumanML3D segment samples use a single random A--W prefix for both
    # numeric IDs and IDs such as M013514.  The pair index stores the suffix.
    match = re.fullmatch(r"[A-W]_(.+)", value)
    return match.group(1) if match else value


def build_pair_index(pair_data):
    """Build index keyed by motion_id instead of text content."""
    pair_index = {}  # motion_id -> text_pair (single entry per motion_id)
    for item in pair_data or []:
        mid = item.get('motion_id', '')
        text_pair = item.get('text_pair', '')
        if mid and text_pair:
            pair_index[mid] = text_pair
    return pair_index


def build_caption_pair_index(pair_data):
    """Build an ordered pair index keyed by motion id and source caption."""
    pair_index = {}
    for item in pair_data or []:
        motion_id = str(item.get('motion_id', '')).strip()
        text_orig = item.get('text_orig', '')
        text_pair = item.get('text_pair', '')
        if not motion_id or not text_orig or not text_pair:
            continue
        key = (motion_id, _normalize_caption(text_orig))
        pair_index.setdefault(key, []).append(text_pair)
    return pair_index


def sample_perturbed_texts(motion_names, pair_index, rng=None):
    rng = rng or random
    pert_texts = []
    matched = 0
    for name in motion_names:
        mid = _extract_motion_id(name)
        if mid in pair_index:
            pert_texts.append(pair_index[mid])
            matched += 1
        else:
            # fallback: will use original text (caller handles this)
            pert_texts.append(None)
    return pert_texts, matched


def sample_caption_locked_perturbed_texts(motion_names, captions, pair_index,
                                          pair_offsets):
    """Resolve perturbations by the same key used by the caption-locked loader."""
    pert_texts = []
    matched = 0
    for name, caption in zip(motion_names, captions):
        key = (_extract_motion_id(name), _normalize_caption(caption))
        candidates = pair_index.get(key)
        if not candidates:
            pert_texts.append(None)
            continue
        offset = pair_offsets.get(key, 0)
        pert_texts.append(candidates[offset % len(candidates)])
        pair_offsets[key] = offset + 1
        matched += 1
    return pert_texts, matched


def _cfg_is_enabled(cfg_scale):
    return abs(float(cfg_scale) - 1.0) > 1e-8


def _cfg_guided_logits(trans_encoder, idxs, feat_clip, cfg_scale=1.0,
                       uncond_feat=None):
    if not _cfg_is_enabled(cfg_scale):
        return trans_encoder.forward(idxs, feat_clip)[:, -1, :]

    if uncond_feat is None:
        uncond_feat = torch.zeros_like(feat_clip)

    if len(idxs) == 0:
        logits = trans_encoder.forward(
            [],
            torch.cat([uncond_feat, feat_clip], dim=0),
        )[:, -1, :]
    else:
        logits = trans_encoder.forward(
            torch.cat([idxs, idxs], dim=0),
            torch.cat([uncond_feat, feat_clip], dim=0),
        )[:, -1, :]
    logits_uncond, logits_cond = logits.chunk(2, dim=0)
    return logits_uncond + float(cfg_scale) * (logits_cond - logits_uncond)


def _make_cfg_uncond_feature(clip_model, batch_size, reference_feat,
                             cfg_scale=1.0, cfg_uncond="zero"):
    if not _cfg_is_enabled(cfg_scale):
        return None
    if cfg_uncond == "zero":
        return torch.zeros_like(reference_feat)
    if cfg_uncond == "empty":
        text = _require_clip_module().tokenize(
            [""] * batch_size, truncate=True
        ).to(reference_feat.device)
        return clip_model.encode_text(text).float()
    raise ValueError(f"Unknown cfg_uncond: {cfg_uncond}")


def _filter_logits(logits, sample_temperature=1.0, top_k=0, top_p=1.0):
    temperature = max(float(sample_temperature), 1e-8)
    logits = logits / temperature

    top_k = int(top_k or 0)
    if top_k > 0 and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    top_p = float(top_p)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    return logits


def _custom_sampling_is_enabled(sample_temperature=1.0, top_k=0, top_p=1.0):
    return (
        abs(float(sample_temperature) - 1.0) > 1e-8
        or int(top_k or 0) > 0
        or float(top_p) < 1.0
    )


def _sample_motion_indices_batch(trans_encoder, feat_clip, sample_categorical=True,
                                 cfg_scale=1.0, uncond_feat=None,
                                 sample_temperature=1.0, top_k=0, top_p=1.0):
    bs = feat_clip.shape[0]
    xs = torch.zeros((bs, 0), dtype=torch.long, device=feat_clip.device)
    code_lens = torch.zeros(bs, dtype=torch.long, device=feat_clip.device)
    finished = torch.zeros(bs, dtype=torch.bool, device=feat_clip.device)
    max_steps = max(1, trans_encoder.block_size - 1)

    for k in range(max_steps):
        logits = _cfg_guided_logits(
            trans_encoder,
            [] if k == 0 else xs,
            feat_clip,
            cfg_scale=cfg_scale,
            uncond_feat=uncond_feat,
        )
        logits = _filter_logits(
            logits,
            sample_temperature=sample_temperature,
            top_k=top_k,
            top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if sample_categorical:
            idx = torch.distributions.Categorical(probs).sample().unsqueeze(-1)
        else:
            _, idx = torch.topk(probs, k=1, dim=-1)

        active = ~finished
        eos = idx.squeeze(-1) == trans_encoder.num_vq
        valid = active & ~eos
        append_idx = idx.clone()
        append_idx[~valid] = 0
        xs = torch.cat((xs, append_idx), dim=1)
        code_lens += valid.long()
        finished |= active & eos
        if finished.all():
            break

    empty = code_lens == 0
    if empty.any():
        xs[empty, 0] = 1
        code_lens[empty] = 1

    max_len = int(code_lens.max().item())
    return xs[:, :max_len], code_lens


def _decode_motion_indices_batch(net, code_idx):
    vq = net.vqvae
    x_d = vq.quantizer.dequantize(code_idx)
    x_d = x_d.permute(0, 2, 1).contiguous()
    x_decoder = vq.decoder(x_d)
    return vq.postprocess(x_decoder)


def _decode_motion_indices_by_length(net, code_idx, code_lens, seq, dim_pose):
    """Decode batched code indices without letting pad tokens affect Conv1d output.

    The original single-sample path decodes each generated sequence at its own
    token length. Decoding a padded batch at max length is faster, but decoder
    convolutions can leak pad-code context into valid frames near sequence ends.
    Grouping by exact token length keeps the decoder inputs equivalent to the
    single-sample path while still batching samples that share a length.
    """
    device = code_idx.device
    bs = code_idx.shape[0]
    pred_pose_eval = torch.zeros((bs, seq, dim_pose), device=device)
    pred_len = torch.ones(bs, dtype=torch.long)

    for length in torch.unique(code_lens.cpu(), sorted=True).tolist():
        length = int(length)
        if length <= 0:
            continue

        rows = (code_lens == length)
        rows_idx = rows.nonzero(as_tuple=False).squeeze(-1)
        codes = code_idx[rows, :length].contiguous()
        pred_pose = _decode_motion_indices_batch(net, codes)
        cur_len = pred_pose.shape[1]
        copy_len = min(cur_len, seq)
        pred_pose_eval[rows_idx, :copy_len] = pred_pose[:, :copy_len]
        pred_len[rows_idx.cpu()] = copy_len

    return pred_pose_eval, pred_len


def _decode_motion_indices_loop(net, code_idx, code_lens, seq, dim_pose):
    """Decode sampled code indices with the same per-sample path as legacy eval."""
    device = code_idx.device
    bs = code_idx.shape[0]
    pred_pose_eval = torch.zeros((bs, seq, dim_pose), device=device)
    pred_len = torch.ones(bs, dtype=torch.long)

    for k in range(bs):
        length = int(code_lens[k].item())
        if length <= 0:
            cur_code = torch.ones(1, 1, device=device).long()
        else:
            cur_code = code_idx[k:k + 1, :length].contiguous()
        pred_pose = net.forward_decoder(cur_code)
        cur_len = pred_pose.shape[1]
        copy_len = min(cur_len, seq)
        pred_len[k] = copy_len
        pred_pose_eval[k:k + 1, :copy_len] = pred_pose[:, :copy_len]

    return pred_pose_eval, pred_len


def _generate_motion_batch_loop(net, trans_encoder, feat_clip, seq, dim_pose,
                                sample_categorical=True, cfg_scale=1.0,
                                uncond_feat=None, sample_temperature=1.0,
                                top_k=0, top_p=1.0):
    bs = feat_clip.shape[0]
    pred_pose_eval = torch.zeros((bs, seq, dim_pose), device=feat_clip.device)
    pred_len = torch.ones(bs).long()

    for k in range(bs):
        try:
            if _cfg_is_enabled(cfg_scale) or _custom_sampling_is_enabled(
                    sample_temperature, top_k, top_p):
                cur_uncond = None if uncond_feat is None else uncond_feat[k:k + 1]
                index_motion, code_lens = _sample_motion_indices_batch(
                    trans_encoder,
                    feat_clip[k:k + 1],
                    sample_categorical,
                    cfg_scale=cfg_scale,
                    uncond_feat=cur_uncond,
                    sample_temperature=sample_temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                index_motion = index_motion[:, :int(code_lens[0].item())]
            else:
                index_motion = trans_encoder.sample(
                    feat_clip[k:k+1], sample_categorical)
        except Exception:
            index_motion = torch.ones(1, 1, device=feat_clip.device).long()
        pred_pose = net.forward_decoder(index_motion)
        cur_len = pred_pose.shape[1]
        copy_len = min(cur_len, seq)
        pred_len[k] = copy_len
        pred_pose_eval[k:k+1, :copy_len] = pred_pose[:, :copy_len]

    return pred_pose_eval, pred_len


def generate_motion_batch(net, trans_encoder, feat_clip, seq, dim_pose,
                          sample_categorical=True, use_fast_batch=False,
                          cfg_scale=1.0, uncond_feat=None,
                          sample_temperature=1.0, top_k=0, top_p=1.0,
                          return_codes=False, strict_fast_path=False):
    if not use_fast_batch:
        if return_codes:
            raise RuntimeError(
                "motion-token generation audit requires the deterministic fast-batch path"
            )
        return _generate_motion_batch_loop(
            net, trans_encoder, feat_clip, seq, dim_pose,
            sample_categorical=sample_categorical,
            cfg_scale=cfg_scale,
            uncond_feat=uncond_feat,
            sample_temperature=sample_temperature,
            top_k=top_k,
            top_p=top_p)

    bs = feat_clip.shape[0]
    try:
        code_idx, code_lens = _sample_motion_indices_batch(
            trans_encoder,
            feat_clip,
            sample_categorical,
            cfg_scale=cfg_scale,
            uncond_feat=uncond_feat,
            sample_temperature=sample_temperature,
            top_k=top_k,
            top_p=top_p)
        pred_pose_eval, pred_len = _decode_motion_indices_loop(
            net, code_idx, code_lens, seq, dim_pose)
        if return_codes:
            codes = [
                code_idx[row, :int(code_lens[row].item())]
                .detach().cpu().tolist()
                for row in range(bs)
            ]
            return pred_pose_eval, pred_len, codes
        return pred_pose_eval, pred_len
    except Exception:
        if strict_fast_path or return_codes:
            raise
        return _generate_motion_batch_loop(
            net, trans_encoder, feat_clip, seq, dim_pose,
            sample_categorical=sample_categorical,
            cfg_scale=cfg_scale,
            uncond_feat=uncond_feat,
            sample_temperature=sample_temperature,
            top_k=top_k,
            top_p=top_p)

@torch.no_grad()
def evaluation_sato_pp(out_dir, val_loader, net, trans_encoder,
                       clip_model, eval_wrapper, logger, writer, nb_iter,
                       pair_data=None, attr_pooling=None, tau=1.0,
                       best_fid_p=1000, best_fid_d=1000,
                       sample_categorical=True, use_fast_batch=False,
                       cfg_scale=1.0, cfg_uncond="zero",
                       sample_temperature=1.0, top_k=0, top_p=1.0,
                       sampling_seed=None, caption_locked=False,
                       collect_sample_index=False,
                       diagnostics_output=None,
                       retrieval_embedding_output=None,
                       retrieval_protocol_fingerprint=None,
                       retrieval_sample_index_sha256=None,
                       retrieval_seed=None,
                       retrieval_batch_size=None,
                       generation_audit_output=None,
                       generation_audit_seed=None, max_samples=None,
                       sampling_pair_rng_mode='common',
                       expected_sample_index=None,
                       perturbed_text_processor=None):
    """
    SATO++ 综合评测: 原始 FID + FID_P + FID_D + RepCos + JSD

    Args:
        pair_data: list of dict, preserving pair 数据用于测 RepCos
        attr_pooling: AttributionPooling 模块
        tau: 归因温度
        perturbed_text_processor: 使用标准 TM2T GloVe+POS 文本编码器处理扰动 caption；
            未提供时按 evaluator 数据集自动构造。

    Returns:
        updated best metrics
    """
    trans_encoder.eval()
    cfg_scale = float(cfg_scale)

    motion_annotation_list = []
    motion_orig_list = []
    motion_pert_list = []

    # R-precision / Matching accumulators (per-batch)
    R_precision_real = 0
    R_precision = 0
    R_precision_pert = np.zeros(3, dtype=np.float64)
    matching_score_real = 0
    matching_score_pred = 0
    matching_score_pert = 0
    perturbed_retrieval_samples = 0
    nb_sample = 0

    attn_jsd_list = []
    repcos_list = []
    pair_index = (
        build_caption_pair_index(pair_data)
        if caption_locked else build_pair_index(pair_data))
    if pair_index and perturbed_text_processor is None:
        from utils.word_vectorizer import WordVectorizer
        repo_root = Path(__file__).resolve().parents[1]
        source_name = str(getattr(
            eval_wrapper.opt,
            'source_dataset_name',
            getattr(eval_wrapper.opt, 'dataset_name', 't2m'),
        ))
        perturbed_text_processor = TM2TTextProcessor(
            WordVectorizer(str(repo_root / 'glove'), 'our_vab'),
            max_text_len=int(getattr(eval_wrapper.opt, 'max_text_len', 20)),
            alphabetic_only=not source_name.startswith('motionx322'),
        )
    pair_offsets = {}
    matched_pairs = 0
    total_pairs = 0
    sample_index = []
    expected_index_offset = 0
    diagnostic_rows = []
    diagnostic_text_embeddings = []
    diagnostic_perturbed_text_embeddings = []
    diagnostic_original_motions = []
    collect_retrieval_embeddings = (
        diagnostics_output is not None or retrieval_embedding_output is not None
    )
    retrieval_batch_sizes = []
    generation_audit = (
        {"original": [], "perturbed": []}
        if generation_audit_output is not None else None
    )
    if generation_audit is not None and not use_fast_batch:
        raise RuntimeError(
            "generation audit requires use_fast_batch=True so sampled motion tokens "
            "can be recorded without changing the generation path"
        )
    if generation_audit is not None and not collect_sample_index:
        raise RuntimeError("generation audit requires collect_sample_index=True")
    pair_level_index = {
        str(item.get('motion_id', '')).strip(): item.get('level')
        for item in (pair_data or [])
        if str(item.get('motion_id', '')).strip()
    }

    total_batches = len(val_loader) if hasattr(val_loader, "__len__") else None
    progress = tqdm(
        val_loader,
        total=total_batches,
        desc=f"SATO++ eval iter {nb_iter}",
        dynamic_ncols=True,
        leave=False,
    )

    for batch_idx, batch in enumerate(progress, start=1):
        if max_samples is not None:
            remaining = int(max_samples) - nb_sample
            if remaining <= 0:
                break
            if remaining < len(batch[2]):
                batch = tuple(value[:remaining] for value in batch)
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]

        # ---- Original text generation ----
        text = _require_clip_module().tokenize(
            clip_text, truncate=True
        ).cuda()
        feat_clip = clip_model.encode_text(text).float()
        uncond_feat = _make_cfg_uncond_feature(
            clip_model, bs, feat_clip, cfg_scale=cfg_scale,
            cfg_uncond=cfg_uncond)

        if sampling_seed is not None:
            batch_seed = int(sampling_seed) + (batch_idx - 1) * 1_000_003
            torch.manual_seed(batch_seed)
            torch.cuda.manual_seed_all(batch_seed)

        original_generated = generate_motion_batch(
            net, trans_encoder, feat_clip, seq, pose.shape[-1],
            sample_categorical=sample_categorical,
            use_fast_batch=use_fast_batch,
            cfg_scale=cfg_scale,
            uncond_feat=uncond_feat,
            sample_temperature=sample_temperature,
            top_k=top_k,
            top_p=top_p,
            return_codes=generation_audit is not None,
            strict_fast_path=generation_audit is not None)
        if generation_audit is not None:
            pred_pose_eval, pred_len, original_codes = original_generated
        else:
            pred_pose_eval, pred_len = original_generated
            original_codes = None

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
        if diagnostics_output is not None:
            diagnostic_text_embeddings.append(et_pred.cpu())
            diagnostic_original_motions.append(pred_pose_eval.cpu())

        pose = pose.cuda().float()
        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length)

        motion_annotation_list.append(em)
        motion_orig_list.append(em_pred)

        # R-precision per batch (original text only)
        temp_R, temp_match = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(
            et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match
        nb_sample += bs

        # ---- Perturbed text generation (if pair data available) ----
        if pair_index:
            # 按 motion_id 匹配扰动文本；没有匹配时回退到原文本。
            if caption_locked:
                pert_texts, batch_matched = sample_caption_locked_perturbed_texts(
                    name, clip_text, pair_index, pair_offsets)
            else:
                pert_texts, batch_matched = sample_perturbed_texts(
                    name, pair_index)
            matched_pairs += batch_matched
            total_pairs += bs

            # fallback to original text for unmatched
            pert_texts = [p if p is not None else orig for p, orig in zip(pert_texts, clip_text)]

            if collect_sample_index:
                batch_sample_index = [{
                    'loader_name': str(sample_name),
                    'motion_id': _extract_motion_id(sample_name),
                    'caption': str(original_text),
                    'perturbed_caption': str(perturbed_text),
                    'level': pair_level_index.get(_extract_motion_id(sample_name)),
                    'motion_length': int(sample_length),
                } for sample_name, original_text, perturbed_text, sample_length in zip(
                    name, clip_text, pert_texts, m_length)]
                if expected_sample_index is not None:
                    expected_batch = expected_sample_index[
                        expected_index_offset:expected_index_offset + len(batch_sample_index)
                    ]
                    if len(expected_batch) != len(batch_sample_index):
                        raise RuntimeError(
                            "frozen sample index ended before the evaluator loader"
                        )
                    for actual, expected in zip(batch_sample_index, expected_batch):
                        # The frozen record is authoritative for the perturbation
                        # level. The legacy evaluator's motion-id-only level map is
                        # ambiguous when one motion has multiple caption pairs.
                        actual['level'] = expected.get('level')
                    for local_idx, (actual, expected) in enumerate(
                            zip(batch_sample_index, expected_batch)):
                        absolute_idx = expected_index_offset + local_idx
                        checks = {
                            'loader_name': str(expected['loader_name']),
                            'motion_id': str(expected['motion_id']),
                            'caption': str(expected['caption']),
                            'perturbed_caption': str(expected['perturbed_caption']),
                            'level': expected.get('level'),
                            'motion_length': int(expected['motion_length']),
                        }
                        if actual != checks:
                            raise RuntimeError(
                                "frozen sample-index mismatch at row "
                                f"{absolute_idx}: actual={actual!r}, expected={checks!r}"
                            )
                    sample_index.extend(dict(row) for row in expected_batch)
                    expected_index_offset += len(batch_sample_index)
                else:
                    sample_index.extend(batch_sample_index)

            text_pert = _require_clip_module().tokenize(
                pert_texts, truncate=True
            ).cuda()
            feat_pert = clip_model.encode_text(text_pert).float()

            if sampling_seed is not None and sampling_pair_rng_mode == 'common':
                torch.manual_seed(batch_seed)
                torch.cuda.manual_seed_all(batch_seed)

            perturbed_generated = generate_motion_batch(
                net, trans_encoder, feat_pert, seq, pose.shape[-1],
                sample_categorical=sample_categorical,
                use_fast_batch=use_fast_batch,
                cfg_scale=cfg_scale,
                uncond_feat=uncond_feat,
                sample_temperature=sample_temperature,
                top_k=top_k,
                top_p=top_p,
                return_codes=generation_audit is not None,
                strict_fast_path=generation_audit is not None)
            if generation_audit is not None:
                pred_pert_eval, pred_pert_len, perturbed_codes = perturbed_generated
            else:
                pred_pert_eval, pred_pert_len = perturbed_generated
                perturbed_codes = None

            if generation_audit is not None:
                if original_codes is None or perturbed_codes is None:
                    raise RuntimeError("generation audit did not receive sampled motion tokens")
                audit_samples = sample_index[-bs:]
                if len(audit_samples) != bs:
                    raise RuntimeError("generation audit sample-index batch is incomplete")
                batch_start = nb_sample - bs
                for local_idx, sample in enumerate(audit_samples):
                    identity = {
                        'sample_offset': batch_start + local_idx,
                        'loader_name': str(sample['loader_name']),
                        'motion_id': str(sample['motion_id']),
                        'level': str(sample.get('level')),
                        'reference_frame_count': int(m_length[local_idx]),
                    }
                    original_tokens = [int(value) for value in original_codes[local_idx]]
                    perturbed_tokens = [int(value) for value in perturbed_codes[local_idx]]
                    generation_audit['original'].append({
                        **identity,
                        'generation_role': 'original',
                        'input_text': str(clip_text[local_idx]),
                        'motion_tokens': original_tokens,
                        'motion_token_count': len(original_tokens),
                        'generated_frame_count': int(pred_len[local_idx]),
                        'evaluator_frame_count': int(pred_len[local_idx]),
                    })
                    generation_audit['perturbed'].append({
                        **identity,
                        'generation_role': 'perturbed',
                        'input_text': str(pert_texts[local_idx]),
                        'motion_tokens': perturbed_tokens,
                        'motion_token_count': len(perturbed_tokens),
                        'generated_frame_count': int(pred_pert_len[local_idx]),
                        'evaluator_frame_count': int(pred_pert_len[local_idx]),
                    })

            _, em_pert = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len,
                pred_pert_eval, pred_pert_len)
            perturbed_text_embedding = perturbed_text_processor.encode(
                eval_wrapper, pert_texts)
            temp_R_pert, temp_match_pert = calculate_R_precision(
                perturbed_text_embedding.cpu().numpy(),
                em_pert.cpu().numpy(),
                top_k=3,
                sum_all=True,
            )
            R_precision_pert += temp_R_pert
            matching_score_pert += temp_match_pert
            perturbed_retrieval_samples += bs
            motion_pert_list.append(em_pert)
            if collect_retrieval_embeddings:
                diagnostic_perturbed_text_embeddings.append(
                    perturbed_text_embedding.cpu())
                retrieval_batch_sizes.append(bs)
            if diagnostics_output is not None:
                diagnostic_rows.extend({
                    'loader_name': str(sample_name),
                    'motion_id': _extract_motion_id(sample_name),
                    'caption': str(original_text),
                    'perturbed_caption': str(perturbed_text),
                    'level': pair_level_index.get(_extract_motion_id(sample_name)),
                    'reference_length': int(reference_length),
                    'original_length': int(original_length),
                    'perturbed_length': int(perturbed_length),
                } for (
                    sample_name,
                    original_text,
                    perturbed_text,
                    reference_length,
                    original_length,
                    perturbed_length,
                ) in zip(
                    name, clip_text, pert_texts, m_length, pred_len, pred_pert_len
                ))

        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            postfix = {
                "samples": nb_sample,
                "mode": "fast" if use_fast_batch else "loop",
            }
            if total_pairs:
                postfix["pairs"] = f"{matched_pairs}/{total_pairs}"
            progress.set_postfix(postfix)

    # ---- Compute FID ----
    motion_ann_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_orig_np = torch.cat(motion_orig_list, dim=0).cpu().numpy()

    gt_mu, gt_cov = calculate_activation_statistics(motion_ann_np)
    orig_mu, orig_cov = calculate_activation_statistics(motion_orig_np)

    fid = calculate_frechet_distance(gt_mu, gt_cov, orig_mu, orig_cov)

    fid_p = 0.0
    fid_d = 0.0
    if motion_pert_list:
        motion_pert_np = torch.cat(motion_pert_list, dim=0).cpu().numpy()
        pert_mu, pert_cov = calculate_activation_statistics(motion_pert_np)
        fid_p = calculate_frechet_distance(gt_mu, gt_cov, pert_mu, pert_cov)
        fid_d = calculate_frechet_distance(orig_mu, orig_cov, pert_mu, pert_cov)

    # ---- Compute R-precision, Diversity, Matching ----
    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    if perturbed_retrieval_samples:
        if perturbed_retrieval_samples != nb_sample:
            raise RuntimeError(
                "Perturbed-text retrieval covered "
                f"{perturbed_retrieval_samples}/{nb_sample} generated motions"
            )
        R_precision_pert = R_precision_pert / perturbed_retrieval_samples
        matching_score_pert = matching_score_pert / perturbed_retrieval_samples
    else:
        R_precision_pert = np.zeros(3, dtype=np.float64)
        matching_score_pert = 0.0

    diversity_real = _evaluation_diversity(motion_ann_np)
    diversity = _evaluation_diversity(motion_orig_np)

    # ---- Logging ----
    msg = (f"Eva. Iter {nb_iter}: FID {fid:.4f}, "
           f"FID_P {fid_p:.4f}, FID_D {fid_d:.4f}, "
           f"Diversity Real {diversity_real:.4f}, Diversity {diversity:.4f}, "
           f"TOP1 {R_precision[0]:.4f}, TOP2 {R_precision[1]:.4f}, TOP3 {R_precision[2]:.4f}, "
           f"Matching {matching_score_pred:.4f}, "
           f"TOP1_P {R_precision_pert[0]:.4f}, "
           f"TOP2_P {R_precision_pert[1]:.4f}, "
           f"TOP3_P {R_precision_pert[2]:.4f}, "
           f"Matching_P {matching_score_pert:.4f}")
    if total_pairs:
        msg += f", PerturbedMatched {matched_pairs}/{total_pairs}"
    if _cfg_is_enabled(cfg_scale):
        msg += f", CFG {cfg_scale:.3g}/{cfg_uncond}"
    if (
        not sample_categorical
        or _custom_sampling_is_enabled(sample_temperature, top_k, top_p)
    ):
        mode = "categorical" if sample_categorical else "greedy"
        msg += (
            f", Sampling {mode}/temp={float(sample_temperature):.3g}"
            f"/top_k={int(top_k or 0)}/top_p={float(top_p):.3g}"
        )
    if repcos_list:
        msg += f", RepCos {np.mean(repcos_list):.4f}"
    if attn_jsd_list:
        msg += f", JSD {np.mean(attn_jsd_list):.4f}"
    logger.info(msg)

    writer.add_scalar('./SATO_PP/FID', fid, nb_iter)
    writer.add_scalar('./SATO_PP/FID_P', fid_p, nb_iter)
    writer.add_scalar('./SATO_PP/FID_D', fid_d, nb_iter)
    writer.add_scalar('./SATO_PP/Diversity', diversity, nb_iter)
    writer.add_scalar('./SATO_PP/TOP1', R_precision[0], nb_iter)
    writer.add_scalar('./SATO_PP/TOP2', R_precision[1], nb_iter)
    writer.add_scalar('./SATO_PP/TOP3', R_precision[2], nb_iter)
    writer.add_scalar('./SATO_PP/Matching', matching_score_pred, nb_iter)
    writer.add_scalar('./SATO_PP/TOP1_PERT', R_precision_pert[0], nb_iter)
    writer.add_scalar('./SATO_PP/TOP2_PERT', R_precision_pert[1], nb_iter)
    writer.add_scalar('./SATO_PP/TOP3_PERT', R_precision_pert[2], nb_iter)
    writer.add_scalar('./SATO_PP/Matching_PERT', matching_score_pert, nb_iter)

    if fid < best_fid_p:
        best_fid_p = fid
    if fid_d < best_fid_d:
        best_fid_d = fid_d

    metrics = {
        'fid': float(fid),
        'fid_p': float(fid_p),
        'fid_d': float(fid_d),
        'diversity_real': float(diversity_real),
        'diversity': float(diversity),
        'top1': float(R_precision[0]),
        'top2': float(R_precision[1]),
        'top3': float(R_precision[2]),
        'matching': float(matching_score_pred),
        'top1_pert': float(R_precision_pert[0]),
        'top2_pert': float(R_precision_pert[1]),
        'top3_pert': float(R_precision_pert[2]),
        'matching_pert': float(matching_score_pert),
        'perturbed_retrieval_samples': int(perturbed_retrieval_samples),
        'matched_pairs': int(matched_pairs),
        'total_pairs': int(total_pairs),
    }
    if collect_sample_index:
        if (
            expected_sample_index is not None
            and expected_index_offset != len(expected_sample_index)
        ):
            raise RuntimeError(
                "frozen sample index was not fully consumed: "
                f"{expected_index_offset}/{len(expected_sample_index)}"
            )
        encoded_index = json.dumps(
            sample_index, ensure_ascii=False, sort_keys=True,
            separators=(',', ':')).encode('utf-8')
        metrics['sample_index'] = sample_index
        metrics['sample_index_sha256'] = hashlib.sha256(encoded_index).hexdigest()

    if retrieval_embedding_output is not None:
        if not collect_sample_index or len(sample_index) != nb_sample:
            raise RuntimeError(
                "retrieval embedding output requires a complete sample index"
            )
        if (
            not motion_pert_list
            or len(diagnostic_perturbed_text_embeddings) != len(retrieval_batch_sizes)
            or sum(retrieval_batch_sizes) != nb_sample
        ):
            raise RuntimeError(
                "retrieval embedding output does not cover every evaluator batch"
            )
        sample_hash = str(metrics.get('sample_index_sha256', ''))
        if (
            retrieval_sample_index_sha256 is not None
            and sample_hash != str(retrieval_sample_index_sha256)
        ):
            raise RuntimeError(
                "retrieval embedding sample-index checksum differs from the frozen index"
            )
        if not retrieval_protocol_fingerprint:
            raise RuntimeError(
                "retrieval embedding output requires a protocol fingerprint"
            )
        batch_size = int(
            retrieval_batch_size or getattr(val_loader, 'batch_size', 0)
        )
        if batch_size <= 0:
            raise RuntimeError("retrieval embedding output requires a batch size")
        retrieval_path = Path(retrieval_embedding_output)
        retrieval_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            retrieval_path,
            perturbed_text_embedding=torch.cat(
                diagnostic_perturbed_text_embeddings, dim=0).numpy(),
            perturbed_motion_embedding=motion_pert_np,
            batch_sizes=np.asarray(retrieval_batch_sizes, dtype=np.int64),
            loader_name=np.asarray([
                str(row['loader_name']) for row in sample_index
            ]),
            motion_id=np.asarray([
                str(row['motion_id']) for row in sample_index
            ]),
            caption=np.asarray([
                str(row['caption']) for row in sample_index
            ]),
            perturbed_caption=np.asarray([
                str(row['perturbed_caption']) for row in sample_index
            ]),
            level=np.asarray([
                str(row['level']) for row in sample_index
            ]),
            schema_version=np.asarray(
                'tm2t-perturbed-retrieval-embeddings/v1'
            ),
            seed=np.asarray(
                -1 if retrieval_seed is None else int(retrieval_seed),
                dtype=np.int64,
            ),
            sample_count=np.asarray(nb_sample, dtype=np.int64),
            batch_size=np.asarray(batch_size, dtype=np.int64),
            sample_index_sha256=np.asarray(sample_hash),
            protocol_fingerprint=np.asarray(
                str(retrieval_protocol_fingerprint)
            ),
        )
        metrics['retrieval_embedding_output'] = str(retrieval_path.resolve())
        metrics['retrieval_embedding_sha256'] = hashlib.sha256(
            retrieval_path.read_bytes()
        ).hexdigest()

    if diagnostics_output is not None:
        if len(diagnostic_rows) != nb_sample or not motion_pert_list:
            raise RuntimeError(
                f"SATO++ diagnostics cover {len(diagnostic_rows)}/{nb_sample} samples"
            )
        path = Path(diagnostics_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            gt_embedding=motion_ann_np,
            original_embedding=motion_orig_np,
            perturbed_embedding=motion_pert_np,
            text_embedding=torch.cat(
                diagnostic_text_embeddings, dim=0).numpy(),
            perturbed_text_embedding=torch.cat(
                diagnostic_perturbed_text_embeddings, dim=0).numpy(),
            original_motion=torch.cat(
                diagnostic_original_motions, dim=0).numpy(),
            loader_name=np.asarray([row['loader_name'] for row in diagnostic_rows]),
            motion_id=np.asarray([row['motion_id'] for row in diagnostic_rows]),
            caption=np.asarray([row['caption'] for row in diagnostic_rows]),
            perturbed_caption=np.asarray([
                row['perturbed_caption'] for row in diagnostic_rows
            ]),
            level=np.asarray([str(row['level']) for row in diagnostic_rows]),
            reference_length=np.asarray([
                row['reference_length'] for row in diagnostic_rows
            ], dtype=np.int64),
            original_length=np.asarray([
                row['original_length'] for row in diagnostic_rows
            ], dtype=np.int64),
            perturbed_length=np.asarray([
                row['perturbed_length'] for row in diagnostic_rows
            ], dtype=np.int64),
        )
        metrics['diagnostics_output'] = str(path.resolve())

    if generation_audit_output is not None:
        if generation_audit is None:
            raise RuntimeError("generation audit output requested without audit state")
        for role in ('original', 'perturbed'):
            if len(generation_audit[role]) != nb_sample:
                raise RuntimeError(
                    f"generation audit {role} covers "
                    f"{len(generation_audit[role])}/{nb_sample} samples"
                )
        audit_path = Path(generation_audit_output)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'schema_version': 't2mgpt-satopp-generation-audit/v1',
            'seed': None if generation_audit_seed is None else int(generation_audit_seed),
            'sample_count': int(nb_sample),
            'original': generation_audit['original'],
            'perturbed': generation_audit['perturbed'],
        }
        audit_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        metrics['generation_audit_output'] = str(audit_path.resolve())

    trans_encoder.train()
    return metrics, best_fid_p, best_fid_d, writer, logger
