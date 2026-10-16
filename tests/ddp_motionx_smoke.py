"""Minimal torchrun smoke test for synchronized VQ and GPT training."""

from argparse import Namespace
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models.t2m_trans as t2m_trans
import models.vqvae as vqvae


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(123)

    args = Namespace(
        dataname="motionx",
        quantizer="ema_reset",
        mu=0.99,
    )
    vq_module = vqvae.HumanVQVAE(
        args, nb_code=16, code_dim=16, output_emb_width=16,
        down_t=1, stride_t=2, width=32, depth=1,
        dilation_growth_rate=2,
    ).to(device)
    vq_model = DistributedDataParallel(vq_module, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(vq_module.parameters(), lr=1e-4)
    motion = torch.randn(2, 32, 263, device=device) + local_rank * 0.01
    output, commitment, _ = vq_model(motion)
    loss = (output - motion).abs().mean() + commitment
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    codebook = vq_module.vqvae.quantizer.codebook
    reference = codebook.clone()
    dist.broadcast(reference, src=0)
    if not torch.allclose(codebook, reference):
        raise RuntimeError("EMA codebooks diverged across ranks")

    gpt_module = t2m_trans.Text2Motion_Transformer(
        num_vq=16, embed_dim=32, clip_dim=16, block_size=9,
        num_layers=2, n_head=4, drop_out_rate=0.1, fc_rate=2,
    ).to(device)
    gpt_model = DistributedDataParallel(gpt_module, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(gpt_module.parameters(), lr=1e-4)
    indices = torch.randint(0, 16, (2, 8), device=device)
    text = torch.randn(2, 16, device=device)
    logits = gpt_model(indices, text)
    target = torch.randint(0, 17, (2, logits.shape[1]), device=device)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 17), target.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite GPT loss")

    if dist.get_rank() == 0:
        print("motionx ddp smoke passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
