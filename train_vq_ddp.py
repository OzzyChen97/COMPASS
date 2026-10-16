#!/usr/bin/env python
"""Four-GPU torchrun VQ-VAE training with synchronized EMA codebooks."""

import json
import os
from contextlib import nullcontext

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import wandb

from dataset import dataset_TM_eval, dataset_VQ
from dataset.config import (
    configure_evaluator_source,
    dataset_max_motion_length,
    evaluator_opt_path,
    get_dataset_spec,
)
from dataset.variable_motion import lengths_to_mask
import models.vqvae as vqvae
from models.evaluator_wrapper import EvaluatorModelWrapper
import options.option_vq as option_vq
from options.get_eval_option import get_opt
import utils.eval_trans as eval_trans
import utils.losses as losses
from utils.distributed_training import (
    NoOpLogger, barrier, cleanup, reduce_sums, setup_distributed,
)
import utils.utils_model as utils_model
from utils.plateau_control import PlateauController
from utils.word_vectorizer import WordVectorizer


def warmup_lr(optimizer, step, warmup_steps, lr):
    current = lr * (step + 1) / (warmup_steps + 1)
    for group in optimizer.param_groups:
        group["lr"] = current
    return current


def prepare_motion_batch(batch, device, unit_length):
    if isinstance(batch, dict):
        motion = batch["motion"].to(device, non_blocking=True).float()
        frame_mask = batch["frame_mask"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        latent_lengths = torch.div(
            lengths + unit_length - 1, unit_length, rounding_mode="floor")
        latent_mask = lengths_to_mask(
            latent_lengths, motion.shape[1] // unit_length, device=device)
        return motion, frame_mask, latent_mask
    motion = batch.to(device, non_blocking=True).float()
    return motion, None, None


def batch_valid_counts(batch, unit_length):
    if isinstance(batch, dict):
        lengths = torch.as_tensor(batch["lengths"], dtype=torch.long)
        frame_count = lengths.sum()
        velocity_count = (lengths - 1).clamp_min(0).sum()
        latent_count = torch.div(
            lengths + unit_length - 1,
            unit_length,
            rounding_mode="floor",
        ).sum()
    else:
        batch_size, frame_length = batch.shape[:2]
        frame_count = torch.tensor(batch_size * frame_length)
        velocity_count = torch.tensor(batch_size * max(frame_length - 1, 0))
        latent_count = torch.tensor(
            batch_size * max(frame_length // unit_length, 1))
    return torch.stack((frame_count, velocity_count, latent_count)).double()


def global_loss_scales(batches, unit_length, device, distributed):
    local_counts = torch.stack([
        batch_valid_counts(batch, unit_length)
        for batch in batches
    ]).to(device=device)
    global_counts = local_counts.sum(dim=0)
    if distributed:
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size() if distributed else 1
    return local_counts * world_size / global_counts.clamp_min(1.0)


@torch.no_grad()
def initialize_codebook_from_batches(module, batches, device, unit_length):
    quantizer = module.vqvae.quantizer
    if not hasattr(quantizer, "init_codebook") or quantizer.init:
        return
    candidates = []
    candidate_count = 0
    for batch in batches:
        motion, _, latent_mask = prepare_motion_batch(
            batch, device, unit_length)
        encoded = module.vqvae.encoder(module.vqvae.preprocess(motion))
        encoded = quantizer.preprocess(encoded)
        if latent_mask is not None:
            encoded = encoded[latent_mask.reshape(-1)]
        take = min(quantizer.nb_code - candidate_count, encoded.shape[0])
        if take > 0:
            candidates.append(encoded[:take])
            candidate_count += take
        if candidate_count >= quantizer.nb_code:
            break
    if not candidates:
        raise RuntimeError("Could not initialize the VQ codebook from the global batch")
    quantizer.init_codebook(torch.cat(candidates, dim=0))


def train_micro_batches(model, module, iterator, optimizer, reconstruction,
                        args, device, distributed):
    optimizer.zero_grad(set_to_none=True)
    totals = [0.0, 0.0, 0.0]
    unit_length = 2 ** args.down_t
    batches = [next(iterator) for _ in range(args.grad_accum_steps)]
    scales = global_loss_scales(
        batches, unit_length, device, distributed)
    initialize_codebook_from_batches(module, batches, device, unit_length)
    quantizer = module.vqvae.quantizer
    accumulate_codebook = hasattr(quantizer, "begin_codebook_accumulation")
    if accumulate_codebook:
        quantizer.begin_codebook_accumulation()
    for micro_step, batch in enumerate(batches):
        motion, frame_mask, latent_mask = prepare_motion_batch(
            batch, device, unit_length)
        sync_context = (
            model.no_sync()
            if distributed and micro_step + 1 < args.grad_accum_steps
            else nullcontext()
        )
        with sync_context:
            prediction, commitment, perplexity = model(
                motion, latent_mask=latent_mask)
            rec_loss = reconstruction(prediction, motion, frame_mask)
            velocity_loss = reconstruction.forward_vel(
                prediction, motion, frame_mask)
            loss = (
                scales[micro_step, 0] * rec_loss
                + args.commit * scales[micro_step, 2] * commitment
                + args.loss_vel * scales[micro_step, 1] * velocity_loss
            )
            loss.backward()
        totals[0] += rec_loss.item() * scales[micro_step, 0].item()
        totals[1] += commitment.item() * scales[micro_step, 2].item()
        if not accumulate_codebook:
            totals[2] += perplexity.item() * scales[micro_step, 2].item()
    if accumulate_codebook:
        totals[2] = quantizer.end_codebook_accumulation().item()
    optimizer.step()
    return totals


def main():
    args = option_vq.get_args_parser()
    distributed, rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.world_size = world_size
    spec = get_dataset_spec(args.dataname)
    if spec.variable_length:
        args.sequence_mode = "full_variable"
        args.window_size = None
        args.max_motion_length = dataset_max_motion_length(args.dataname)
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be positive")
    args.global_batch_size = args.batch_size * world_size * args.grad_accum_steps
    args.out_dir = os.path.join(args.out_dir, args.exp_name)
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    barrier(distributed)

    if is_main:
        logger = utils_model.get_logger(args.out_dir)
        writer = SummaryWriter(args.out_dir)
        logger.info(json.dumps(vars(args), indent=2, sort_keys=True))
        wandb_run = None
        if not args.disable_wandb and args.wandb_mode != "disabled":
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name or args.exp_name,
                group=args.wandb_group,
                job_type="vqvae",
                tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()],
                config=vars(args),
                dir=args.out_dir,
                mode=args.wandb_mode,
                resume="allow",
            )
            wandb_run.define_metric("optimization_step")
            wandb_run.define_metric("train/*", step_metric="optimization_step")
            wandb_run.define_metric("eval/*", step_metric="optimization_step")
            with open(os.path.join(args.out_dir, "wandb_run_url.txt"), "w") as handle:
                handle.write((wandb_run.url or "") + "\n")
        vectorizer = WordVectorizer("./glove", "our_vab")
        val_loader = dataset_TM_eval.DATALoader(
            args.dataname, False, 32, vectorizer, unit_length=2 ** args.down_t
        )
        wrapper_opt = get_opt(evaluator_opt_path(args.dataname), device)
        wrapper_opt = configure_evaluator_source(wrapper_opt, args.dataname)
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    else:
        logger, writer, wandb_run, val_loader, eval_wrapper = NoOpLogger(), None, None, None, None

    args.nb_joints = spec.joints
    train_loader = dataset_VQ.DATALoader(
        args.dataname,
        args.batch_size,
        num_workers=args.num_workers,
        window_size=args.window_size,
        unit_length=2 ** args.down_t,
        distributed=distributed,
        seed=args.seed,
    )
    iterator = dataset_VQ.cycle(train_loader)
    module = vqvae.HumanVQVAE(
        args, args.nb_code, args.code_dim, args.output_emb_width, args.down_t,
        args.stride_t, args.width, args.depth, args.dilation_growth_rate,
        args.vq_act, args.vq_norm,
    ).to(device)
    if args.resume_pth:
        checkpoint = torch.load(args.resume_pth, map_location="cpu")
        module.load_state_dict(checkpoint["net"], strict=True)
    model = DistributedDataParallel(
        module, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=True,
    ) if distributed else module
    if is_main and wandb_run is not None and args.wandb_watch_log != "none":
        wandb_run.watch(module, log=args.wandb_watch_log, log_freq=args.wandb_watch_freq)

    optimizer = optim.AdamW(
        module.parameters(), lr=args.lr, betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    scheduler = None
    plateau = None
    if args.lr_policy == "fixed":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_scheduler, gamma=args.gamma
        )
    else:
        plateau = PlateauController(
            mode="min", factor=args.plateau_factor,
            lr_patience=args.plateau_patience,
            stop_patience=args.early_stop_patience,
            min_index=args.min_train_iter,
            min_lr=args.plateau_min_lr,
        )
    reconstruction = losses.ReConsLoss(
        args.recons_loss, args.nb_joints, dataset_name=args.dataname)

    running = [0.0, 0.0, 0.0]
    latest_perplexity = 0.0
    peak_perplexity = 0.0
    for step in range(1, args.warm_up_iter):
        current_lr = warmup_lr(optimizer, step, args.warm_up_iter, args.lr)
        values = train_micro_batches(
            model, module, iterator, optimizer, reconstruction,
            args, device, distributed)
        for index, value in enumerate(values):
            running[index] += value
        if step % args.print_iter == 0:
            totals = reduce_sums(running, device, distributed)
            values = [value / (args.print_iter * world_size) for value in totals]
            if is_main:
                logger.info(
                    "Warmup step %d lr %.6g reconstruction %.6f commitment %.6f perplexity %.3f",
                    step, current_lr, values[0], values[1], values[2],
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "train/phase": 0,
                        "train/reconstruction_loss": values[0],
                        "train/commitment_loss": values[1],
                        "train/codebook_perplexity": values[2],
                        "optimizer/learning_rate": current_lr,
                        "optimization_step": step,
                    })
            running = [0.0, 0.0, 0.0]

    best = [1000, 0, 100, 0, 0, 0, 100]
    barrier(distributed)
    initial_fid = float("nan")
    if is_main:
        (*best, writer, logger, initial_metrics) = eval_trans.evaluation_vqvae(
            args.out_dir, val_loader, module, logger, writer, 0,
            best_fid=best[0], best_iter=best[1], best_div=best[2],
            best_top1=best[3], best_top2=best[4], best_top3=best[5],
            best_matching=best[6], eval_wrapper=eval_wrapper,
            wandb_run=wandb_run, wandb_step=args.warm_up_iter,
            return_metrics=True,
        )
        initial_fid = initial_metrics["fid"]
    metric_tensor = torch.tensor(initial_fid, dtype=torch.float64, device=device)
    if distributed:
        dist.broadcast(metric_tensor, src=0)
    if plateau is not None:
        plateau.step(float(metric_tensor.item()), 0, [optimizer])
    barrier(distributed)

    running = [0.0, 0.0, 0.0]
    final_step = 0
    for step in range(1, args.total_iter + 1):
        final_step = step
        module.train()
        values = train_micro_batches(
            model, module, iterator, optimizer, reconstruction,
            args, device, distributed)
        if scheduler is not None:
            scheduler.step()
        for index, value in enumerate(values):
            running[index] += value

        if step % args.print_iter == 0:
            totals = reduce_sums(running, device, distributed)
            values = [value / (args.print_iter * world_size) for value in totals]
            latest_perplexity = float(values[2])
            peak_perplexity = max(peak_perplexity, latest_perplexity)
            if is_main:
                writer.add_scalar("./Train/L1", values[0], step)
                writer.add_scalar("./Train/Commit", values[1], step)
                writer.add_scalar("./Train/PPL", values[2], step)
                logger.info(
                    "Train step %d reconstruction %.6f commitment %.6f perplexity %.3f",
                    step, values[0], values[1], values[2],
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "train/phase": 1,
                        "train/reconstruction_loss": values[0],
                        "train/commitment_loss": values[1],
                        "train/codebook_perplexity": values[2],
                        "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
                        "optimization_step": args.warm_up_iter + step,
                    })
            running = [0.0, 0.0, 0.0]

        if step % args.save_iter == 0 and is_main:
            torch.save({
                "net": module.state_dict(), "iter": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "plateau": plateau.state_dict() if plateau else None,
                "latest_perplexity": latest_perplexity,
                "peak_perplexity": peak_perplexity,
            },
                       os.path.join(args.out_dir, f"net_iter_{step}.pth"))
        if step % args.eval_iter == 0:
            barrier(distributed)
            current_fid = float("nan")
            if is_main:
                perplexity_eligible = (
                    args.min_perplexity_ratio <= 0
                    or latest_perplexity >= args.min_perplexity_ratio * peak_perplexity
                )
                (*best, writer, logger, current_metrics) = eval_trans.evaluation_vqvae(
                    args.out_dir, val_loader, module, logger, writer, step,
                    best_fid=best[0], best_iter=best[1], best_div=best[2],
                    best_top1=best[3], best_top2=best[4], best_top3=best[5],
                    best_matching=best[6], eval_wrapper=eval_wrapper,
                    wandb_run=wandb_run, wandb_step=args.warm_up_iter + step,
                    return_metrics=True,
                    checkpoint_eligible=perplexity_eligible,
                )
                current_fid = current_metrics["fid"]
            metric_tensor = torch.tensor(current_fid, dtype=torch.float64, device=device)
            if distributed:
                dist.broadcast(metric_tensor, src=0)
            stop_training = False
            if plateau is not None:
                decision = plateau.step(float(metric_tensor.item()), step, [optimizer])
                stop_training = decision.stop
                if is_main:
                    logger.info(
                        "Plateau control step %d fid %.6f best %.6f bad %d lr %.3g reduce=%s stop=%s",
                        step, float(metric_tensor.item()), decision.best_metric,
                        decision.bad_evals, decision.current_lr,
                        decision.reduce_lr, decision.stop,
                    )
                    if wandb_run is not None:
                        wandb_run.log({
                            "control/current_fid": float(metric_tensor.item()),
                            "control/best_fid": decision.best_metric,
                            "control/bad_evals": decision.bad_evals,
                            "control/lr_reduced": int(decision.reduce_lr),
                            "control/early_stop": int(decision.stop),
                            "optimization_step": args.warm_up_iter + step,
                        })
            barrier(distributed)
            if stop_training:
                break

    if is_main:
        torch.save({
            "net": module.state_dict(), "iter": final_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "plateau": plateau.state_dict() if plateau else None,
            "latest_perplexity": latest_perplexity,
            "peak_perplexity": peak_perplexity,
        },
                   os.path.join(args.out_dir, "net_final.pth"))
        writer.close()
        if wandb_run is not None:
            wandb_run.finish()
    cleanup(distributed)


if __name__ == "__main__":
    main()
