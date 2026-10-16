#!/usr/bin/env python
"""Four-GPU torchrun training for the T2M-GPT autoregressive baseline."""

import json
import os
from contextlib import nullcontext

import clip
import numpy as np
import torch
import torch.distributed as dist
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import wandb

from dataset import dataset_TM_eval, dataset_TM_train
from dataset.config import (
    configure_evaluator_source,
    dataset_max_motion_length,
    evaluator_opt_path,
    get_dataset_spec,
)
import models.t2m_trans as trans
import models.vqvae as vqvae
from models.evaluator_wrapper import EvaluatorModelWrapper
import options.option_transformer as option_trans
from options.get_eval_option import get_opt
import utils.eval_trans as eval_trans
from utils.distributed_training import (
    NoOpLogger, barrier, cleanup, reduce_sums, setup_distributed,
)
import utils.utils_model as utils_model
from utils.plateau_control import PlateauController
from utils.word_vectorizer import WordVectorizer


def compute_training_batch(batch, model, clip_model, criterion, args, device):
    captions, motion_tokens, token_lengths = batch
    motion_tokens = motion_tokens.to(device)
    token_lengths = token_lengths.to(device)
    text_tokens = clip.tokenize(captions, truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()

    input_indices = motion_tokens[:, :-1]
    if args.pkeep == -1:
        probability = float(np.random.rand())
        mask = torch.bernoulli(
            probability * torch.ones_like(input_indices, dtype=torch.float32))
    else:
        mask = torch.bernoulli(
            args.pkeep * torch.ones_like(input_indices, dtype=torch.float32))
    mask = mask.round().long()
    random_indices = torch.randint_like(input_indices, args.nb_code)
    corrupted = mask * input_indices + (1 - mask) * random_indices
    logits = model(corrupted, text_features).contiguous()

    batch_size = motion_tokens.shape[0]
    loss = torch.zeros((), device=device)
    correct = 0
    token_count = 0
    for index in range(batch_size):
        length = int(token_lengths[index].item()) + 1
        current_logits = logits[index, :length]
        current_target = motion_tokens[index, :length]
        loss = loss + criterion(current_logits, current_target) / batch_size
        probabilities = torch.softmax(current_logits, dim=-1)
        if args.if_maxtest:
            predicted = probabilities.argmax(dim=-1)
        else:
            predicted = Categorical(probabilities).sample()
        correct += int((predicted == current_target).sum().item())
        token_count += length
    return loss, correct, token_count


def should_run_periodic_eval(step, eval_start_iter, eval_iter):
    return step >= eval_start_iter and step % eval_iter == 0


def main():
    args = option_trans.get_args_parser()
    distributed, rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.world_size = world_size
    spec = get_dataset_spec(args.dataname)
    if spec.variable_length:
        args.sequence_mode = "full_variable"
        args.seq_len = None
        args.max_motion_length = dataset_max_motion_length(args.dataname)
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be positive")
    if args.eval_iter < 1 or args.eval_start_iter < 0:
        raise ValueError("Evaluation interval must be positive and start non-negative")
    args.global_batch_size = args.batch_size * world_size * args.grad_accum_steps
    args.out_dir = os.path.join(args.out_dir, args.exp_name)
    args.vq_dir = str(get_dataset_spec(args.dataname).root / args.vq_name)
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
                job_type="t2m-gpt",
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
        val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, vectorizer)
        wrapper_opt = get_opt(evaluator_opt_path(args.dataname), device)
        wrapper_opt = configure_evaluator_source(wrapper_opt, args.dataname)
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    else:
        logger, writer, wandb_run, val_loader, eval_wrapper = NoOpLogger(), None, None, None, None

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    vq_model = vqvae.HumanVQVAE(
        args, args.nb_code, args.code_dim, args.output_emb_width, args.down_t,
        args.stride_t, args.width, args.depth, args.dilation_growth_rate,
    ).to(device)
    vq_checkpoint = torch.load(args.resume_pth, map_location="cpu")
    vq_model.load_state_dict(vq_checkpoint["net"], strict=True)
    vq_model.eval()

    module = trans.Text2Motion_Transformer(
        num_vq=args.nb_code,
        embed_dim=args.embed_dim_gpt,
        clip_dim=args.clip_dim,
        block_size=args.block_size,
        num_layers=args.num_layers,
        n_head=args.n_head_gpt,
        drop_out_rate=args.drop_out_rate,
        fc_rate=args.ff_rate,
    ).to(device)
    # The legacy table is retained for checkpoint compatibility; forward uses pos_embed.
    module.trans_base.pos_embedding.requires_grad_(False)
    if args.resume_trans:
        checkpoint = torch.load(args.resume_trans, map_location="cpu")
        module.load_state_dict(checkpoint["trans"], strict=True)
    model = DistributedDataParallel(
        module, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False,
    ) if distributed else module
    if is_main and wandb_run is not None and args.wandb_watch_log != "none":
        wandb_run.watch(module, log=args.wandb_watch_log, log_freq=args.wandb_watch_freq)

    optimizer = utils_model.initial_optim(
        args.decay_option, args.lr, args.weight_decay, module, args.optimizer
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
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = dataset_TM_train.DATALoader(
        args.dataname, args.batch_size, args.nb_code, args.vq_name,
        unit_length=2 ** args.down_t, num_workers=args.num_workers,
        distributed=distributed, seed=args.seed,
    )
    iterator = dataset_TM_train.cycle(train_loader)

    best = [1000, 0, 100, 0, 0, 0, 100]
    barrier(distributed)
    initial_fid = float("nan")
    if is_main and not args.skip_initial_eval:
        (*best, writer, logger, initial_metrics) = eval_trans.evaluation_transformer(
            args.out_dir, val_loader, vq_model, module, logger, writer, 0,
            best_fid=best[0], best_iter=best[1], best_div=best[2],
            best_top1=best[3], best_top2=best[4], best_top3=best[5],
            best_matching=best[6], clip_model=clip_model,
            eval_wrapper=eval_wrapper, wandb_run=wandb_run,
            return_metrics=True,
            guard_fid=args.guard_fid_selection,
            fid_top1_tolerance=args.fid_top1_tolerance,
            fid_matching_tolerance=args.fid_matching_tolerance,
        )
        initial_fid = initial_metrics["fid"]
    barrier(distributed)
    if is_main and args.skip_initial_eval:
        logger.info("Skipping step-0 evaluation because it was completed before recovery")
    if plateau is not None and not args.skip_initial_eval:
        metric_tensor = torch.tensor(initial_fid, dtype=torch.float64, device=device)
        if distributed:
            dist.broadcast(metric_tensor, src=0)
        plateau.step(float(metric_tensor.item()), 0, [optimizer])

    running_loss = 0.0
    running_correct = 0.0
    running_tokens = 0.0
    final_step = 0
    for step in range(1, args.total_iter + 1):
        final_step = step
        module.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        correct = 0
        token_count = 0
        for micro_step in range(args.grad_accum_steps):
            sync_context = (
                model.no_sync()
                if distributed and micro_step + 1 < args.grad_accum_steps
                else nullcontext()
            )
            with sync_context:
                loss, micro_correct, micro_tokens = compute_training_batch(
                    next(iterator), model, clip_model, criterion, args, device)
                (loss / args.grad_accum_steps).backward()
            loss_value += loss.item() / args.grad_accum_steps
            correct += micro_correct
            token_count += micro_tokens
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss_value
        running_correct += correct
        running_tokens += token_count

        if step % args.print_iter == 0:
            totals = reduce_sums(
                [running_loss, running_correct, running_tokens], device, distributed
            )
            mean_loss = totals[0] / (args.print_iter * world_size)
            accuracy = 100.0 * totals[1] / max(totals[2], 1.0)
            if is_main:
                writer.add_scalar("./Loss/train", mean_loss, step)
                writer.add_scalar("./ACC/train", accuracy, step)
                logger.info("Train step %d loss %.6f accuracy %.4f", step, mean_loss, accuracy)
                if wandb_run is not None:
                    wandb_run.log({
                        "train/cross_entropy": mean_loss,
                        "train/token_accuracy_percent": accuracy,
                        "train/tokens": int(totals[2]),
                        "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
                        "optimization_step": step,
                    })
            running_loss = running_correct = running_tokens = 0.0

        if step % args.save_iter == 0 and is_main:
            torch.save({
                "trans": module.state_dict(), "iter": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "plateau": plateau.state_dict() if plateau else None,
            },
                       os.path.join(args.out_dir, f"net_iter_{step}.pth"))
        if should_run_periodic_eval(
                step, args.eval_start_iter, args.eval_iter):
            barrier(distributed)
            current_fid = float("nan")
            if is_main:
                (*best, writer, logger, current_metrics) = eval_trans.evaluation_transformer(
                    args.out_dir, val_loader, vq_model, module, logger, writer, step,
                    best_fid=best[0], best_iter=best[1], best_div=best[2],
                    best_top1=best[3], best_top2=best[4], best_top3=best[5],
                    best_matching=best[6], clip_model=clip_model,
                    eval_wrapper=eval_wrapper, wandb_run=wandb_run,
                    return_metrics=True,
                    guard_fid=args.guard_fid_selection,
                    fid_top1_tolerance=args.fid_top1_tolerance,
                    fid_matching_tolerance=args.fid_matching_tolerance,
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
                            "optimization_step": step,
                        })
            barrier(distributed)
            if stop_training:
                break

    if is_main:
        torch.save({
            "trans": module.state_dict(), "iter": final_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "plateau": plateau.state_dict() if plateau else None,
        },
                   os.path.join(args.out_dir, "net_final.pth"))
        writer.close()
        if wandb_run is not None:
            wandb_run.finish()
    cleanup(distributed)


if __name__ == "__main__":
    main()
