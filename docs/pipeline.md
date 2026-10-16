# T2M-GPT + COMPASS Pipeline

本文整理当前项目的完整训练 pipeline：先训练 VQ-VAE，再训练正常 T2M-GPT autoregressive backbone，最后在 AR backbone 上做 COMPASS fine-tuning。

## 总览

当前项目可以理解成三阶段：

```text
Stage 1: VQ-VAE training
  continuous motion -> discrete motion tokens

Stage 2: T2M-GPT AR backbone training
  text -> autoregressive motion token prediction

Stage 3: COMPASS fine-tuning
  pretrained AR backbone + preserve/alter pair constraints
```

在本仓库里，Stage 1 和 Stage 2 已经有预训练权重，因此通常直接从 Stage 3 开始跑 COMPASS。

## Stage 1: 训练 VQ-VAE

### 目标

VQ-VAE 负责把连续动作编码成离散 motion tokens，并能把 tokens 解码回连续动作。

```text
continuous motion
 -> VQ-VAE encoder
 -> discrete motion tokens
 -> VQ-VAE decoder
 -> reconstructed motion
```

### 输入数据

主要来自 HumanML3D：

```text
dataset/HumanML3D/new_joint_vecs/*.npy
```

单个 motion feature 常见 shape：

```text
(T, 263)
```

### 训练脚本

```text
train_vq.py
```

原始 README 中的典型命令是：

```bash
python3 train_vq.py \
  --batch-size 256 \
  --lr 2e-4 \
  --total-iter 300000 \
  --lr-scheduler 200000 \
  --nb-code 512 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --out-dir output \
  --dataname t2m \
  --vq-act relu \
  --quantizer ema_reset \
  --loss-vel 0.5 \
  --recons-loss l1_smooth \
  --exp-name VQVAE
```

### 输出

训练完成后得到 VQ-VAE checkpoint，例如：

```text
pretrained/VQVAE/net_best_fid.pth
pretrained/VQVAE/net_last.pth
```

同时可以用 VQ-VAE 把 motion 预先 tokenize，生成：

```text
dataset/HumanML3D/VQVAE/*.npy
```

这些 `.npy` 是后续 T2M-GPT 和 COMPASS 训练实际使用的 motion token 监督。

### 后续阶段是否微调

COMPASS 阶段不微调 VQ-VAE。

VQ-VAE 在后续阶段的作用是：

- 提供离散 motion token 空间。
- 评估/生成时把 motion tokens 解码回连续动作。

## Stage 2: 训练 T2M-GPT AR Backbone

### 目标

T2M-GPT 的 autoregressive backbone 学习：

```text
text -> motion token sequence
```

具体是根据文本条件和已经生成的前缀 motion tokens，预测下一个 motion token：

```text
p_theta(y_t | y_<t, x)
```

### 输入数据

文本：

```text
dataset/HumanML3D/texts/*.txt
```

motion token：

```text
dataset/HumanML3D/VQVAE/*.npy
```

训练划分：

```text
dataset/HumanML3D/train.txt
```

### 训练脚本

```text
train_t2m_trans.py
```

原始 README 中的典型命令是：

```bash
python3 train_t2m_trans.py \
  --exp-name GPT \
  --batch-size 128 \
  --num-layers 9 \
  --embed-dim-gpt 1024 \
  --nb-code 512 \
  --n-head-gpt 16 \
  --block-size 51 \
  --ff-rate 4 \
  --drop-out-rate 0.1 \
  --resume-pth output/VQVAE/net_last.pth \
  --vq-name VQVAE \
  --out-dir output \
  --total-iter 300000 \
  --lr-scheduler 150000 \
  --lr 0.0001 \
  --dataname t2m \
  --down-t 2 \
  --depth 3 \
  --quantizer ema_reset \
  --eval-iter 10000 \
  --pkeep 0.5 \
  --dilation-growth-rate 3 \
  --vq-act relu
```

### 基础 loss

原始 T2M-GPT 主要使用 motion token cross-entropy：

```text
L_gen = CE(predicted motion tokens, target motion tokens)
```

### 输出

得到正常 T2M-GPT Transformer checkpoint，例如：

```text
pretrained/VQTransformer_corruption05/net_best_fid.pth
```

这个 checkpoint 是 COMPASS fine-tuning 的 backbone 初始化。

## Stage 3: COMPASS Fine-tuning

### 目标

COMPASS 在已经训练好的 T2M-GPT AR backbone 上继续微调，加入 preserve/alter 文本 pair 监督，让模型：

- 对语义不变的文本改写保持稳定。
- 对关键语义变化敏感。
- 保持原始 T2M-GPT 的生成能力。

### 输入数据

基础数据仍然来自 HumanML3D：

```text
dataset/HumanML3D/train.txt
dataset/HumanML3D/texts/*.txt
dataset/HumanML3D/VQVAE/*.npy
```

COMPASS 额外使用 pair 数据：

```text
dataset/HumanML3D/preserving_pairs_filtered.jsonl
```

这个文件不是 motion 数据，而是文本 pair 标注：

```text
text_orig
text_pair
relation = preserve / alter
edit_type
motion_id
```

另有 alter-only 补充数据：

```text
dataset/HumanML3D/train_alter_v2.json
```

它默认不直接参与当前主训练，需要转换/合并或改数据加载器。

### 训练脚本

单卡：

```text
train_sato_pp.py
```

多卡 DDP：

```text
train_sato_pp_ddp.py
```

已有实验启动脚本：

```text
launch_sato_v3.sh
```

### 初始化方式

COMPASS 初始化包含几部分：

```text
VQ-VAE
  <- pretrained/VQVAE/net_best_fid.pth
  固定，不微调

AR backbone / Text2Motion Transformer
  <- pretrained/VQTransformer_corruption05/net_best_fid.pth
  继续训练

Teacher Transformer
  <- AR backbone 的冻结副本
  固定，不训练

COMPASS heads
  AttributionPooling
  PredictorHead
  新初始化，参与训练

CLIP
  默认代码支持解冻或冻结
  当前很多实验脚本使用 --freeze-clip
```

### COMPASS 的输入样本

原始 T2M-GPT 样本是：

```text
caption x
motion tokens y
```

COMPASS 样本是：

```text
caption_orig x
caption_pair x+ 或 x-
motion tokens y
relation = preserve / alter
edit_mask m
```

其中：

- `x+` 表示 preserve pair，语义不变。
- `x-` 表示 alter pair，语义改变。
- `m` 表示哪些 token 被编辑。

### COMPASS 的训练目标

总目标：

```text
L = L_gen
  + lambda_teach * L_teach
  + 1[r = preserve] * L_preserve
  + 1[r = alter]    * L_alter
```

其中：

```text
L_gen
```

仍然是原始 T2M-GPT 的 motion token cross-entropy。

```text
L_teach
```

约束当前模型不要偏离 frozen teacher，保护原始生成能力。

Preserve branch：

```text
L_preserve = lambda_align * L_align
           + lambda_var   * L_var
           + lambda_cov   * L_cov
```

作用：

- `L_align`: 让语义不变改写的动作相关表示接近。
- `L_var`: 防止所有表示坍缩成同一个点。
- `L_cov`: 减少表示维度冗余。

Alter branch：

```text
L_alter = lambda_diff * L_diff
        + lambda_loc  * L_loc
```

作用：

- `L_diff`: 被编辑 slot 的 attribution 应该变化。
- `L_loc`: 未编辑 slot 的 attribution 应该保持稳定。

### 单卡训练示例

```bash
cd /mnt/workspace/Wilson/T2M-GPT

CUDA_VISIBLE_DEVICES=0 /mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python train_sato_pp.py \
  --dataname t2m \
  --resume-pth pretrained/VQVAE/net_best_fid.pth \
  --resume-trans pretrained/VQTransformer_corruption05/net_best_fid.pth \
  --out-dir output_GPT_Final \
  --exp-name my_sato_pp_test \
  --vq-name VQVAE \
  --down-t 2 \
  --block-size 51 \
  --embed-dim-gpt 1024 \
  --num-layers 9 \
  --n-head-gpt 16 \
  --batch-size 64 \
  --total-iter 100000 \
  --eval-iter 10000 \
  --save-iter 5000 \
  --milestone-save-iter 20000 \
  --lr 2e-4 \
  --lr-scheduler 30000 \
  --gamma 0.1 \
  --optimizer adamw \
  --freeze-clip \
  --attribution-method grad_input \
  --attribution-K 16 \
  --tau 1.0 \
  --clip-dim-attr 512 \
  --proj-type linear \
  --proj-dim 512 \
  --preserve-ratio 0.5 \
  --pair-data-path dataset/HumanML3D/preserving_pairs_filtered.jsonl \
  --teacher-loss-type kl \
  --lambda-teach 1.0 \
  --lambda-align 0.1 \
  --lambda-var 0.2 \
  --lambda-cov 0.02 \
  --lambda-diff 0.2 \
  --lambda-loc 0.05 \
  --seed 123
```

## 实际使用时的简化流程

因为本仓库已经有预训练权重：

```text
pretrained/VQVAE/net_best_fid.pth
pretrained/VQTransformer_corruption05/net_best_fid.pth
```

所以通常不用重新跑 Stage 1 和 Stage 2。

实际常用流程是：

```text
1. 确认 HumanML3D 数据、VQVAE tokens、pair 数据存在。
2. 直接加载 pretrained VQ-VAE 和 pretrained T2M-GPT Transformer。
3. 运行 train_sato_pp.py 或 train_sato_pp_ddp.py 做 COMPASS fine-tuning。
4. 用 eval_standard_and_perturbed.py 评估标准指标和扰动指标。
```

## 评估阶段

标准 T2M 指标：

```text
FID
Diversity
R-Precision
Matching Score
```

COMPASS 扰动指标：

```text
FID_P
FID_D
PerturbedMatched
```

单个 checkpoint 评估示例：

```bash
cd /mnt/workspace/Wilson/T2M-GPT

CUDA_VISIBLE_DEVICES=0 /mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python \
  eval_standard_and_perturbed.py \
  --checkpoint output_GPT_Final/sato_v3c_l1_teach_low_sato_2gpu/sato_pp_best_fid.pth \
  --gpu 0 \
  --pair-data-path dataset/HumanML3D/test_perb_deterministic.json \
  --out-json output_GPT_Final/sato_v3c_l1_teach_low_sato_2gpu/eval_standard_perturbed.json
```

## 最核心的理解

完整 pipeline 是：

```text
VQ-VAE learns motion token space
  -> T2M-GPT AR backbone learns text-to-token generation
  -> COMPASS fine-tunes AR backbone with attribution-aware preserve/alter constraints
```

COMPASS 阶段不重新学习 motion token 空间，而是在已有 T2M-GPT 的基础上，增强它对文本语义改写和语义编辑的稳定性与敏感性。

