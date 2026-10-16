"""
SATO++ Option Parser

在 T2M-GPT option_transformer.py 基础上新增 SATO++ 专用超参数。
"""

import argparse
from options.option_transformer import get_parser as get_base_parser


def get_sato_pp_args():
    parser = get_base_parser()

    # ======== SATO++ Preserving Losses ========
    parser.add_argument('--lambda-align', type=float, default=0.2,
                        help='Preserving align loss weight (SimSiam)')
    parser.add_argument('--lambda-var', type=float, default=0.5,
                        help='VICReg variance loss weight')
    parser.add_argument('--lambda-cov', type=float, default=0.04,
                        help='VICReg covariance loss weight')
    parser.add_argument('--var-threshold', type=float, default=1.0,
                        help='VICReg variance threshold')

    # ======== SATO++ Altering Losses ========
    parser.add_argument('--lambda-diff', type=float, default=0.5,
                        help='Altering attribution diff loss weight')
    parser.add_argument('--lambda-loc', type=float, default=0.1,
                        help='Altering locality loss weight')

    # ======== Teacher ========
    parser.add_argument('--lambda-teach', type=float, default=0.05,
                        help='Teacher KL loss weight')
    parser.add_argument('--teacher-loss-type', type=str, default='kl',
                        choices=['kl', 'l1'],
                        help='Teacher 约束类型：kl 为分布蒸馏，l1 为 logits L1 约束')

    # ======== Attribution ========
    parser.add_argument('--attribution-K', type=int, default=16,
                        help='AR score: 前 K 步计算')
    parser.add_argument('--attribution-method', type=str, default='grad_input',
                        choices=['grad_input', 'attention_raw'],
                        help='归因计算方法')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Attribution softmax temperature')
    parser.add_argument(
        '--attribution-create-graph',
        action='store_true',
        help=(
            'Retain the Grad x Input graph so attribution losses backpropagate '
            'through the text-to-motion backbone. This uses second-order gradients.'
        ),
    )

    # ======== Representation ========
    # clip-dim 在 base parser 里默认 512（用于 Transformer embed_dim），
    # SATO++ 用 CLIP ViT-B/32 的 transformer width=768 做 AttributionPooling。
    # 用 --clip-dim-attr 区分两个用途。
    parser.add_argument('--clip-dim-attr', type=int, default=512,
                        help='CLIP text transformer width for AttributionPooling (ViT-B/32=512)')
    parser.add_argument('--proj-type', type=str, default='linear',
                        choices=['linear', 'mlp'],
                        help='φ projector 类型')
    parser.add_argument('--proj-dim', type=int, default=512,
                        help='u(x) 输出维度')

    # ======== Data ========
    # --pair-data-path 已在 base parser (option_transformer.py) 中定义
    parser.add_argument('--preserve-ratio', type=float, default=0.5,
                        help='Batch 中 preserving 样本比例')
    parser.add_argument(
        '--dataset-limit',
        type=int,
        default=None,
        help='Optional number of training motion ids to load for smoke tests.',
    )

    # ======== Checkpointing ========
    if '--save-iter' not in parser._option_string_actions:
        parser.add_argument('--save-iter', type=int, default=1000,
                            help='完整 SATO++ checkpoint 保存间隔')
    else:
        parser.set_defaults(save_iter=1000)
    parser.add_argument('--milestone-save-iter', type=int, default=5000,
                        help='额外保留带 iter 编号 checkpoint 的间隔；<=0 表示关闭')
    parser.add_argument('--resume-sato-pp', type=str, default=None,
                        help='从完整 SATO++ checkpoint 恢复训练')

    # ======== CLIP Training ========
    parser.add_argument('--clip-lr', type=float, default=1e-5,
                        help='CLIP 学习率（解冻后用小 lr）')
    parser.add_argument('--clip-min-lr', type=float, default=1e-7)
    parser.add_argument('--unfreeze-clip', action='store_true',
                        dest='unfreeze_clip',
                        help='解冻 CLIP 进行微调')
    parser.add_argument('--freeze-clip', action='store_false',
                        dest='unfreeze_clip',
                        help='冻结 CLIP，只训练 transformer 和 SATO++ heads')
    parser.set_defaults(unfreeze_clip=True)

    # ======== Evaluation ========
    parser.add_argument('--eval-perturbed', action='store_true',
                        help='评测时同时跑 perturbed text (FID_P, FID_D)')
    parser.add_argument('--robust-eval-iter', type=int, default=5000)
    parser.add_argument('--robust-eval-repeats', type=int, default=3)
    parser.add_argument('--robust-val-pair-paths', nargs='*', default=[])
    parser.add_argument('--robust-fid-tolerance', type=float, default=0.05)
    parser.add_argument('--cfg-scale', type=float, default=1.3)
    parser.add_argument('--eval-draw', action='store_true',
                        help='训练评估时写入 3D motion 可视化视频；默认关闭以减少评估开销')

    return parser.parse_args()
