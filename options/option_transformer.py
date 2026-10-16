import argparse

from utils.plateau_control import add_plateau_args

def _build_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amas',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader

    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=64, help='training motion length')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='micro-batches per optimizer step')

    ## optimization
    parser.add_argument('--total-iter', default=100000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[60000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')

    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=3, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')

    ## gpt arch
    parser.add_argument("--block-size", type=int, default=25, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=512, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=2, help="nb of transformer layers")
    parser.add_argument("--n-head-gpt", type=int, default=8, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")

    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume vq pth')
    parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')


    ## output directory
    parser.add_argument('--out-dir', type=str, default='output_GPT_Final/', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--vq-name', type=str, default='exp_debug', help='name of the generated dataset .npy, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=5000, type=int, help='evaluation frequency')
    parser.add_argument('--eval-start-iter', default=0, type=int,
                        help='first optimization step eligible for periodic evaluation')
    parser.add_argument('--save-iter', default=10000, type=int)
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=1.0, help='keep rate for gpt training')
    parser.add_argument('--aug-prob', type=float, default=0.5, help='probability of synonym augmentation per sample')
    parser.add_argument('--pair-data-path', type=str, default=None, help='path to preserving pairs JSONL for text augmentation')
    parser.add_argument('--skip-initial-eval', action='store_true',
                        help='skip the step-0 evaluation when recovering a run that already completed it')
    parser.add_argument('--guard-fid-selection', action='store_true')
    parser.add_argument('--fid-top1-tolerance', type=float, default=0.01)
    parser.add_argument('--fid-matching-tolerance', type=float, default=0.02)

    ## experiment tracking
    parser.add_argument('--wandb-project', type=str, default='SATO++-KIT-Ablation')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--wandb-run-id', type=str, default=None,
                        help='Stable W&B run id used to resume one continuous run')
    parser.add_argument('--wandb-group', type=str, default=None)
    parser.add_argument('--wandb-tags', type=str, default='')
    parser.add_argument('--wandb-mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb-watch-log', type=str, default='gradients',
                        choices=['gradients', 'parameters', 'all', 'none'])
    parser.add_argument('--wandb-watch-freq', type=int, default=200)
    parser.add_argument('--disable-wandb', action='store_true')

    add_plateau_args(
        parser, min_index_name='min-train-iter', min_index_default=0,
        stop_patience=6, min_lr=5e-6,
    )
    return parser


def get_args_parser():
    """Original API: returns parsed args (backward compatible)."""
    return _build_parser().parse_args()


def get_parser():
    """SATO++ API: returns ArgumentParser for further extension."""
    return _build_parser()
