# COMPASS

COMPASS is a research framework for relation-conditioned prompt robustness in text-to-motion generation, built on [T2M-GPT](https://github.com/Mael-zys/T2M-GPT). It combines motion-conditioned attribution, preserving/altering text pairs, and frozen-teacher consistency. It supports HumanML3D, KIT-ML, and native 322D Motion-X workflows, with MotionGPT integration scripts.

本项目原名 SATO++，现更名为 **COMPASS**。为兼容已有 checkpoint、命令行参数和实验脚本，代码中的 `sato` / `sato_pp` / `satopp` 标识符保持不变；既有图像中也可能保留旧名称。

## Installation

```bash
conda env create -f environment.yml
conda activate T2M-GPT
pip install wandb openai
bash dataset/prepare/download_glove.sh
bash dataset/prepare/download_extractor.sh
```

`environment.yml` is the inherited T2M-GPT environment, not a newly validated lockfile. Full-gradient attribution and recent backbone integrations may require a newer compatible PyTorch/CUDA environment. This release has been checked for Python/shell syntax, credential patterns, and successful manuscript compilation; training and numerical reproduction have not been rerun.

## Data and pretrained models

Download [HumanML3D](https://github.com/EricGuo5513/HumanML3D), KIT-ML, or [Motion-X](https://github.com/IDEA-Research/Motion-X) separately under `dataset/`. Obtain pretrained checkpoints according to the upstream project instructions and their licenses. No datasets, generated text pairs, model weights, or external backbone repositories are distributed here. Rendering additionally requires the separately licensed SMPL assets and fitting priors expected under `visualize/joints2smpl/smpl_models/`.

For each dataset, prepare motion features, captions, normalization statistics, and the official train/validation/test splits. See [the pipeline guide](docs/pipeline.md) and [method notes](docs/sato_pp_understanding.md). Local paths and GPU assignments in historical experiment launchers must be adapted to your environment.

Text-pair generation reads credentials from the environment:

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export SATO_PAIR_MODEL="<available-model>"
python tools/build_sato_pairs.py --help
```

Some dataset-specific legacy generators select their model in the script. Never commit credentials.

## Training and evaluation

```bash
python train_vq.py --help
python train_t2m_trans.py --help
python train_sato_pp.py --help
python train_sato_pp_ddp.py --help
python eval_standard_and_perturbed.py --help
```

The training sequence is VQ-VAE, autoregressive baseline, then COMPASS fine-tuning. `models/clip_attribution.py`, `models/sato_pp_losses.py`, and `dataset/dataset_sato_pp.py` implement the main attribution, losses, and paired-data components. Reusable launchers live in `ablation/`; evaluation and analysis utilities live in `tools/`. `MotionGPT/` contains integration scripts only and requires the upstream MotionGPT repository and its dependencies.

## Paper and results

The manuscript source is [paper_tip/main.tex](paper_tip/main.tex). Only its referenced result tables and analysis figures are included. Historical experiment reports, unused result tables, run logs, and build backups are excluded. Numerical table values are preserved from the current manuscript.

```bash
# Requires latexmk, BibTeX, and a LaTeX installation with IEEEtran.
bash paper_tip/build.sh
```

## Tests

```bash
python -m unittest -v tests.test_attribution_gradients
```

Additional tests are under `tests/`; some require optional backbones, datasets, or checkpoints.

## License and attribution

The project retains the upstream [Apache 2.0 license](LICENSE) and Tencent copyright notice. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and the vendored CLIP license. Datasets, model weights, and separately installed external backbones retain their respective licenses.

T2M-GPT: Zhang et al., *T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations*, CVPR 2023. Additional foundations include [CLIP](https://github.com/openai/CLIP), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), and [text-to-motion](https://github.com/EricGuo5513/text-to-motion).
