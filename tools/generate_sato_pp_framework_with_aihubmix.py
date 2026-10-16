import argparse
import base64
import os
from io import BytesIO
from pathlib import Path

from PIL import Image
from openai import OpenAI


BASE_PROMPT = """
Create a clean, publication-ready research framework figure for a computer vision / AI paper.

Topic:
SATO++ for text-to-motion robustness with relation-conditioned learning.

Figure goal:
Show the full pipeline from paired text input to motion-conditioned attribution, relation-conditioned branching, losses, and final robust text-to-motion generation.

Required blocks:

1. Input block on the far left:
- Original text x
- Paired text x'
- Relation label r in {preserve, alter}
- For alter pairs, edited-slot mask m
Show two example pair types:
  - preserve: semantically equivalent rephrase
  - alter: local semantic edit such as direction / speed / body part / count / order

2. Shared text encoding block:
- Token-level text features H(x) = [h1, h2, ..., hL]
- Same for x'
- Label this as text encoder / token features

3. Motion-conditioned attribution block:
- Early-step autoregressive score:
  s_AR(x, y) = (1/K) sum_{t=1..K} log p(y_t | y_<t, x)
- Grad × Input attribution:
  a_i = || (partial s_AR / partial h_i) elementwise-times h_i ||_1
- Normalized attribution:
  alpha_i = softmax(ReLU(a_i) / tau)
Visually emphasize that attribution identifies motion-relevant tokens.

4. Attribution-weighted semantic representation block:
- Main representation:
  u(x) = sum_i alpha_i phi(h_i)
- Predictor:
  z(x) = q(u(x))
Show the same for x'
Make this look like a semantic pooling module.

5. Relation-conditioned branching in the center-right:

Preserve branch:
- Goal: semantic invariance
- Align loss:
  L_align = -cos(z, stopgrad(z'))
- Variance regularization:
  L_var
- Covariance regularization:
  L_cov
- Preserve loss:
  L_preserve = lambda_align L_align + lambda_var L_var + lambda_cov L_cov
Visually annotate:
  preserve = sentence-level semantic consistency, not token-by-token identity

Alter branch:
- Goal: edited factor changes, non-edited factors remain stable
- Attribution difference on edited slots:
  L_diff = - mean(|alpha - alpha'| on edited slots)
- Locality on non-edited slots:
  L_loc = mean(|alpha - alpha'| on non-edited slots)
- Alter loss:
  L_alter = lambda_diff L_diff + lambda_loc L_loc
Visually annotate:
  local semantic edit only affects edited factor

6. Shared generation supervision block:
- Base text-to-motion generation loss:
  L_gen = - sum_t log p(y_t | y_<t, x)
- Teacher consistency:
  L_teach = KL(teacher || student)
Show a frozen teacher model supervising the student generator.

7. Final objective block on the far right:
- Overall objective:
  L = L_gen + lambda_teach L_teach
      + 1[r=preserve] L_preserve
      + 1[r=alter] L_alter
Use a very short summary only:
  preserve -> invariant semantics
  alter -> localized semantic change

8. Bottom evaluation strip:
- FID
- FID_P
- FID_D
- RepCos
- JSD
- Slot Attribution Shift
Small, neat metric boxes are fine.

Typography:
- All text in English
- Compact but readable
- Equations should be clean and legible
- Prefer symbolic expressions over sentence descriptions
- Prefer 1 to 3 word labels instead of full sentences
- Avoid paragraph-like annotations inside boxes

Constraints:
- No watermark
- No decorative characters
- No 3D rendering
- No hand-drawn style
- No poster aesthetics
- Keep it suitable for direct use in a paper or slide
""".strip()


STYLE_PROMPTS = {
    "cvpr": """
Layout and style:
- CVPR method figure style
- Wide horizontal layout, left-to-right reading order
- White background
- Flat vector scientific diagram
- Minimalistic, dense but clean
- Crisp thin arrows, aligned boxes, consistent spacing
- Professional academic typography
- Soft blue / teal for shared modules
- Green for preserve branch
- Orange / red for alter branch
- Gray for teacher / auxiliary supervision
- Black text
- Very light separators, almost no shadow
- Keep labels concise
- Reduce decorative icons
- Prioritize clarity, modularity, and equation readability
- Make it look like a polished CVPR pipeline figure, not a poster
- Use symbolic notation wherever possible: x, x', r, m, H(x), alpha, u, z, L_gen, L_teach, L_preserve, L_alter
- Replace explanatory sentences with short tags, operators, braces, and equations
- Minimize prose; the figure should read like a compact method diagram
- Keep only essential labels and formulas, not narrative descriptions
""".strip(),
    "default": """
Layout and style:
- Wide horizontal scientific diagram
- White background
- Vector-like flat design
- Conference-paper style
- Neat left-to-right flow with grouped modules, arrows, labels, and small equations
- Professional flat vector scientific diagram
- Soft blue / teal for shared modules
- Green for preserve branch
- Orange / red for alter branch
- Gray for teacher / auxiliary supervision
- Black text
- Minimal shadows
- Sharp arrows and box groupings
""".strip(),
}


def build_prompt(style: str) -> str:
    style_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["default"])
    return f"{BASE_PROMPT}\n\n{style_prompt}".strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a SATO++ framework figure with AIHubMix via gemini-3-pro-image-preview."
    )
    parser.add_argument(
        "--output-dir",
        default="img/generated",
        help="Directory to save the generated figure.",
    )
    parser.add_argument(
        "--filename",
        default="sato_pp_framework_diagram.png",
        help="Output image filename.",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="16:9",
        help="Aspect ratio passed via system prompt, e.g. 16:9, 4:3, 3:2.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-pro-image-preview",
        help="Model name served by AIHubMix.",
    )
    parser.add_argument(
        "--base-url",
        default="https://aihubmix.com/v1",
        help="API base URL.",
    )
    parser.add_argument(
        "--style",
        default="cvpr",
        choices=sorted(STYLE_PROMPTS.keys()),
        help="Prompt style preset.",
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Also save the exact prompt beside the image.",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main():
    args = parse_args()
    api_key = require_env("AIHUBMIX_API_KEY")
    prompt = build_prompt(args.style)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": f"aspect_ratio={args.aspect_ratio}"},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        modalities=["text", "image"],
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.filename
    prompt_path = out_dir / (Path(args.filename).stem + ".prompt.txt")

    parts = response.choices[0].message.multi_mod_content
    if not parts:
        raise RuntimeError("No valid multimodal response received.")

    text_parts = []
    image_saved = False

    for part in parts:
        if "text" in part:
            text_parts.append(part["text"])
        if "inline_data" in part:
            image_data = base64.b64decode(part["inline_data"]["data"])
            image = Image.open(BytesIO(image_data))
            image.save(out_path)
            image_saved = True

    if text_parts:
        print("\n".join(text_parts).strip())

    if args.save_prompt:
        prompt_path.write_text(prompt + "\n", encoding="utf-8")

    if not image_saved:
        raise RuntimeError("Response did not contain image data.")

    print(f"Image saved: {out_path}")
    if args.save_prompt:
        print(f"Prompt saved: {prompt_path}")


if __name__ == "__main__":
    main()
