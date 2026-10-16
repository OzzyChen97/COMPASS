#!/usr/bin/env bash
# Compile the self-contained main article; all tables are in the body.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLED_TEX="$HERE/../.tools/texlive/bin/x86_64-linux"
if [[ -d "$BUNDLED_TEX" ]]; then
  export PATH="$BUNDLED_TEX:$PATH"
fi
command -v latexmk >/dev/null || { echo 'latexmk is required.' >&2; exit 1; }
cd "$HERE"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
if grep -E 'LaTeX Warning: (Reference|Citation).*undefined|There were undefined references|multiply defined|^! ' main.log; then
  echo 'Unresolved LaTeX diagnostics remain.' >&2
  exit 1
fi
printf '\nBuilt %s/main.pdf\n' "$HERE"