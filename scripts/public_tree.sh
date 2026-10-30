#!/usr/bin/env bash
# Assemble the public GitHub tree of chai-thermo: the product alone.  The package with the shipped
# family, the configs the family was made from, the tests, the scripts that make and check the
# product, and the user-facing docs.
# No results, corpora, paper, browser app, legacy code, development log (LOG.md, PLAN.md), study
# scripts, study docs, earlier configs or git history: the public tree knows one family.  No
# native/: the reference implementations (KETCH.c, RDAAM.cpp, Mad_He.f90) are third-party sources
# under their authors' terms and are not redistributed (docs/release.md §12).
# Writes a fresh repository with one commit and the release tag.
#
#   scripts/public_tree.sh [out-dir]          # default dist/github/chai
#   cd dist/github/chai && git remote add origin git@github.com:jiaor/CHAI.git && git push -u origin master --tags
#
# The one file over GitHub's 50 MB warning is src/chai/assets/flow_v22_250.pt (55 MB; the hard limit
# is 100 MB).  Zenodo's GitHub integration archives each release of the public repository under a DOI.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$ROOT/dist/github/chai}
cd "$ROOT"
VERSION=$(uv version --short)

# What goes: whole directories, and named files.  Tracked files only, from the working tree as it
# stands (so uncommitted edits to these files go too).
DIRS=(src tests)
FILES=(
  README.md README-pypi.md LICENSE CITATION.cff pyproject.toml uv.lock .python-version .gitignore Makefile MANIFEST.in
  configs/prior_v22.yaml configs/prior_v22_250.yaml configs/observation_v2.yaml
  configs/train_flow_v22_100.yaml configs/train_flow_v22_250.yaml configs/chooser_v22.yaml
  scripts/simulate.py scripts/train_gaussian.py scripts/train_chooser.py scripts/calibration_study.py
  scripts/export_onnx.py scripts/wheel_smoke.py scripts/public_tree.sh
  scripts/validate_ketcham.py scripts/validate_rdaam.py scripts/validate_published.py
  scripts/benchmark_forward.py scripts/speed_accuracy.py
  docs/api.md docs/architecture.md docs/family.md
)

# Once the tree has been pushed its history is public: keep .git and add a commit on top, rather than
# a fresh root that would need a force-push of master.
KEEP=""
if [ -d "$OUT/.git" ]; then KEEP=$(mktemp -d); mv "$OUT/.git" "$KEEP/"; fi
rm -rf "$OUT"
mkdir -p "$OUT"
if [ -n "$KEEP" ]; then mv "$KEEP/.git" "$OUT/"; rmdir "$KEEP"; fi
git ls-files -z -- "${DIRS[@]}" "${FILES[@]}" \
  | grep -zv '^src/chai_thermo.egg-info/' \
  | tar --null -T - -cf - | tar -C "$OUT" -xf -

# Every relative link in the Markdown must resolve inside the tree.
missing=0
while IFS= read -r f; do
  while IFS= read -r target; do
    t=${target%%#*}
    [ -z "$t" ] && continue
    case "$t" in http*|mailto*) continue;; esac
    if [ ! -e "$OUT/$(dirname "$f")/$t" ] && [ ! -e "$OUT/$t" ]; then
      echo "broken link in $f: $target"; missing=$((missing + 1))
    fi
  done < <(grep -o '](\([^)]*\))' "$OUT/$f" | sed 's/^](//; s/)$//')
done < <(cd "$OUT" && find . -name '*.md' | sed 's|^\./||')

# Nothing of the earlier families or the package's earlier names may be in the tree (this script and
# the lock file of third-party packages excepted).
STALE='v2\.1|\bv21\b|prior_v1|prior_v2\b|prior_v2_|prior_v21|prior_v250|flow_v1|flow_v2_|flow_v21|flow_v250|chooser_v1|chooser_v2\b|chooser_v21|corpus_v1|corpus_v2\b|corpus_v21|corpus_v250|latent_v1|latent_v2\b|latent_v21|latent_v250|EARLIER_FAMILIES|results/families|tchron|TChron'
stale=$(grep -rnIE "$STALE" "$OUT" --exclude-dir=.git --exclude='*.pt' --exclude=uv.lock --exclude=public_tree.sh || true)
if [ -n "$stale" ]; then
  echo "earlier-version mentions in the tree:"; echo "$stale" | sed 's/^/  /'
fi

cd "$OUT"
[ -d .git ] || git init -q -b master
git add -A
# The public identity (the private repository's commits carry the machine's hostname).
git config user.name "Ruohong Jiao"
git config user.email "rjiao@uvic.ca"
FROM=$(cd "$ROOT" && git rev-parse --short HEAD)
TRAILER="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Uih9hJPnNZ791SBphZqxPM"
if ! git rev-parse -q --verify HEAD >/dev/null; then
  git commit -q -m "chai-thermo $VERSION: the package with its trained family, configs, tests, scripts and docs

Assembled by scripts/public_tree.sh from the private repository at $FROM.

$TRAILER"
elif ! git diff --cached --quiet; then
  git commit -q -m "chai-thermo $VERSION: update from the private repository at $FROM

$(git diff --cached --stat | tail -1 | sed 's/^ *//')

$TRAILER"
else
  echo "no change since the last commit"
fi
MOVED=""
if git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null; then
  if [ "$(git rev-parse "v$VERSION^{commit}")" != "$(git rev-parse HEAD)" ]; then MOVED=1; fi
  git tag -f -a "v$VERSION" -m "chai-thermo $VERSION" >/dev/null
else
  git tag -a "v$VERSION" -m "chai-thermo $VERSION"
fi

echo "---"
echo "tree: $OUT  ($(git ls-files | wc -l) files, $(du -sh --exclude=.git . | cut -f1), .git $(du -sh .git | cut -f1))"
echo "over 50 MB (GitHub warns; 100 MB is the limit):"
find . -path ./.git -prune -o -type f -size +50M -print | sed 's|^\./|  |'
[ "$missing" -eq 0 ] && echo "every Markdown link resolves" || echo "$missing broken Markdown link(s)"
[ -z "$stale" ] && echo "no earlier-version mentions" || echo "earlier-version mentions found (listed above)"
echo "commit $(git rev-parse --short HEAD), tag v$VERSION; push with:"
if git remote | grep -q origin; then
  echo "  cd $OUT && git push origin master"
  # moving a pushed tag rewrites what the tag names: only before a GitHub release (Zenodo archives the tag)
  if [ -n "$MOVED" ]; then
    echo "  cd $OUT && git push --force origin v$VERSION     # the tag moved; only if no GitHub release uses it yet"
  fi
else
  echo "  cd $OUT && git remote add origin git@github.com:jiaor/CHAI.git && git push -u origin master --tags"
fi
[ "$missing" -eq 0 ] && [ -z "$stale" ]
