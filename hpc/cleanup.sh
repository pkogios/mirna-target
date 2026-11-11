#!/bin/bash
# Leave nothing behind on the cluster. Run from the project folder on ARIS *after*
# results/ has been copied back (git push, or scp/rsync to your own machine).
#
#   bash hpc/cleanup.sh          # shows what would be removed
#   bash hpc/cleanup.sh --yes    # removes it
set -euo pipefail
TARGETS=(data artifacts logs .venv)

if [[ "${1:-}" != "--yes" ]]; then
    echo "Would remove (run again with --yes):"
    for target in "${TARGETS[@]}"; do [[ -e "$target" ]] && du -sh "$target"; done
    exit 0
fi
if [[ -n "$(git status --porcelain results 2>/dev/null)" ]]; then
    echo "results/ has uncommitted changes. Commit and push (or copy) them first." >&2
    exit 1
fi
for target in "${TARGETS[@]}"; do rm -rf -- "$target"; done
echo "Removed: ${TARGETS[*]}. You can now delete this folder itself."
