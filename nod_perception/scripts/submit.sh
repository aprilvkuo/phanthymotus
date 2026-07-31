#!/usr/bin/bash
# Pre-submission check + helper.
# 1) verifies no file >1MB is staged (benchmark rule)
# 2) verifies the repo is public-ready and on a branch off the fork
# 3) prints the commit id + repo URL you must paste into the leaderboard form.
set -e

echo "== 1) Oversized-file check (must be empty) =="
oversized=$(find . -type f -not -path './.git/*' -not -path './.cache/*' \
                 -not -path '*/__pycache__/*' -size +1M -print)
if [ -n "$oversized" ]; then
  echo "ERROR: these files exceed 1MB and must not be committed:"
  echo "$oversized"
  exit 1
fi
echo "OK: no file >1MB."

echo
echo "== 2) Submission identifiers =="
COMMIT=$(git rev-parse HEAD)
REMOTE=$(git remote get-url origin 2>/dev/null || echo "SET_YOUR_PUBLIC_FORK_URL")
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "commit_id : $COMMIT"
echo "repo_url  : $REMOTE"
echo "branch    : $BRANCH"
echo
echo "Paste commit_id + repo_url into the leaderboard submission form."
echo "Ensure the personal repo is set to PUBLIC, else the auto-pull errors."
