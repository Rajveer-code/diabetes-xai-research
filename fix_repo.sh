#!/bin/bash
# =============================================================
#  FIX SCRIPT — run this once inside your project folder
#  Fixes:  1) author name showing as "Your Name"
#           2) adds your real notebooks
#  Path:   /c/diabetes_prediction_project
# =============================================================

set -e

cd /c/diabetes_prediction_project

# ── Step 1: Set correct identity going forward ───────────────
git config user.name  "Rajveer Singh Pall"
git config user.email "rajveerpall04@gmail.com"

echo "Identity set."

# ── Step 2: Rewrite ALL past commit author names ─────────────
# This fixes every commit that currently shows "Your Name"

git filter-branch -f --env-filter '
    OLD_EMAIL="your@email.com"
    CORRECT_NAME="Rajveer Singh Pall"
    CORRECT_EMAIL="rajveerpall04@gmail.com"

    # Fix commits where name is wrong (catches "Your Name" too)
    if [ "$GIT_COMMITTER_NAME" = "Your Name" ] || [ "$GIT_AUTHOR_NAME" = "Your Name" ]; then
        export GIT_COMMITTER_NAME="$CORRECT_NAME"
        export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
        export GIT_AUTHOR_NAME="$CORRECT_NAME"
        export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
    fi
' --tag-name-filter cat -- --branches --tags

echo "Author name rewritten on all commits."

# ── Step 3: Verify the fix worked ────────────────────────────
echo ""
echo "Last 5 commits (check author name):"
git log --format="%ad | %an | %s" --date=format:"%Y-%m-%d" -5

echo ""
echo "============================================================"
echo "  Author fix complete."
echo "  Now copy your real notebooks manually (see instructions)"
echo "  then run: git push origin main --force"
echo "============================================================"
