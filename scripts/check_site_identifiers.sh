#!/bin/bash
# Fail if tracked files (or the given files) contain a cluster username, personal cluster path or Slurm account.
# Site values belong in the untracked leonardo.env (see leonardo.env.example). Patterns are generic on purpose so the
# check itself names no account: CINECA usernames (letters + 2 digits) under a Leonardo path, userexternal/<user>,
# and AI-Factory style account ids (letters_fNN_NNN).
set -euo pipefail
pattern='userexternal/[a-z]|/leonardo_(work|scratch)/[^ )`"]*/[a-z]{3,}[0-9]{2}(/|$)|\b[a-z]{3,}_f[0-9]{2}_[0-9]{3}\b'
if (( $# )); then
  ! grep -niE "$pattern" "$@"
else
  ! git grep -niE "$pattern" -- . ':!*.example'
fi || { echo "site identifiers found (move them to leonardo.env)" >&2; exit 1; }
