#!/usr/bin/env bash
set -euo pipefail

RMAS_URL="https://github.com/recursivemas/recursivemas"
RMAS_COMMIT="38f7da45c1728747979c7a35bf5f66f17e67b1bb"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${ROOT}/refs/recursive_mas"

if [[ ! -d "${DEST}/.git" ]]; then
    git clone --quiet "${RMAS_URL}" "${DEST}"
fi
git -C "${DEST}" checkout --quiet "${RMAS_COMMIT}"

cd "${ROOT}"
uv sync
