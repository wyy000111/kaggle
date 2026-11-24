#!/usr/bin/env bash
# 凭证 fallback：优先 KAGGLE_API_TOKEN，未配置则退到 KAGGLE_USERNAME + KAGGLE_KEY
# 用法：kaggle.sh kernels push -p ./kaggle_task
set -euo pipefail

if [ -n "${KAGGLE_API_TOKEN:-}" ]; then
  echo "[kaggle.sh] Using KAGGLE_API_TOKEN"
  unset KAGGLE_USERNAME KAGGLE_KEY
elif [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  echo "[kaggle.sh] Using KAGGLE_USERNAME + KAGGLE_KEY"
  unset KAGGLE_API_TOKEN
else
  echo "::error::未配置任何 Kaggle 凭证：请设置 KAGGLE_API_TOKEN 或 KAGGLE_USERNAME + KAGGLE_KEY"
  exit 1
fi

exec kaggle "$@"
