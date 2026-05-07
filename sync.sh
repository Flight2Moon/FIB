#!/bin/bash

set -e

ACTION="$1"
PROJECT_DIR="${2:-$(pwd)}"

cd "$PROJECT_DIR" || exit 1

DEVICE=$(hostname)
USER_NAME=$(whoami)
NOW=$(date "+%Y-%m-%d %H:%M:%S %Z")
BRANCH=$(git branch --show-current)

if [ "$ACTION" = "pull" ]; then
  echo "[$NOW] Pull from GitHub..."
  echo "Device: $DEVICE / User: $USER_NAME"
  git pull origin "$BRANCH" --no-rebase

elif [ "$ACTION" = "push" ]; then
  echo "[$NOW] Push to GitHub..."
  echo "Device: $DEVICE / User: $USER_NAME"

  git status

  if [ -z "$(git status --porcelain)" ]; then
    echo "변경사항이 없습니다."
    exit 0
  fi

  git add .

  COMMIT_MSG="sync: $NOW from $DEVICE by $USER_NAME"
  git commit -m "$COMMIT_MSG"

  git push origin "$BRANCH"

else
  echo "사용법:"
  echo "  ./git_sync.sh pull [프로젝트_경로]"
  echo "  ./git_sync.sh push [프로젝트_경로]"
  echo
  echo "예시:"
  echo "  ./sync.sh force-pull"
  echo "  ./git_sync.sh pull"
  echo "  ./git_sync.sh push"
  echo "  ./git_sync.sh pull /home/jihoney/workdir/main_workdir/FIB/FIB_M3_jh/FIB-M3_v4_6var_jh_0506"
  exit 1
fi