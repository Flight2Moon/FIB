#!/bin/bash

set -e

PROJECT_DIR="${1:-$(pwd)}"

cd "$PROJECT_DIR" || {
  echo "프로젝트 디렉토리로 이동 실패: $PROJECT_DIR"
  exit 1
}

DEVICE_NAME=$(hostname)
USER_NAME=$(whoami)
NOW=$(date "+%Y-%m-%d %H:%M:%S %Z")
BRANCH=$(git branch --show-current)

echo "Project : $PROJECT_DIR"
echo "Branch  : $BRANCH"
echo "Device  : $DEVICE_NAME"
echo "User    : $USER_NAME"
echo "Time    : $NOW"
echo

# Git 저장소인지 확인
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "여기는 Git 저장소가 아닙니다."
  exit 1
fi

# 원격 저장소 확인
if ! git rev-parse --abbrev-ref --symbolic-full-name @{u} > /dev/null 2>&1; then
  echo "현재 브랜치에 upstream 원격 브랜치가 설정되어 있지 않습니다."
  echo "예: git push -u origin $BRANCH"
  exit 1
fi

echo "원격 저장소 상태 확인 중..."
git fetch

LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u})
BASE=$(git merge-base @ @{u})

if [ "$LOCAL" = "$REMOTE" ]; then
  echo "로컬과 원격이 같은 상태입니다."
elif [ "$LOCAL" = "$BASE" ]; then
  echo "로컬이 원격보다 뒤처져 있습니다. pull 실행..."
  git pull
elif [ "$REMOTE" = "$BASE" ]; then
  echo "로컬이 원격보다 앞서 있습니다. push 필요 상태입니다."
else
  echo "로컬과 원격이 서로 다르게 수정되었습니다."
  echo "수동 merge 또는 rebase가 필요합니다."
  exit 1
fi

# 변경사항 확인
if [ -n "$(git status --porcelain)" ]; then
  echo "변경사항 발견. commit 생성 중..."

  git add .

  COMMIT_MSG="sync: $NOW from $DEVICE_NAME by $USER_NAME"

  git commit -m "$COMMIT_MSG"

  echo "push 실행..."
  git push

  echo "동기화 완료."
else
  echo "커밋할 변경사항이 없습니다."

  # 이미 commit은 되어 있는데 아직 push 안 된 경우 처리
  AHEAD_COUNT=$(git rev-list --count @{u}..@)

  if [ "$AHEAD_COUNT" -gt 0 ]; then
    echo "아직 push되지 않은 commit이 있습니다. push 실행..."
    git push
  fi

  echo "완료."
fi