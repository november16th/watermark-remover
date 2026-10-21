#!/usr/bin/env bash

# 현재 스크립트 디렉토리로 이동
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

VENV_PYTHON="$DIR/.venv/bin/python"

# 1. 가상환경 및 패키지 검증 (없으면 자동으로 setup_env.sh 실행)
if [ ! -x "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "⚙️ 실행 환경이 준비되지 않았습니다. 환경 자동 설정을 시작합니다..."
  ./setup_env.sh
fi

PYTHON_BIN="$VENV_PYTHON"

# 만약 여전히 .venv가 없으면 시스템 인터프리터 탐색 fallback
if [ ! -x "$PYTHON_BIN" ]; then
  CANDIDATES=(
    "./venv/bin/python"
    "/opt/anaconda3/bin/python3"
    "$(which python3 2>/dev/null)"
  )
  for cand in "${CANDIDATES[@]}"; do
    if [ -n "$cand" ] && [ -x "$cand" ] && "$cand" -c "import fastapi, uvicorn" 2>/dev/null; then
      PYTHON_BIN="$cand"
      break
    fi
  done
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "❌ 서버를 실행할 수 있는 Python 환경을 찾지 못했습니다."
  echo "   'npm install'을 실행하여 환경을 먼저 구축해 주세요."
  exit 1
fi

echo "✨ Python 인터프리터: $PYTHON_BIN"

# 2. 포트 8001 기존 프로세스 정리 (포트 충돌 방지)
PORT=8001
EXISTING_PID=$(lsof -ti :$PORT 2>/dev/null)
if [ -n "$EXISTING_PID" ]; then
  echo "🔄 포트 $PORT 를 점유 중인 기존 프로세스(PID: $EXISTING_PID)를 종료합니다..."
  kill -9 $EXISTING_PID 2>/dev/null || true
  sleep 1
fi

# 3. 브라우저 자동 오픈
(sleep 1.5 && {
  if command -v open &>/dev/null; then
    open "http://127.0.0.1:$PORT"
  elif command -v xdg-open &>/dev/null; then
    xdg-open "http://127.0.0.1:$PORT"
  fi
} 2>/dev/null) &

# 4. 서버 실행
echo "🚀 워터마크 제거 웹 서버를 시작합니다 (http://127.0.0.1:$PORT)..."
exec "$PYTHON_BIN" web_app.py
