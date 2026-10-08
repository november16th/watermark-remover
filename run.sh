#!/usr/bin/env bash

# 현재 스크립트 디렉토리로 이동
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 1. 사용할 Python 인터프리터 자동 탐색
CANDIDATES=(
  "/opt/anaconda3/bin/python3"
  "/opt/anaconda3/bin/python"
  "$HOME/opt/anaconda3/bin/python3"
  "$HOME/anaconda3/bin/python3"
  "$HOME/miniconda3/bin/python3"
  "./venv/bin/python3"
  "./.venv/bin/python3"
  "$(which python3 2>/dev/null)"
  "$(which python 2>/dev/null)"
)

PYTHON_BIN=""

for cand in "${CANDIDATES[@]}"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then
    if "$cand" -c "import fastapi, uvicorn" 2>/dev/null; then
      PYTHON_BIN="$cand"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  if command -v conda &>/dev/null; then
    if conda run -n base python -c "import fastapi, uvicorn" 2>/dev/null; then
      PYTHON_BIN="conda run --no-capture-output -n base python"
    elif conda run -n watermark python -c "import fastapi, uvicorn" 2>/dev/null; then
      PYTHON_BIN="conda run --no-capture-output -n watermark python"
    fi
  fi
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "❌ 필요한 패키지(fastapi, uvicorn 등)가 설치된 Python 환경을 찾을 수 없습니다."
  echo "   requirements.txt 패키지 설치 또는 가상환경을 확인해 주세요."
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

# 3. 1.5초 후 브라우저 자동 오픈
(sleep 1.5 && {
  if command -v open &>/dev/null; then
    open "http://127.0.0.1:$PORT"
  elif command -v xdg-open &>/dev/null; then
    xdg-open "http://127.0.0.1:$PORT"
  fi
} 2>/dev/null) &

# 4. 서버 실행
if [[ "$PYTHON_BIN" == conda\ run* ]]; then
  eval exec "$PYTHON_BIN web_app.py"
else
  exec "$PYTHON_BIN" web_app.py
fi
