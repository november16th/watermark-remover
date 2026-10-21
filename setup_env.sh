#!/usr/bin/env bash
set -e

# 스크립트 위치로 이동
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=========================================================="
echo "🚀 Watermark Remover 환경 자동 설정 (npm postinstall)"
echo "=========================================================="

VENV_DIR="$DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# 1. 기존 가상환경이 정상 작동하는지 빠른 확인
if [ -x "$VENV_PYTHON" ]; then
  if "$VENV_PYTHON" -c "import fastapi, uvicorn, cv2, rembg, simple_lama_inpainting" 2>/dev/null; then
    echo "✅ 이미 모든 패키지가 설치된 가상환경(.venv)이 준비되어 있습니다."
    echo "👉 'npm run dev'를 실행하면 바로 앱이 시작됩니다!"
    exit 0
  fi
fi

# 2. 적절한 시스템 Python 인터프리터 탐색 (Python 3.10 ~ 3.13 권장)
CANDIDATE_COMMANDS=(
  "python3.13"
  "python3.12"
  "python3.11"
  "python3.10"
  "/opt/homebrew/bin/python3.13"
  "/opt/homebrew/bin/python3.12"
  "/opt/homebrew/bin/python3.11"
  "/opt/homebrew/bin/python3.10"
  "/usr/local/bin/python3.13"
  "/usr/local/bin/python3.12"
  "/usr/local/bin/python3.11"
  "/usr/local/bin/python3.10"
  "python3"
  "python"
)

CHOSEN_PYTHON=""

for cmd in "${CANDIDATE_COMMANDS[@]}"; do
  if command -v "$cmd" &>/dev/null || [ -x "$cmd" ]; then
    # 파이썬 버전 확인
    VER_OK=$("$cmd" -c '
import sys
major, minor = sys.version_info[:2]
# 3.10 ~ 3.13 최우선 선호, 그 외 3.10 이상 허용
if major == 3 and 10 <= minor <= 13:
    print("preferred")
elif major == 3 and minor >= 10:
    print("ok")
else:
    print("no")
' 2>/dev/null || echo "no")
    
    if [ "$VER_OK" = "preferred" ]; then
      CHOSEN_PYTHON="$cmd"
      break
    elif [ "$VER_OK" = "ok" ] && [ -z "$CHOSEN_PYTHON" ]; then
      CHOSEN_PYTHON="$cmd"
    fi
  fi
done

if [ -z "$CHOSEN_PYTHON" ]; then
  echo "❌ 지원되는 Python 3 (3.10 이상)을 찾을 수 없습니다."
  echo "   Python 3.10 ~ 3.13 버전을 설치해 주세요."
  echo "   - macOS: brew install python@3.13"
  exit 1
fi

echo "🔍 감지된 Python 인터프리터: $CHOSEN_PYTHON ($("$CHOSEN_PYTHON" --version))"

# 3. 가상환경(.venv) 생성
if [ ! -d "$VENV_DIR" ]; then
  echo "📦 독립된 Python 가상환경(.venv)을 생성합니다..."
  "$CHOSEN_PYTHON" -m venv "$VENV_DIR"
fi

# 4. pip 업그레이드 및 requirements.txt 패키지 설치
echo "⚡ 필요한 AI/영상처리 패키지를 설치합니다 (최초 1회만 진행)..."
"$VENV_PIP" install --upgrade pip --quiet

if [ -f "requirements.txt" ]; then
  echo "📥 requirements.txt 패키지 다운로드 및 설치 중..."
  "$VENV_PIP" install -r requirements.txt
else
  echo "❌ requirements.txt 파일이 존재하지 않습니다."
  exit 1
fi

# 5. FFmpeg 설치 여부 확인 안내
if ! command -v ffmpeg &>/dev/null; then
  echo "⚠️ [주의] 시스템에 'ffmpeg'가 설치되어 있지 않습니다."
  echo "   오디오 무손실 합성을 위해 ffmpeg 설치를 권장합니다:"
  echo "   - macOS: brew install ffmpeg"
  echo "   - Ubuntu: sudo apt install -y ffmpeg"
else
  echo "🎬 FFmpeg 감지 완료: $(command -v ffmpeg)"
fi

echo "=========================================================="
echo "🎉 모든 환경 설정이 성공적으로 완료되었습니다!"
echo "👉 실행 방법: npm run dev"
echo "=========================================================="
