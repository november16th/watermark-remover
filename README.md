# 🎬 영상 워터마크 제거 도구

Gemini AI 등이 생성한 영상의 워터마크를 AI 인페인팅으로 자연스럽게 제거하는 Python CLI 도구입니다.

## ✨ 주요 기능

- **LaMa AI 인페인팅** — 딥러닝 기반 고품질 워터마크 제거
- **ROI 크롭 최적화** — 워터마크 주변만 처리하여 10초 영상을 ~8초에 완료
- **Apple Silicon GPU 가속** — MPS 자동 감지 및 활용
- **자동 워터마크 감지** — 모서리 영역에서 워터마크 위치 자동 탐지
- **마우스 선택** — GUI로 워터마크 영역을 직접 선택
- **오디오 보존** — FFmpeg로 원본 오디오 무손실 합성
- **일괄 처리** — 여러 파일을 한 번에 처리

## 📦 설치

### 1. Python 환경 준비

> **참고:** 이 도구는 Anaconda Python 3.13 환경에서 실행됩니다.
> 패키지는 `/opt/anaconda3/bin/python3`에 설치되어 있습니다.

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. FFmpeg 설치 (오디오 보존에 필요)

```bash
# macOS
brew install ffmpeg
```

## 🚀 사용법

### 기본 사용 (LaMa AI, 우측 상단 자동 감지)

```bash
python3 remove_watermark.py input.mp4
# 또는 Anaconda python 직접 지정
/opt/anaconda3/bin/python3 remove_watermark.py input.mp4
```

### 출력 파일 지정

```bash
python remove_watermark.py input.mp4 -o clean_video.mp4
```

### 워터마크 좌표 직접 지정

```bash
# x,y,w,h 형식 (좌측 상단 기준)
python remove_watermark.py input.mp4 --region 1860,20,50,50
```

### 마우스로 영역 선택

```bash
python remove_watermark.py input.mp4 --select
```

### OpenCV 모드 (빠르지만 품질 낮음)

```bash
python remove_watermark.py input.mp4 --method opencv
```

### 여러 파일 일괄 처리

```bash
python remove_watermark.py video1.mp4 video2.mp4 video3.mp4
python remove_watermark.py *.mp4
```

### 워터마크가 다른 모서리에 있을 때

```bash
# 좌측 상단
python remove_watermark.py input.mp4 --corner top-left

# 우측 하단
python remove_watermark.py input.mp4 --corner bottom-right
```

## ⚙️ 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-o`, `--output` | `{입력파일}_clean.mp4` | 출력 파일 경로 |
| `--method` | `lama` | 인페인팅 방법 (`lama` 또는 `opencv`) |
| `--region` | 자동 감지 | 워터마크 좌표 `x,y,w,h` |
| `--corner` | `top-right` | 자동 감지할 모서리 |
| `--select` | - | 마우스로 영역 선택 |
| `--margin` | `28` | ROI 마진 (px) |
| `--ema` | `0.85` | Temporal EMA 계수 (깜빡임 방지) |
| `--crf` | `18` | 출력 품질 (0-51, 낮을수록 고품질) |

## 🔧 동작 원리

1. **프레임 읽기** — OpenCV로 영상을 프레임 단위로 읽음
2. **ROI 크롭** — 워터마크 주변 ~256×256 영역만 추출
3. **AI 인페인팅** — LaMa 모델이 워터마크를 주변 패턴으로 자연스럽게 채움
4. **Temporal EMA** — 프레임 간 부드러운 전환으로 깜빡임 방지
5. **Feathered 블렌딩** — 처리된 영역을 원본과 자연스럽게 합성
6. **오디오 합성** — FFmpeg로 원본 오디오 무손실 복사

## 📝 참고사항

- 첫 실행 시 LaMa 모델을 자동 다운로드합니다 (~200MB)
- Apple Silicon Mac에서는 MPS GPU 가속이 자동 적용됩니다
- Intel Mac 또는 GPU 없는 환경에서는 CPU로 동작합니다 (ROI 크롭 덕분에 여전히 빠름)
- `--method opencv`는 모델 다운로드 없이 즉시 사용 가능합니다
