# ✨ AI Video Watermark Remover (영상 워터마크 정밀 제거 도구)

Gemini AI 등 생성형 AI로 제작된 영상의 워터마크(✦ 스파클 다이아몬드 로고 등)를 **딥러닝 인페인팅(LaMa AI)** 기술로 흔적 없이 제거하는 도구입니다.

단순 사각형 블러 방식이 아닌, **정밀 형태 마스크(Shape Mask)**를 사용하여 멀쩡한 배경(마룻바닥, 옷자락 결 등)을 온전히 보존하므로 **동영상 재생 시 발생하는 일렁거림·부글거림을 억제**합니다.

웹 브라우저 UI와 CLI(터미널) 방식을 모두 지원합니다.

---

## 🌟 주요 특징

- **정밀 스파클 마스크 (Shape Mask)**: 워터마크 아이콘 형태(외곽선)만 정밀하게 따서 복원하여 배경 훼손 및 일렁거림 최소화
- **웹 UI 마우스 지정**: 브라우저 화면에서 마우스 드래그로 워터마크 위치 지정 및 실시간 마스크 미리보기 제공
- **자동 다운로드**: 작업 완료 즉시 원본 파일명 기반 `{파일명}_removed_watermark.mp4`로 자동 저장
- **초고속 ROI 크롭 최적화**: 1080p 전체가 아닌 워터마크 주변부만 잘라내어 처리 (10초 영상 기준 약 8~10초 소요)
- **Apple Silicon GPU 가속**: Mac M1/M2/M3/M4의 Metal Performance Shaders(MPS) 자동 활용
- **무손실 오디오 보존**: FFmpeg 스트림 복사를 통해 원본 음질을 100% 유지

---

## 💻 필수 요구 사양 및 설치 파일

### 1. 시스템 요구 사항
- **OS**: macOS (Apple Silicon M1/M2/M3/M4 또는 Intel) / Linux / Windows
- **Python**: Python 3.10 ~ 3.13 권장 (Anaconda 권장)

### 2. FFmpeg 설치 (오디오 무손실 합성용)
동영상의 음성을 원본 그대로 보존하기 위해 시스템에 `ffmpeg`가 필요합니다.

- **macOS (Homebrew)**:
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu/Debian**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **Windows**:
  - [FFmpeg 공식 홈페이지](https://ffmpeg.org/download.html)에서 다운로드 후 환경 변수(PATH)에 등록하거나 `winget install Gyan.FFmpeg` 실행

---

## 📦 패키지 설치 방법

### 1. 저장소 복제 (Clone)
```bash
git clone https://github.com/november16th/watermark-remover.git
cd watermark-remover
```

### 2. 파이썬 의존성 패키지 설치
```bash
pip install -r requirements.txt
```
> **설치되는 주요 라이브러리:**
> - `opencv-python`, `numpy`, `Pillow`, `tqdm`
> - `torch`, `torchvision`, `simple-lama-inpainting` (LaMa AI 모델)
> - `fastapi`, `uvicorn`, `python-multipart` (웹 UI 구동용)

---

## 🚀 사용 방법

### 방법 1. 웹 브라우저 UI로 사용하기 (가장 추천! 👍)

1. 터미널에서 웹 서버를 실행합니다:
   ```bash
   python web_app.py
   # 또는 특정 Python 경로 사용 시:
   /opt/anaconda3/bin/python3 web_app.py
   ```

2. 브라우저를 열고 아래 주소로 접속합니다:
   👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

3. 사용 절차:
   - 동영상 파일을 화면에 드래그하거나 클릭하여 업로드합니다.
   - 첫 프레임 화면에서 **워터마크 주변을 마우스로 드래그**하여 영역을 지정합니다.
   - 우측의 **[정밀 마스크 미리보기]**에서 스파클 다이아몬드 윤곽선이 제대로 잡혔는지 확인합니다.
   - **[🚀 워터마크 제거 시작]** 버튼을 클릭합니다.
   - 처리가 완료되면 깨끗해진 영상이 화면에 재생되며, 브라우저에서 **`{기본파일명}_removed_watermark.mp4`** 파일로 자동 다운로드됩니다.
   - 새 영상을 작업하려면 브라우저를 새로고침(`Cmd + R` 또는 `F5`)하면 됩니다.

---

### 방법 2. 터미널 CLI로 사용하기

대량의 영상 일괄 처리나 터미널 작업 시 사용합니다.

```bash
# 기본 사용 (우측 하단 워터마크 자동 감지 후 즉시 처리)
python remove_watermark.py input.mp4

# 마우스로 첫 프레임에서 워터마크 영역 선택하기
python remove_watermark.py input.mp4 --select

# 처리 전 감지된 영역 미리보기 이미지만 저장
python remove_watermark.py input.mp4 --preview

# 출력 파일 경로 직접 지정
python remove_watermark.py input.mp4 -o clean_output.mp4

# 워터마크 좌표 직접 지정 (x,y,w,h)
python remove_watermark.py input.mp4 --region 568,1139,50,50

# 여러 영상 일괄 처리 (배치)
python remove_watermark.py video1.mp4 video2.mp4 video3.mp4
python remove_watermark.py *.mp4
```

#### CLI 옵션 안내

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-o`, `--output` | `{파일명}_clean.mp4` | 출력 동영상 저장 경로 |
| `--method` | `lama` | 인페인팅 엔진 (`lama`: 고품질 AI, `opencv`: 빠른 일반 모드) |
| `--select` | - | 화면에 첫 프레임을 띄워 마우스 드래그로 영역 선택 |
| `--preview` | - | 워터마크 영역이 표시된 미리보기 이미지만 저장하고 종료 |
| `--region` | 자동 감지 | 워터마크 좌표 `x,y,w,h` 직접 입력 |
| `--corner` | `bottom-right` | 자동 감지 탐색 모서리 (`bottom-right`, `top-right` 등) |
| `--crf` | `18` | 비디오 인코딩 품질 (0~51, 낮을수록 고화질) |
| `--ema` | `0.85` | 프레임 간 일렁거림 방지 시간축 스무딩 강도 (0.0~1.0) |

---

## 🔧 동작 원리

1. **프레임 추출 및 ROI 크롭**: 영상의 첫 프레임과 워터마크 좌표를 기반으로 연산에 필요한 최소 주변부(Margin 포함)만 추출
2. **정밀 Shape Masking**: 박스 내부의 휘도/윤곽선을 분석하여 워터마크 픽셀만 타이트하게 3~4px 마스킹
3. **LaMa FFC Inpainting**: 고주파 푸리에 합성 기반 LaMa 모델로 결손 영역 자연 복원
4. **Temporal EMA 블렌딩**: 이전 프레임의 복원 결과를 일정 비율 가중 합성하여 프레임 간 깜빡임 억제
5. **Feathered Blending**: 복원된 패치를 부드러운 가우시안 경계선으로 원본 프레임에 합성
6. **FFmpeg 무손실 오디오 머지**: 원본 오디오 스트림을 손실 없이 패키징하여 최종 MP4 생성

---

## 💡 팁 및 참고사항

- 최초 1회 실행 시 LaMa 모델 체크포인트 파일(약 200MB)이 자동 다운로드되어 로컬 캐시에 저장됩니다.
- Gemini 생성 영상의 워터마크는 기본적으로 우측 하단 플레이어 컨트롤러 윗부분에 위치합니다.
