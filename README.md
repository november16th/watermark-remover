# ✨ AI Video Watermark Remover (영상 워터마크 정밀 제거 도구)

Gemini AI 등 생성형 AI로 제작된 영상의 워터마크(✦ 스파클 다이아몬드 로고 등)를 **딥러닝 인페인팅(LaMa AI)** 기술로 흔적 없이 제거하는 고성능 도구입니다.

단순 사각형 박스를 통째로 밀어버리는 기존 방식과 달리, **워터마크 형태만 따내는 정밀 형태 마스크(Shape Mask)**, **고속 푸리에 합성곱(FFC) 기반 AI 복원**, **프레임 간 일렁거림을 억제하는 시간축 EMA 스무딩**을 결합하여 고품질의 자연스러운 결과물을 제공합니다.

웹 브라우저 UI와 CLI(터미널) 방식을 모두 지원합니다.

---

## 📑 목차
1. [🌟 주요 특징](#-주요-특징)
2. [💻 시스템 요구 사항 및 설치](#-시스템-요구-사항-및-설치)
3. [🚀 소스코드 상세 사용법](#-소스코드-상세-사용법)
   - [방법 1. 웹 브라우저 UI로 사용하기 (추천)](#방법-1-웹-브라우저-ui로-사용하기-추천-)
   - [방법 2. 터미널 CLI로 사용하기](#방법-2-터미널-cli로-사용하기)
   - [방법 3. 파이썬 스크립트에서 직접 모듈로 호출하기](#방법-3-파이썬-스크립트에서-직접-모듈로-호출하기)
4. [🔬 전체 동작 파이프라인 (How It Works)](#-전체-동작-파이프라인-how-it-works)
5. [🧠 핵심 라이브러리 및 동작 원리 (Libraries & Principles)](#-핵심-라이브러리-및-동작-원리-libraries--principles)
   - [1. LaMa AI (Large Mask Inpainting via Fast Fourier Convolutions)](#1-lama-ai-large-mask-inpainting-via-fast-fourier-convolutions)
   - [2. OpenCV 영상 처리 및 정밀 형태 마스킹 (Shape Masking)](#2-opencv-영상-처리-및-정밀-형태-마스킹-shape-masking)
   - [3. 시간축 스무딩 알고리즘 (Temporal Exponential Moving Average)](#3-시간축-스무딩-알고리즘-temporal-exponential-moving-average)
   - [4. 알파 페더링 블렌딩 (Gaussian Feathering)](#4-알파-페더링-블렌딩-gaussian-feathering)
   - [5. FFmpeg 무손실 오디오 패스스루 & 최적화 인코딩](#5-ffmpeg-무손실-오디오-패스스루--최적화-인코딩)
   - [6. FastAPI & HTML5 Canvas 웹 아키텍처](#6-fastapi--html5-canvas-웹-아키텍처)
6. [📂 프로젝트 디렉토리 구조](#-프로젝트-디렉토리-구조)

---

## 🌟 주요 특징

- **정밀 스파클 마스크 (Shape Mask)**: 워터마크 아이콘 윤곽선만 타이트하게 분리하여 마룻바닥 결, 옷주름 등 주변 배경 손실을 최소화
- **일렁거림/깜빡임 억제 (Anti-Flicker)**: 동영상 인페인팅 특유의 프레임 간 픽셀 불일치(Boiling Effect)를 시간축 EMA 필터로 억제
- **직관적인 웹 UI**: 브라우저 드래그 앤 드롭 업로드, 캔버스 마우스 클릭/드래그 지정, **키보드 방향키 1px 미세조정**
- **자동 다운로드**: 워터마크 제거 완료 시 원본 파일명 기반 `{파일명}_removed_watermark.mp4`로 브라우저 즉시 자동 다운로드
- **초고속 ROI 크롭 최적화**: 1080p 전체 화면 대신 워터마크 국소 주변부(Margin 포함)만 크롭하여 처리 (10초 클립 기준 약 8~10초 내외 처리)
- **Apple Silicon & CUDA 가속**: Mac M1/M2/M3/M4 Metal(MPS) 및 NVIDIA GPU 가속 자동 지원
- **무손실 음질 보존**: FFmpeg 패스스루를 통해 원본 오디오 스트림(AAC/MP3 등)을 재압축 없이 100% 비손실 보존

---

## 💻 시스템 요구 사항 및 설치

### 1. 시스템 요구 사양
- **운영체제**: macOS (Apple Silicon M1/M2/M3/M4 또는 Intel), Linux, Windows
- **Python**: Python 3.10 ~ 3.13 (Anaconda 가상환경 적극 권장)
- **하드웨어 가속**: Apple Silicon MPS 또는 NVIDIA CUDA 지원 GPU (CPU만으로도 구동 가능)

### 2. FFmpeg 설치 (오디오 무손실 합성 필수)
시스템에 `ffmpeg`가 설치되어 있어야 최종 영상에 오디오가 무손실로 합성됩니다.

- **macOS (Homebrew)**:
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu / Debian**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **Windows**:
  ```powershell
  winget install Gyan.FFmpeg
  ```

---

## 📦 패키지 설치 방법

### 1. 저장소 복제 (Git Clone)
```bash
git clone https://github.com/november16th/watermark-remover.git
cd watermark-remover
```

### 2. 파이썬 가상환경 생성 (권장)
```bash
# conda 사용 시
conda create -n watermark python=3.10 -y
conda activate watermark

# 또는 venv 사용 시
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 라이브러리 설치
```bash
pip install -r requirements.txt
```

> **설치되는 핵심 패키지:**
> - `torch`, `torchvision`: 딥러닝 텐서 연산 및 신경망 엔진
> - `simple-lama-inpainting`: LaMa (Large Mask Inpainting) 사전학습 모델 구동기
> - `opencv-python`, `numpy`, `Pillow`: 컴퓨터 비전 및 이미지 행렬 연산
> - `fastapi`, `uvicorn`, `python-multipart`: 고성능 비동기 웹 프레임워크 및 서버
> - `tqdm`: 터미널 작업 진행률 표시바

---

## 🚀 소스코드 상세 사용법

### 방법 1. 웹 브라우저 UI로 사용하기 (추천 👍)

누구나 직관적으로 마우스와 키보드로 조작할 수 있는 전용 웹 인터페이스입니다.

1. **웹 서버 실행**:
   ```bash
   python web_app.py
   # 특정 파이썬 인터프리터 경로를 사용할 경우:
   # /opt/anaconda3/bin/python3 web_app.py
   ```
   *서버가 기동되면 콘솔에 `INFO: Uvicorn running on http://127.0.0.1:8000` 문구가 출력됩니다.*

2. **브라우저 접속**:
   크롬 또는 사파리 브라우저를 열고 다음 주소로 이동합니다:
   👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

3. **작업 순서**:
   1. **동영상 업로드**: 화면 중앙 점선 상자에 MP4/MOV/WebM 영상을 **드래그 앤 드롭**하거나 클릭하여 선택합니다.
   2. **위치 지정**:
      - 영상 첫 프레임이 나타나면 워터마크 위치를 **마우스로 드래그하거나 클릭**합니다.
      - **키보드 방향키 (`←`, `→`, `↑`, `↓`)**: 워터마크 박스를 **1px씩 세밀하게 이동**할 수 있습니다.
      - **`Shift` + 방향키**: 5px씩 빠르게 이동합니다.
   3. **마스크 확인**:
      - 우측 **[마스크 미리보기]** 창에 다이아몬드 스파클 윤곽선이 제대로 잡혔는지 실시간 확인합니다.
      - 필요 시 **[마스크 여유 크기]** 슬라이더로 테두리 여백(2px~12px)을 조절합니다.
   4. **워터마크 제거 시작**:
      - **[워터마크 제거 시작]** 버튼을 클릭하면 화면 정중앙에 프로그레스 바가 뜨며 작업이 진행됩니다.
      - 완료 즉시 `{기본파일명}_removed_watermark.mp4` 파일로 **자동 다운로드**됩니다.

---

### 방법 2. 터미널 CLI로 사용하기

대량 영상 일괄 처리(배치)나 서버 자동화 파이프라인에서 사용하기 적합합니다.

#### 1) 기본 자동 감지 실행
영상 우측 하단의 워터마크를 스스로 감지하여 기본 처리합니다:
```bash
python remove_watermark.py input.mp4
# 결과물: input_clean.mp4 생성
```

#### 2) 화면에서 마우스로 직접 영역 선택 (`--select`)
첫 프레임 GUI 창을 띄워 마우스로 드래그하여 정확한 위치를 지정합니다:
```bash
python remove_watermark.py input.mp4 --select
```

#### 3) 좌표 직접 지정 (`--region x,y,w,h`)
워터마크의 정확한 픽셀 좌표를 알고 있을 때 사용합니다:
```bash
python remove_watermark.py input.mp4 --region 560,1130,70,70
```

#### 4) 미리보기 이미지 확인 (`--preview`)
비디오를 렌더링하지 않고 감지된 영역이 표시된 확인용 이미지만 저장합니다:
```bash
python remove_watermark.py input.mp4 --preview
```

#### 5) 폴더 내 모든 영상 일괄(Batch) 처리
```bash
# 특정 여러 파일
python remove_watermark.py clip1.mp4 clip2.mp4 clip3.mp4

# 와일드카드 패턴 사용
python remove_watermark.py ./videos/*.mp4
```

#### 6) CLI 옵션 상세 안내

| 옵션 플래그 | 기본값 | 설명 |
| :--- | :---: | :--- |
| `-o`, `--output` | `{파일명}_clean.mp4` | 출력 동영상 파일 경로 지정 |
| `--method` | `lama` | 인페인팅 엔진 선택 (`lama`: 고품질 딥러닝, `opencv`: 빠른 CPU 알고리즘) |
| `--select` | `False` | 첫 프레임을 GUI 창으로 띄워 마우스 드래그로 영역 선택 |
| `--region` | 자동 감지 | 워터마크 좌표 `x,y,w,h` 수동 지정 |
| `--corner` | `bottom-right` | 워터마크 자동 탐색 모서리 (`bottom-right`, `top-right`, `bottom-left`, `top-left`) |
| `--crf` | `18` | H.264 인코딩 품질 (0~51, 낮을수록 고화질, 18은 시각적 무손실 수준) |
| `--ema` | `0.85` | 프레임 간 일렁거림 방지 시간축 스무딩 가중치 ($0.0 \sim 1.0$) |
| `--margin` | `28` | 워터마크 주변부 인페인팅 연산 버퍼 크기 (px) |
| `--preview` | `False` | 처리 없이 워터마크 감지 위치가 표시된 PNG 이미지만 생성 후 종료 |

---

### 방법 3. 파이썬 스크립트에서 직접 모듈로 호출하기

다른 파이썬 프로젝트나 백엔드 서비스에 라이브러리 형태로 내장할 수 있습니다:

```python
from remove_watermark import process_video

# 동영상 워터마크 제거 실행
success = process_video(
    input_path="input.mp4",
    output_path="output_cleaned.mp4",
    watermark_box=(568, 1140, 70, 70),  # (x, y, w, h)
    method="lama",                      # 'lama' (AI) 또는 'opencv'
    margin=28,                          # ROI 마진
    ema_alpha=0.85,                     # 시간축 스무딩 강도
    crf=18                              # x264 화질 계수
)

if success:
    print("워터마크 제거 및 비디오 인코딩 성공!")
```

---

## 🔬 전체 동작 파이프라인 (How It Works)

전체 워터마크 제거 파이프라인은 다음과 같은 7단계 정밀 과정을 거칩니다:

```
[입력 동영상]
     │
     ▼
[Step 1: 프레임 디코딩 & 메타데이터 추출] (OpenCV VideoCapture)
     │
     ▼
[Step 2: 워터마크 ROI 좌표 확정] (웹 드래그 / 방향키 미세조정 / 자동 감지)
     │
     ▼
[Step 3: 정밀 형태 마스킹 (Shape Mask)] (밝기 분석 + Canny Edge + 4점 기하 템플릿)
     │
     ▼
┌────┴────────────────────────────────────────────────┐
│  매 프레임 순회 처리 루프 (Per-Frame Loop)          │
│                                                     │
│  [Step 4: 국소 ROI 크롭 & LaMa AI 인페인팅 추론]    │
│           (FFC 푸리에 합성곱 기반 결손 복원)        │
│                           │                         │
│  [Step 5: 프레임 간 Temporal EMA 스무딩]            │
│           (이전 복원 프레임과의 가중 혼합)          │
│                           │                         │
│  [Step 6: 가우시안 알파 페더링 합성]                │
│           (배경 원본과 복원 패치 자연스럽게 결합)   │
└────┬────────────────────────────────────────────────┘
     │
     ▼
[Step 7: FFmpeg 무손실 오디오 머지 & WebM/MP4 패키징]
     │
     ▼
[최종 결과물 다운로드]
```

1. **Step 1: 프레임 디코딩 & 메타데이터 분석**:
   `cv2.VideoCapture`를 통해 영상의 해상도, FPS, 총 프레임 수, 색상 프로파일을 읽어옵니다.
2. **Step 2: 워터마크 ROI(관심 영역) 확정**:
   사용자가 지정한 좌표 `(x, y, w, h)`에 컨텍스트 파악을 위한 버퍼 마진(기본 28px)을 더해 연산 대상 영역인 ROI를 설정합니다.
3. **Step 3: 정밀 형태 마스킹 (Shape Masking)**:
   배경을 네모나게 뭉개지 않도록, 지정된 박스 내부에서 다이아몬드 스파클 형태의 픽셀만 마스크(255)로 추출하고 나머지 배경(0)은 보존합니다.
4. **Step 4: 국소 ROI 크롭 & 딥러닝 인페인팅 추론**:
   1080×1920 전체 프레임을 신경망에 넣으면 극심한 연산 지연이 발생하므로, 워터마크 주변의 작은 ROI 패치(예: 120×120)만 잘라내어 LaMa AI 모델에 전달합니다.
5. **Step 5: 시간축 EMA 스무딩 (Temporal Smoothing)**:
   인페인팅 결과 패치를 이전 프레임의 결과와 지수이동평균(EMA)으로 블렌딩하여 프레임마다 결과가 튀는 현상을 잡습니다.
6. **Step 6: 가우시안 알파 페더링 합성**:
   복원된 패치의 외곽선에 부드러운 가우시안 블러 마스크(Feathering)를 적용하여 원본 영상의 마룻바닥, 벽면 등과 이음새 없이 합성합니다.
7. **Step 7: FFmpeg 무손실 오디오 머지**:
   영상 렌더링이 끝나면 FFmpeg 서브프로세스를 호출하여 원본 오디오 스트림을 손실 없이 패스스루 결합하고 web-ready MP4를 완성합니다.

---

## 🧠 핵심 라이브러리 및 동작 원리 (Libraries & Principles)

### 1. LaMa AI (Large Mask Inpainting via Fast Fourier Convolutions)
- **사용 라이브러리**: `torch`, `torchvision`, `simple-lama-inpainting` (Big-LaMa 모델)
- **핵심 기술**: **Fast Fourier Convolution (FFC, 고속 푸리에 합성곱)**

#### 💡 기존 CNN 기반 인페인팅의 한계
일반적인 합성곱 신경망(CNN)은 국소 수용 영역(Local Receptive Field)에 갇혀 있어, 결손 영역이 조금만 커지거나 주변에 반복 패턴(격자 타일, 나뭇결, 텍스처)이 있을 경우 멀리 있는 구조적 맥락을 파악하지 못해 뭉개짐이나 왜곡이 발생합니다.

#### 💡 FFC의 수학적 원리
LaMa는 피처 맵을 **공간 분기(Spatial Branch)**와 **주파수 분기(Spectral Branch)**의 두 갈래로 나누어 처리합니다:
1. 공간 분기: 일반 Conv 연산을 통해 국소 세부 디테일(Local Detail) 보존
2. 주파수 분기: 2차원 이산 푸리에 변환(2D Real FFT)을 수행하여 공간 도메인을 주파수 도메인으로 변환:
   $$\hat{X}(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} X(x, y) e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$
3. 주파수 도메인에서는 한 픽셀의 변화가 전체 이미지의 주기적 패턴 정보를 담게 되므로, **수용 영역(Receptive Field)이 즉시 이미지 전체(Global Image Coverage)로 확장**됩니다.
4. 이후 복소수 합성곱(Complex Convolution)을 거친 뒤 역 푸리에 변환(Inverse FFT)을 통해 공간 도메인으로 되돌려 결손 영역을 사실적인 질감으로 채워 넣습니다.

#### 💡 하드웨어 가속 최적화
- **macOS Metal(MPS)**: PyTorch의 `mps` 백엔드를 자동 탐지하여 Apple Silicon 통합 메모리(Unified Memory)와 GPU 코어를 풀 활용합니다.
- **메모리 절약 지연 로딩(Lazy Loading)**: 인페인터 인스턴스를 싱글톤 형태로 유지하며, 최초 추론 시에만 가중치를 VRAM에 적재하여 리소스를 절약합니다.

---

### 2. OpenCV 영상 처리 및 정밀 형태 마스킹 (Shape Masking)
- **사용 라이브러리**: `opencv-python` (cv2), `numpy`

단순한 네모 상자로 마스킹하면 멀쩡한 배경(꽃잎, 잔디, 마룻바닥 등)까지 AI가 다시 그리면서 일렁거림이 심해집니다. 이를 방지하기 위해 정밀 형태 추출 알고리즘을 사용합니다:

1. **가우시안 블러링 & 통계적 임계처리 (Statistical Thresholding)**:
   워터마크 영역의 평균 밝기($\mu$)와 표준편차($\sigma$)를 계산하여, 반투명 흰색 워터마크 픽셀을 적응적으로 분리:
   $$\text{Threshold} = \min(\max(\mu + 0.6\sigma, 120), 220)$$
2. **Canny 엣지 검출 및 모폴로지 결합**:
   `cv2.Canny`로 스파클의 외곽 윤곽선을 따고, 타원형 커널(`MORPH_ELLIPSE`)로 모폴로지 닫힘(`MORPH_CLOSE`) 연산을 수행하여 끊어진 선을 이어줍니다.
3. **중심 거리 가중치 및 외곽선 필터링 (Contour Analysis)**:
   검출된 윤곽선 중 박스 중심점 $(c_x, c_y)$과의 유클리드 거리가 가장 가깝고 적절한 면적을 가진 컨투어를 찾아 볼록 껍질(`cv2.convexHull`)을 생성합니다.
4. **기하학적 4점 스파클 템플릿 안전망 (Fallback Polyfill)**:
   배경이 극도로 밝거나 복잡하여 윤곽선 검출에 실패하더라도, Gemini 고유의 4점 마름모 다이아몬드 기하 형상(`cv2.fillConvexPoly`)을 생성하여 워터마크가 삐져나가는 일이 없도록 보장합니다.
5. **모폴로지 팽창 (Dilation)**:
   반투명 안티앨리어싱 경계선의 잔상을 남기지 않도록 타원 커널을 사용하여 외곽으로 3~4px 안전하게 마스크를 확장합니다.

---

### 3. 시간축 스무딩 알고리즘 (Temporal Exponential Moving Average)
- **사용 라이브러리**: `opencv-python`, `numpy`

#### 💡 동영상 인페인팅의 깜빡임(Boiling/Flickering) 문제
프레임 단위(Frame-by-frame)로 정지 이미지 AI 모델을 돌릴 때 발생하는 가장 큰 문제는, 매 프레임마다 미세한 픽셀 오차나 노이즈가 발생하여 동영상 재생 시 해당 영역이 끓어오르듯 일렁거리는 현상입니다.

#### 💡 해결책: 시간축 지수이동평균(EMA) 필터
현재 프레임의 복원 결과 $I_t$와 직전 프레임의 복원 결과 $I_{t-1}^{blended}$를 가중 선형 결합합니다:
$$I_t^{blended} = \alpha \cdot I_t + (1 - \alpha) \cdot I_{t-1}^{blended}$$
- 기본 계수: $\alpha = 0.85$
- 시간적 연속성(Temporal Consistency)을 부여하여 정지된 배경이나 천천히 움직이는 장면에서 일렁거림을 완벽히 억제합니다.

---

### 4. 알파 페더링 블렌딩 (Gaussian Feathering)
- **사용 라이브러리**: `cv2.GaussianBlur`, `numpy`

복원된 패치를 원본 영상에 붙여 넣을 때 경계선이 칼로 자른 듯 도드라지는 현상을 방지하기 위해 알파 페더링을 적용합니다:
1. 바이너리 마스크(0 또는 255)를 $0.0 \sim 1.0$ 범위의 float32 행렬로 변환합니다.
2. 커널 반경(3~4px)의 가우시안 블러를 적용하여 부드러운 경계 알파 맵 $M_\alpha$를 생성합니다.
3. 3채널 선형 보간 합성:
   $$F_{final} = F_{inpainted} \times M_\alpha + F_{original} \times (1.0 - M_\alpha)$$
이 과정을 통해 원본 배경과 복원 영역의 경계선이 육안으로 식별 불가능할 정도로 매끄럽게 융합됩니다.

---

### 5. FFmpeg 무손실 오디오 패스스루 & 최적화 인코딩
- **사용 라이브러리**: `ffmpeg` (CLI 서브프로세스)

OpenCV의 `VideoWriter`는 순수 영상 프레임만 기록할 수 있으며 오디오 트랙을 다루지 못합니다. 이 프로젝트는 시스템의 FFmpeg를 활용하여 다음과 같은 고품질 인코딩을 수행합니다:

```bash
ffmpeg -y \
  -i temp_noaudio.mp4 \
  -i original.mp4 \
  -map 0:v:0 \
  -map 1:a:0? \
  -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p \
  -c:a copy \
  -movflags +faststart \
  output.mp4
```
- `-map 1:a:0?`: 원본 영상에 오디오가 있을 때만 가져오며, 무음 영상일 경우 에러 없이 스킵합니다.
- `-c:a copy`: 오디오를 재인코딩(Re-encoding)하지 않고 원본 비트스트림 그대로 패스스루하여 음질 손실이 0%입니다.
- `-crf 18`: 사람이 시각적으로 구별하기 어려운 최고 수준의 H.264 화질을 유지합니다.
- `-movflags +faststart`: 영상 메타데이터(moov atom)를 파일 맨 앞으로 이동시켜 웹 브라우저에서 버퍼링 없이 즉시 재생되도록 최적화합니다.

---

### 6. FastAPI & HTML5 Canvas 웹 아키텍처
- **사용 라이브러리**: `FastAPI`, `Uvicorn`, `HTML5 Canvas API`

- **좌표 역변환(Canvas Coordinate Inversion)**:
  9:16 세로 숏폼 영상이 모니터 화면에 넘치지 않도록 브라우저 창 높이(`window.innerHeight`)에 맞춰 캔버스 뷰포트를 자동 스케일링($S$)합니다. 사용자가 마우스로 클릭하거나 키보드로 이동한 화면 좌표 $(x', y')$를 영상의 원본 해상도 좌표 $(x, y)$로 정확하게 변환합니다:
  $$x = \left\lfloor \frac{x'}{S} \right\rfloor, \quad y = \left\lfloor \frac{y'}{S} \right\rfloor$$
- **키보드 이벤트 캡처 & 디바운스(Debounce)**:
  키보드 방향키 이벤트(`ArrowLeft`, `ArrowRight`, `ArrowUp`, `ArrowDown`)를 감지하여 1px씩 이동시키며, 이동 중에는 화면 재렌더링만 즉각 수행하고 서버 마스크 미리보기 API 요청은 120ms 디바운스로 지연시켜 서버 부하를 방지합니다.
- **백그라운드 스레드 렌더링**:
  동영상 렌더링은 긴 시간이 소요될 수 있으므로, FastAPI 서버 내에서 `threading.Thread` 백그라운드 태스크로 처리되며, 프론트엔드는 `/api/progress/{task_id}`를 500ms마다 폴링하여 정밀한 진행률(%)을 화면 중앙 모달에 표시합니다.

---

## 📂 프로젝트 디렉토리 구조

```plaintext
watermark-remover/
├── README.md               # 프로젝트 매뉴얼 및 기술 문서
├── requirements.txt        # 파이썬 의존 패키지 목록
├── remove_watermark.py     # 코어 인페인팅 엔진 및 CLI 처리 스크립트
├── web_app.py              # FastAPI 웹 애플리케이션 백엔드 서버
├── static/
│   └── index.html          # HTML5 Canvas 인터랙티브 웹 프론트엔드
└── web_temp/               # 임시 세션 파일 및 비디오 처리 작업 디렉토리
```

---

## ⚖️ 라이선스 및 참고
- 본 프로젝트는 MIT 라이선스를 따릅니다.
- LaMa 인페인팅 기술은 [advimman/lama](https://github.com/advimman/lama) 및 [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) 오픈소스 프로젝트를 기반으로 합니다.
- 생성형 AI 영상의 워터마크 제거 기능은 개인 연구 및 비상업적 학습 목적으로 책임감 있게 사용해 주시기 바랍니다.
