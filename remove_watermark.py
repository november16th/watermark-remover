#!/opt/anaconda3/bin/python3
"""
고품질 영상 워터마크 제거 도구
================================
Gemini AI 등이 생성한 영상의 워터마크를 AI 인페인팅으로 자연스럽게 제거합니다.

지원 엔진:
  - lama  : LaMa AI 인페인팅 (기본, 고품질)
  - opencv: OpenCV 인페인팅 (빠름, 보통 품질)

사용법:
  python remove_watermark.py input.mp4
  python remove_watermark.py input.mp4 -o output.mp4
  python remove_watermark.py input.mp4 --method opencv
  python remove_watermark.py input.mp4 --select
  python remove_watermark.py input.mp4 --region 1860,20,50,50
  python remove_watermark.py *.mp4
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────
# 마스크 생성
# ──────────────────────────────────────────────

def create_mask_for_roi(roi_shape, watermark_local_box, dilation_iterations=2):
    """
    ROI 내부에서 워터마크 영역에 대한 바이너리 마스크를 생성합니다.
    반투명 가장자리 잔상 방지를 위해 마스크를 dilate합니다.

    Args:
        roi_shape: (H, W) ROI 크기
        watermark_local_box: (x, y, w, h) ROI 내부에서의 워터마크 좌표
        dilation_iterations: 마스크 확장 반복 횟수
    Returns:
        numpy array (H, W) uint8 바이너리 마스크
    """
    rh, rw = roi_shape[:2]
    mask = np.zeros((rh, rw), dtype=np.uint8)
    lx, ly, lw, lh = watermark_local_box
    cv2.rectangle(mask, (lx, ly), (lx + lw, ly + lh), 255, -1)
    # 타원형 커널로 확장 — 사각형 아티팩트 방지
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    return mask


def create_precise_shape_mask(roi_img, watermark_local_box, dilate_px=4, mask_mode="auto"):
    """
    지정된 워터마크 영역 내에서 스파클(다이아몬드) 아이콘 형태를 정밀하게 분리합니다.
    돌담, 꽃잎 등 복잡하거나 명암비가 낮은 배경에서도 다이아몬드 윤곽을 안정적으로 검출합니다.

    Args:
        roi_img: BGR ROI 이미지
        watermark_local_box: (x, y, w, h) ROI 내부에서의 워터마크 영역
        dilate_px: 스파클 외곽 테두리 확장 픽셀 (기본: 4px)
        mask_mode: 'auto' (정밀 스파클 추출), 'box' (사각형 영역 전체)
    Returns:
        numpy array (H, W) uint8 바이너리 마스크 (255: 복원 대상, 0: 배경 보존)
    """
    rh, rw = roi_img.shape[:2]
    lx, ly, lw, lh = watermark_local_box

    if mask_mode == "box":
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.rectangle(mask, (lx, ly), (lx + lw, ly + lh), 255, -1)
        return mask

    crop = roi_img[ly:ly+lh, lx:lx+lw]
    if crop.size == 0 or lw < 10 or lh < 10:
        return create_mask_for_roi((rh, rw), watermark_local_box)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask_crop = np.zeros((lh, lw), dtype=np.uint8)

    # 1. 다중 임계값 탐색 및 그라디언트 엣지 조합
    # Gemini 워터마크는 4점 별 모양/다이아몬드이며 반투명함
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 상위 밝기 픽셀 기반 1차 추출
    mean_val = np.mean(blurred)
    std_val = np.std(blurred)
    # 배경 평균 대비 밝은 영역 (최소 120 이상)
    thresh_val = min(max(int(mean_val + std_val * 0.6), 120), 220)
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    # Canny 엣지와 합성하여 윤곽선 강화
    edges = cv2.Canny(blurred, 30, 100)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    combined = cv2.bitwise_or(binary, edges_closed)

    # 중심부 가중치 마스크 (테두리 밖의 잡음 꽃잎 배제)
    h_c, w_c = lh, lw
    cx, cy = w_c / 2, h_c / 2
    y_coords, x_coords = np.ogrid[:h_c, :w_c]
    dist_from_center = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    radius_max = min(w_c, h_c) * 0.48
    center_roi_mask = (dist_from_center <= radius_max).astype(np.uint8) * 255
    combined_center = cv2.bitwise_and(combined, center_roi_mask)

    contours, _ = cv2.findContours(combined_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    if contours:
        # 중심에 가장 가깝고 적절한 크기(전체 영역의 5%~60%)를 가진 컨투어 탐색
        box_area = lw * lh
        def score_contour(cnt):
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area < 30 or area > box_area * 0.75:
                return 999999
            dist_sq = (bx + bw/2 - cx)**2 + (by + bh/2 - cy)**2
            return dist_sq

        valid = [c for c in contours if score_contour(c) < 999999]
        if valid:
            target_cnt = min(valid, key=score_contour)
            hull = cv2.convexHull(target_cnt)
            cv2.drawContours(mask_crop, [hull], -1, 255, -1)
            detected = True

    # 2. 만약 배경 명암비가 복잡하여 스파클 형태가 잘 안 잡힌 경우:
    # ✦ 완벽한 기하학적 다이아몬드/스파클 템플릿 마스크를 생성하여 합성
    if not detected or cv2.countNonZero(mask_crop) < (lw * lh * 0.08):
        # 다이아몬드 스파클 4점 다각형 그리기 (안전망)
        pts = np.array([
            [int(cx), int(cy - h_c * 0.42)],  # 상
            [int(cx + w_c * 0.42), int(cy)],  # 우
            [int(cx), int(cy + h_c * 0.42)],  # 하
            [int(cx - w_c * 0.42), int(cy)]   # 좌
        ], dtype=np.int32)
        cv2.fillConvexPoly(mask_crop, pts, 255)

    # 테두리 확장(Dilation) 적용
    if dilate_px > 0:
        ksize = dilate_px * 2 + 1
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask_crop = cv2.dilate(mask_crop, kernel_dilate, iterations=1)

    full_mask = np.zeros((rh, rw), dtype=np.uint8)
    full_mask[ly:ly+lh, lx:lx+lw] = mask_crop
    return full_mask



def create_feathered_blend_mask(inpaint_mask, feather_radius=4):
    """
    실제 인페인팅 마스크(스파클 형태)를 기반으로 외곽만 살짝 블러링된 블렌딩 마스크를 생성합니다.
    사각형 전체를 블렌딩하지 않고 오직 복원된 픽셀 주변만 합성하므로 배경 일렁거림을 원천 차단합니다.

    Args:
        inpaint_mask: uint8 바이너리 마스크 (0 or 255)
        feather_radius: 가장자리 블러 반경 (기본: 4px)
    Returns:
        numpy array (H, W) float32 [0, 1] 블렌딩 마스크
    """
    mask_f = (inpaint_mask > 0).astype(np.float32)
    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)
    return np.clip(mask_f, 0.0, 1.0)


# ──────────────────────────────────────────────
# 마우스 선택 UI
# ──────────────────────────────────────────────

def select_region_with_mouse(frame):
    """
    첫 프레임을 보여주고 마우스 드래그로 워터마크 영역을 선택합니다.

    Args:
        frame: BGR 프레임
    Returns:
        (x, y, w, h) 선택된 영역 또는 선택 취소 시 None
    """
    window_name = "Drag to select watermark region, then press ENTER/SPACE"
    # 화면에 맞게 축소
    h, w = frame.shape[:2]
    scale = 1.0
    if w > 1280:
        scale = 1280.0 / w
    display = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()

    roi = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    if roi[2] == 0 or roi[3] == 0:
        return None

    # 원본 해상도로 좌표 변환
    x, y, rw, rh = roi
    x = int(x / scale)
    y = int(y / scale)
    rw = int(rw / scale)
    rh = int(rh / scale)
    return (x, y, rw, rh)


# ──────────────────────────────────────────────
# 자동 워터마크 위치 감지
# ──────────────────────────────────────────────

def detect_watermark_region(frame, corner="bottom-right", scan_ratio=0.4):
    """
    영상 하단 영역에서 Gemini ✦ 스파클 워터마크를 자동 감지합니다.
    하단의 넓은 영역을 스캔하여 작은 밝은 아이콘을 찾습니다.

    Args:
        frame: BGR 프레임
        corner: 검색 방향 ("bottom-right", "top-right", 등)
        scan_ratio: 프레임 대비 스캔 영역 비율 (기본 40%)
    Returns:
        (x, y, w, h) 감지된 워터마크 영역
    """
    h, w = frame.shape[:2]

    # 스캔 영역: 프레임의 하단/상단 40%, 우측/좌측 50%를 넓게 스캔
    scan_h = int(h * scan_ratio)
    scan_w = int(w * 0.5)

    if corner == "bottom-right":
        cx, cy = w - scan_w, h - scan_h
    elif corner == "top-right":
        cx, cy = w - scan_w, 0
    elif corner == "bottom-left":
        cx, cy = 0, h - scan_h
    elif corner == "top-left":
        cx, cy = 0, 0
    else:
        cx, cy = w - scan_w, h - scan_h

    cx = max(0, cx)
    cy = max(0, cy)
    region = frame[cy:cy + scan_h, cx:cx + scan_w]

    # 밝은 영역 감지 (반투명 흰색 워터마크)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # 모폴로지 연산으로 노이즈 제거 및 스파클 형태 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=3)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel, iterations=1)

    # 컨투어 찾기
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 워터마크 크기 범위에 맞는 컨투어 필터링 (10~100px)
        valid_contours = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            # 워터마크 크기 + 너무 납작하지 않은 것 (종횡비 체크)
            if 10 <= bw <= 100 and 10 <= bh <= 100 and area >= 80:
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if aspect < 3.0:  # 정사각형에 가까운 것
                    valid_contours.append(c)

        if valid_contours:
            # 하단-우측에 가장 가까운 것 우선 (Gemini 워터마크 위치)
            def corner_distance(c):
                bx, by, bw, bh = cv2.boundingRect(c)
                center_x = bx + bw / 2
                center_y = by + bh / 2
                if "bottom" in corner:
                    dy = scan_h - center_y
                else:
                    dy = center_y
                if "right" in corner:
                    dx = scan_w - center_x
                else:
                    dx = center_x
                return dx * dx + dy * dy

            best = min(valid_contours, key=corner_distance)
            bx, by, bw, bh = cv2.boundingRect(best)

            # 워터마크 중심 계산
            center_x = cx + bx + bw // 2
            center_y = cy + by + bh // 2

            # 이전 노란색 상자 크기(약 100~110px)만큼 넉넉하게 확장하여 워터마크 전체를 완벽히 포함
            # 워터마크가 짤리지 않도록 충분한 고정 크기(또는 배수)로 박스 설정
            target_size = max(int(max(bw, bh) * 2.5), 110)
            target_w = target_size
            target_h = target_size

            gx = center_x - target_w // 2
            gy = center_y - target_h // 2

            # 경계 클램프
            gx = max(0, min(w - target_w, gx))
            gy = max(0, min(h - target_h, gy))
            gw = min(w - gx, target_w)
            gh = min(h - gy, target_h)
            return (gx, gy, gw, gh)

    # 감지 실패 시 Gemini 워터마크 기본 위치 사용
    # Gemini ✦ 아이콘: 우측 하단 영역, 넉넉한 110x110 크기로 잘림 방지
    print("⚠️  워터마크 자동 감지 실패, 기본 영역을 사용합니다.")
    print("   💡 정확한 위치 지정은 --select 옵션을 사용하세요.")
    default_w = 110
    default_h = 110
    if corner == "bottom-right":
        return (w - int(w * 0.18), h - int(h * 0.12), default_w, default_h)
    elif corner == "top-right":
        return (w - int(w * 0.18), int(h * 0.03), default_w, default_h)
    elif corner == "top-left":
        return (int(w * 0.05), int(h * 0.03), default_w, default_h)
    else:
        return (int(w * 0.05), h - int(h * 0.12), default_w, default_h)


# ──────────────────────────────────────────────
# 미리보기 & 확인
# ──────────────────────────────────────────────

def preview_region(frame, watermark_box, margin=35, save_path=None):
    """
    감지된 워터마크 영역을 노란색 박스로 표시한 미리보기 이미지를 생성합니다.

    Args:
        frame: BGR 프레임
        watermark_box: (x, y, w, h) 워터마크 좌표
        margin: ROI 마진
        save_path: 미리보기 이미지 저장 경로 (None이면 자동 생성)
    Returns:
        저장된 미리보기 이미지 경로
    """
    preview = frame.copy()
    wx, wy, ww, wh = watermark_box
    h, w = frame.shape[:2]

    # 제거 대상 워터마크 영역 (노란색 실선)
    cv2.rectangle(preview, (wx, wy), (wx + ww, wy + wh), (0, 255, 255), 2)

    # 라벨 표시
    label = f"Watermark: ({wx},{wy}) {ww}x{wh}"
    cv2.putText(preview, label, (max(10, wx - 50), max(25, wy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if save_path is None:
        save_path = "watermark_preview.png"

    cv2.imwrite(save_path, preview)
    return save_path


def confirm_region_interactive(frame, watermark_box, margin=28):
    """
    워터마크 영역을 화면에 표시하고 사용자에게 확인을 받습니다.
    'y' 또는 Enter → 진행, 'n' → 취소, 's' → 마우스로 다시 선택

    Returns:
        확인된 watermark_box 또는 None (취소)
    """
    preview = frame.copy()
    wx, wy, ww, wh = watermark_box

    # 빨간 박스 표시
    cv2.rectangle(preview, (wx, wy), (wx + ww, wy + wh), (0, 0, 255), 2)
    label = f"({wx},{wy}) {ww}x{wh} - Press ENTER to confirm, 's' to reselect, ESC to cancel"
    cv2.putText(preview, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 화면에 맞게 축소
    h, w = preview.shape[:2]
    scale = 1.0
    if w > 1280:
        scale = 1280.0 / w
    display = cv2.resize(preview, None, fx=scale, fy=scale) if scale < 1.0 else preview

    window_name = "Watermark Region Preview"
    cv2.imshow(window_name, display)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13 or key == 10 or key == ord('y'):  # Enter or 'y'
            cv2.destroyWindow(window_name)
            return watermark_box
        elif key == 27 or key == ord('n'):  # ESC or 'n'
            cv2.destroyWindow(window_name)
            return None
        elif key == ord('s'):  # 's' for reselect
            cv2.destroyWindow(window_name)
            new_box = select_region_with_mouse(frame)
            return new_box


# ──────────────────────────────────────────────

class OpenCVInpainter:
    """OpenCV 기반 인페인팅 엔진 (빠름, CPU)"""

    def __init__(self, method="ns", radius=3):
        self.flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
        self.radius = radius

    def inpaint(self, bgr_roi, mask):
        """
        Args:
            bgr_roi: BGR ROI 이미지 (numpy uint8)
            mask: 바이너리 마스크 (numpy uint8, 255=인페인팅 대상)
        Returns:
            인페인팅된 BGR ROI
        """
        return cv2.inpaint(bgr_roi, mask, self.radius, self.flag)


class LamaInpainter:
    """LaMa AI 인페인팅 엔진 (고품질, MPS/CPU)"""

    def __init__(self):
        self.model = None
        self.device = None

    def _lazy_init(self):
        """최초 호출 시 모델 로드 (지연 초기화)"""
        if self.model is not None:
            return

        try:
            import torch
            # simple_lama_inpainting의 유틸리티 함수만 사용
            from simple_lama_inpainting.utils import prepare_img_and_mask, download_model

            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("🚀 Apple Silicon GPU (MPS) 가속 사용")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("🚀 NVIDIA GPU (CUDA) 가속 사용")
            else:
                self.device = torch.device("cpu")
                print("💻 CPU 모드 사용")

            # 모델 다운로드 (이미 있으면 캐시 사용)
            import os
            lama_url = os.environ.get(
                "LAMA_MODEL_URL",
                "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
            )
            model_path = os.environ.get("LAMA_MODEL") or download_model(lama_url)

            # 핵심 수정: map_location='cpu'로 로드 후 디바이스 이동
            # (원본 SimpleLama는 map_location 없이 로드해서 macOS에서 CUDA 에러 발생)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.model = torch.jit.load(model_path, map_location="cpu")
            self.model.eval()
            self.model.to(self.device)

            # prepare_img_and_mask 함수 캐시
            self._prepare = prepare_img_and_mask

            print("✅ LaMa 모델 로드 완료")
        except ImportError:
            print("❌ simple-lama-inpainting이 설치되지 않았습니다.")
            print("   설치: pip install simple-lama-inpainting")
            sys.exit(1)

    def inpaint(self, bgr_roi, mask):
        """
        Args:
            bgr_roi: BGR ROI 이미지 (numpy uint8)
            mask: 바이너리 마스크 (numpy uint8, 255=인페인팅 대상)
        Returns:
            인페인팅된 BGR ROI
        """
        import torch

        self._lazy_init()

        # OpenCV BGR → PIL RGB
        rgb_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_roi)
        pil_mask = Image.fromarray(mask).convert("L")

        # 원본 크기 기억 (LaMa는 내부적으로 8의 배수로 패딩함)
        orig_h, orig_w = bgr_roi.shape[:2]

        # 이미지와 마스크 전처리
        image_tensor, mask_tensor = self._prepare(pil_image, pil_mask, self.device)

        # LaMa 추론
        with torch.inference_mode():
            inpainted = self.model(image_tensor, mask_tensor)
            result_np = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)

        # 패딩된 출력을 원본 크기로 크롭
        result_np = result_np[:orig_h, :orig_w]

        # RGB → BGR
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        return result_bgr


# ──────────────────────────────────────────────
# 비디오 처리 파이프라인
# ──────────────────────────────────────────────

def process_video(
    input_path,
    output_path,
    watermark_box,
    method="lama",
    margin=28,
    ema_alpha=0.85,
    crf=18,
):
    """
    영상에서 워터마크를 제거합니다.

    Args:
        input_path: 입력 영상 경로
        output_path: 출력 영상 경로
        watermark_box: (x, y, w, h) 워터마크 좌표
        method: 인페인팅 방법 ("lama" 또는 "opencv")
        margin: ROI 마진 (px)
        ema_alpha: Temporal EMA 블렌딩 계수 (1.0이면 비활성)
        crf: x264 품질 (낮을수록 고품질, 기본 18)
    """
    # 인페인터 초기화
    if method == "lama":
        inpainter = LamaInpainter()
    else:
        inpainter = OpenCVInpainter(method="ns", radius=3)

    # 영상 열기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0:
        print(f"❌ 영상 프레임을 읽을 수 없습니다: {input_path}")
        cap.release()
        return False

    print(f"\n📹 입력: {input_path}")
    print(f"   해상도: {width}×{height}, FPS: {fps:.1f}, 프레임: {total_frames}")
    print(f"   길이: {total_frames / fps:.1f}초")
    print(f"   엔진: {'LaMa AI' if method == 'lama' else 'OpenCV'}")

    # ROI 계산 (마진 포함, 경계 클램프)
    wx, wy, ww, wh = watermark_box
    rx = max(0, wx - margin)
    ry = max(0, wy - margin)
    rw = min(width - rx, ww + 2 * margin)
    rh = min(height - ry, wh + 2 * margin)

    print(f"   워터마크: ({wx}, {wy}, {ww}×{wh})")
    print(f"   ROI 영역: ({rx}, {ry}, {rw}×{rh})")

    # 첫 프레임 읽어서 정밀 스파클 마스크 생성 (일렁거림 방지 핵심)
    ret_first, first_frame = cap.read()
    if not ret_first:
        print(f"❌ 첫 프레임을 읽을 수 없습니다: {input_path}")
        cap.release()
        return False
        
    first_roi = first_frame[ry:ry + rh, rx:rx + rw].copy()
    inpaint_mask = create_precise_shape_mask(first_roi, (local_wx, local_wy, ww, wh), dilate_px=4)
    blend_mask = create_feathered_blend_mask(inpaint_mask, feather_radius=3)
    blend_mask_3ch = np.stack([blend_mask] * 3, axis=-1)

    # 비디오 포인터 처음으로 되돌리기
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 임시 파일에 영상 저장 (오디오 없이)
    temp_dir = tempfile.mkdtemp()
    temp_video = os.path.join(temp_dir, "temp_noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    prev_inpainted_roi = None
    pbar = tqdm(total=total_frames, desc="🔧 처리 중", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. ROI 추출
        roi = frame[ry:ry + rh, rx:rx + rw].copy()

        # 2. 인페인팅
        inpainted_roi = inpainter.inpaint(roi, inpaint_mask)

        # 3. Temporal EMA 블렌딩 (깜빡임 방지)
        if prev_inpainted_roi is not None and ema_alpha < 1.0:
            inpainted_roi = cv2.addWeighted(
                inpainted_roi, ema_alpha,
                prev_inpainted_roi, 1.0 - ema_alpha,
                0,
            )
        prev_inpainted_roi = inpainted_roi.copy()

        # 4. Feathered 블렌딩으로 ROI를 원본에 자연스럽게 합성
        original_roi = frame[ry:ry + rh, rx:rx + rw].astype(np.float32)
        inpainted_float = inpainted_roi.astype(np.float32)
        blended = (inpainted_float * blend_mask_3ch +
                   original_roi * (1.0 - blend_mask_3ch))
        frame[ry:ry + rh, rx:rx + rw] = blended.astype(np.uint8)

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # 5. FFmpeg로 오디오 합성 + 고품질 재인코딩
    print("🔊 오디오 합성 중...")
    success = merge_audio_ffmpeg(temp_video, input_path, output_path, crf=crf)

    # 임시 파일 정리
    try:
        os.remove(temp_video)
        os.rmdir(temp_dir)
    except OSError:
        pass

    if success:
        # 출력 파일 크기
        out_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 완료: {output_path} ({out_size:.1f} MB)")
    return success


def merge_audio_ffmpeg(video_path, original_path, output_path, crf=18):
    """
    FFmpeg를 사용하여 인페인팅된 영상에 원본 오디오를 합성합니다.

    Args:
        video_path: 오디오 없는 인페인팅 영상
        original_path: 원본 영상 (오디오 소스)
        output_path: 최종 출력 경로
        crf: x264 CRF 값 (기본 18)
    Returns:
        성공 여부 (bool)
    """
    # ffmpeg 존재 확인
    ffmpeg_path = _find_ffmpeg()
    if not ffmpeg_path:
        print("⚠️  ffmpeg를 찾을 수 없습니다. 오디오 없이 저장합니다.")
        # ffmpeg 없으면 그냥 복사
        import shutil
        shutil.copy2(video_path, output_path)
        return True

    cmd = [
        ffmpeg_path, "-y",
        "-i", video_path,       # 인페인팅된 영상
        "-i", original_path,    # 원본 (오디오 소스)
        "-map", "0:v:0",        # 첫 번째 입력의 비디오
        "-map", "1:a:0?",       # 두 번째 입력의 오디오 (없으면 무시)
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-pix_fmt", "yuv420p",  # 호환성
        "-c:a", "copy",         # 오디오 무손실 복사
        "-movflags", "+faststart",  # 웹 스트리밍 최적화
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"⚠️  ffmpeg 오류: {result.stderr[:200]}")
            # 실패 시 오디오 없이 저장
            import shutil
            shutil.copy2(video_path, output_path)
        return True
    except FileNotFoundError:
        print("⚠️  ffmpeg를 실행할 수 없습니다. 오디오 없이 저장합니다.")
        import shutil
        shutil.copy2(video_path, output_path)
        return True
    except subprocess.TimeoutExpired:
        print("⚠️  ffmpeg 시간 초과")
        return False


def _find_ffmpeg():
    """ffmpeg 실행 파일 경로를 찾습니다."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    # 일반적인 macOS 경로
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.isfile(candidate):
            return candidate
    return None


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_region(region_str):
    """'x,y,w,h' 문자열을 튜플로 파싱합니다."""
    parts = region_str.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"영역 형식이 잘못되었습니다: '{region_str}' (올바른 형식: x,y,w,h)"
        )
    try:
        return tuple(int(p.strip()) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"영역 좌표는 정수여야 합니다: '{region_str}'"
        )


def generate_output_path(input_path, suffix="_clean"):
    """입력 파일명에 suffix를 붙인 출력 경로를 생성합니다."""
    p = Path(input_path)
    return str(p.parent / f"{p.stem}{suffix}{p.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="🎬 고품질 영상 워터마크 제거 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  %(prog)s input.mp4                          # 기본 (자동 감지 → 바로 처리)
  %(prog)s input.mp4 --preview                # 미리보기만 저장 (처리 안 함)
  %(prog)s input.mp4 --select                 # 마우스로 영역 선택
  %(prog)s input.mp4 -o output.mp4            # 출력 파일 지정
  %(prog)s input.mp4 --region 1860,20,50,50   # 좌표 직접 지정
  %(prog)s input.mp4 --method opencv          # OpenCV 모드 (빠름)
  %(prog)s input.mp4 --confirm                # 감지 결과 확인 후 처리
  %(prog)s *.mp4                              # 여러 파일 일괄 처리
        """,
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help="입력 영상 파일 경로 (여러 개 가능, 와일드카드 지원)",
    )
    parser.add_argument(
        "-o", "--output",
        help="출력 파일 경로 (단일 파일일 때만 사용, 기본: input_clean.mp4)",
    )
    parser.add_argument(
        "--method",
        choices=["lama", "opencv"],
        default="lama",
        help="인페인팅 방법 (기본: lama)",
    )
    parser.add_argument(
        "--region",
        type=parse_region,
        help="워터마크 영역 좌표 x,y,w,h (예: 1860,20,50,50)",
    )
    parser.add_argument(
        "--corner",
        choices=["top-right", "top-left", "bottom-right", "bottom-left"],
        default="bottom-right",
        help="워터마크 위치 모서리 (자동 감지용, 기본: bottom-right)",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="마우스로 워터마크 영역을 선택합니다",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=28,
        help="ROI 마진 (px, 기본: 28)",
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=0.85,
        help="Temporal EMA 계수 (0-1, 기본: 0.85, 1.0이면 비활성)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="출력 영상 품질 CRF (0-51, 낮을수록 고품질, 기본: 18)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="워터마크 영역 미리보기만 표시 (처리하지 않음)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="자동 감지 후 확인을 받고 처리 (기본: 확인 없이 바로 처리)",
    )

    args = parser.parse_args()

    # 입력 파일 확장 (glob 처리)
    input_files = []
    for pattern in args.inputs:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        else:
            input_files.append(pattern)

    # 유효한 파일만 필터링
    valid_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    valid_files = []
    for f in input_files:
        if not os.path.isfile(f):
            print(f"⚠️  파일을 찾을 수 없습니다: {f}")
            continue
        if Path(f).suffix.lower() not in valid_extensions:
            print(f"⚠️  지원하지 않는 형식입니다: {f}")
            continue
        valid_files.append(f)

    if not valid_files:
        print("❌ 처리할 영상 파일이 없습니다.")
        sys.exit(1)

    # 출력 경로 처리
    if args.output and len(valid_files) > 1:
        print("⚠️  여러 파일을 처리할 때는 -o 옵션을 사용할 수 없습니다.")
        print("    각 파일에 '_clean' 접미사가 자동으로 붙습니다.")
        args.output = None

    print(f"\n{'='*50}")
    print(f"🎬 영상 워터마크 제거 도구")
    print(f"{'='*50}")
    print(f"  엔진: {'LaMa AI (고품질)' if args.method == 'lama' else 'OpenCV (빠름)'}")
    print(f"  파일: {len(valid_files)}개")
    print(f"  CRF: {args.crf}")
    print(f"{'='*50}")

    # 워터마크 영역 결정
    watermark_box = None
    if args.region:
        watermark_box = args.region
        print(f"\n📌 지정된 워터마크 영역: {watermark_box}")
    elif args.select:
        # 첫 번째 파일의 첫 프레임으로 선택
        cap = cv2.VideoCapture(valid_files[0])
        ret, frame = cap.read()
        cap.release()
        if ret:
            watermark_box = select_region_with_mouse(frame)
            if watermark_box is None:
                print("❌ 영역 선택이 취소되었습니다.")
                sys.exit(1)
            print(f"\n📌 선택된 워터마크 영역: {watermark_box}")
        else:
            print("❌ 첫 프레임을 읽을 수 없습니다.")
            sys.exit(1)

    # 각 파일 처리
    success_count = 0
    for i, input_path in enumerate(valid_files):
        if len(valid_files) > 1:
            print(f"\n{'─'*40}")
            print(f"  [{i+1}/{len(valid_files)}] {os.path.basename(input_path)}")
            print(f"{'─'*40}")

        # 출력 경로
        if args.output and len(valid_files) == 1:
            output_path = args.output
        else:
            output_path = generate_output_path(input_path)

        # 자동 감지 (region/select 미지정 시)
        current_box = watermark_box
        if current_box is None:
            cap = cv2.VideoCapture(input_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                current_box = detect_watermark_region(frame, corner=args.corner)
                print(f"🔍 자동 감지된 워터마크 영역: {current_box}")

                # 미리보기 이미지 저장
                preview_path = generate_output_path(input_path, suffix="_preview")
                preview_path = str(Path(preview_path).with_suffix(".png"))
                preview_region(frame, current_box, margin=args.margin, save_path=preview_path)
                print(f"📸 미리보기 저장됨: {preview_path}")

                if args.preview:
                    # --preview 모드: 미리보기만 보고 종료
                    print("   (--preview 모드: 처리하지 않고 미리보기만 저장합니다)")
                    continue

                if args.confirm:
                    # 사용자에게 확인 받기
                    print("\n   👀 미리보기 이미지를 확인하세요.")
                    print("   노란 박스 = 제거할 워터마크 영역")
                    response = input("   ✅ 이대로 진행할까요? (y/Enter=진행, s=마우스로 다시 선택, n=취소): ").strip().lower()
                    if response == 's':
                        current_box = select_region_with_mouse(frame)
                        if current_box is None:
                            print("❌ 영역 선택이 취소되었습니다.")
                            continue
                        print(f"📌 새로 선택된 워터마크 영역: {current_box}")
                    elif response == 'n':
                        print("⏭️  건너뜁니다.")
                        continue
                    # 'y', '', Enter → 그대로 진행
            else:
                print(f"❌ 프레임을 읽을 수 없습니다: {input_path}")
                continue

        # 처리
        ok = process_video(
            input_path=input_path,
            output_path=output_path,
            watermark_box=current_box,
            method=args.method,
            margin=args.margin,
            ema_alpha=args.ema,
            crf=args.crf,
        )
        if ok:
            success_count += 1

    # 요약
    print(f"\n{'='*50}")
    print(f"📊 완료: {success_count}/{len(valid_files)}개 파일 처리됨")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
