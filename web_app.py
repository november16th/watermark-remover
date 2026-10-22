#!/usr/bin/env python3
"""
FastAPI 기반 고품질 AI 영상 워터마크 정밀 제거 웹 서버
"""

import os
import sys

# macOS OpenMP 중복 로드 충돌 방지 (OMP: Error #15 크래시 원천 차단)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# rembg (AI 배경 제거) — 세션을 한 번만 생성하여 모델 재로드 방지
from rembg import remove as rembg_remove, new_session as rembg_new_session

# 인페인터 및 유틸리티 가져오기
from remove_watermark import (
    LamaInpainter,
    create_feathered_blend_mask,
    create_precise_shape_mask,
    detect_watermark_region,
    merge_audio_ffmpeg,
)

app = FastAPI(title="Video Edit Editor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMP_DIR = BASE_DIR / "web_temp"
TEMP_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 상태 저장소
sessions = {}
tasks = {}

# 단일 Lama Inpainter 인스턴스 (메모리 절약)
inpainter = LamaInpainter()

# rembg 모델 세션 (지연 로딩: 사용 시점에 로드하여 서버 즉각 기동)
_rembg_sessions = {}


def get_rembg_session(model_name: str = "fast"):
    actual_model = "u2netp" if model_name == "fast" else "isnet-general-use"
    if actual_model not in _rembg_sessions:
        print(f"📥 rembg AI 모델({actual_model}) 로드 중...")
        _rembg_sessions[actual_model] = rembg_new_session(actual_model)
    return _rembg_sessions[actual_model]


class BoxModel(BaseModel):
    x: int
    y: int
    w: int
    h: int


class ProcessRequest(BaseModel):
    box: BoxModel
    dilate: int = 4
    mode: str = "auto"  # 'auto' (다이아몬드 정밀), 'diamond' (강제 다이아몬드), 'box' (전체 사각형)


class FrameRequest(BaseModel):
    time_sec: float = 0.0


@app.get("/")
def read_root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/extract-frame")
async def extract_frame(video: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    sess_path = TEMP_DIR / session_id
    sess_path.mkdir(exist_ok=True)

    input_path = sess_path / "input.mp4"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JSONResponse({"error": "영상의 첫 프레임을 읽지 못했습니다."}, status_code=400)

    frame_path = sess_path / "frame_first.png"
    cv2.imwrite(str(frame_path), frame)

    # 자동 감지 추천 박스 계산 (우측 하단 기본)
    suggested = detect_watermark_region(frame, corner="bottom-right")
    sx, sy, sw, sh = suggested

    sessions[session_id] = {
        "input_video": str(input_path),
        "frame_path": str(frame_path),
        "frame_bgr": frame,
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
    }

    return {
        "session_id": session_id,
        "frame_url": f"/api/files/{session_id}/frame_first.png",
        "suggested_box": {"x": sx, "y": sy, "w": sw, "h": sh},
        "duration": round(duration, 2),
        "fps": round(fps, 2),
        "total_frames": total_frames,
    }


@app.post("/api/extract-frame-at")
async def extract_frame_at(req: FrameRequest):
    """비디오의 특정 시간(초) 프레임을 추출하여 화면 갱신"""
    if not sessions:
        return JSONResponse({"error": "세션을 찾을 수 없습니다."}, status_code=400)

    session_id = list(sessions.keys())[-1]
    sess = sessions[session_id]
    input_path = sess["input_video"]

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame = max(0, int(req.time_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JSONResponse({"error": "해당 시간의 프레임을 추출할 수 없습니다."}, status_code=400)

    sess_path = TEMP_DIR / session_id
    frame_path = sess_path / "frame_current.png"
    cv2.imwrite(str(frame_path), frame)

    sess["frame_bgr"] = frame
    sess["frame_path"] = str(frame_path)

    # 이 시점의 프레임에서 추천 박스도 재계산
    suggested = detect_watermark_region(frame, corner="bottom-right")
    sx, sy, sw, sh = suggested

    return {
        "frame_url": f"/api/files/{session_id}/frame_current.png",
        "suggested_box": {"x": sx, "y": sy, "w": sw, "h": sh},
    }


@app.post("/api/preview-mask")
async def preview_mask(req: ProcessRequest):
    # 가장 최근 세션 사용
    if not sessions:
        return JSONResponse({"error": "세션을 찾을 수 없습니다."}, status_code=400)

    session_id = list(sessions.keys())[-1]
    sess = sessions[session_id]
    frame = sess["frame_bgr"]
    h, w = frame.shape[:2]

    # ROI 추출 (마진 25px)
    margin = 25
    bx, by, bw, bh = req.box.x, req.box.y, req.box.w, req.box.h
    rx = max(0, bx - margin)
    ry = max(0, by - margin)
    rw = min(w - rx, bw + 2 * margin)
    rh = min(h - ry, bh + 2 * margin)

    roi = frame[ry:ry + rh, rx:rx + rw]
    local_bx = bx - rx
    local_by = by - ry

    # 정밀 스파클 또는 사각형 박스 마스크 생성
    mask = create_precise_shape_mask(
        roi, (local_bx, local_by, bw, bh), dilate_px=req.dilate, mask_mode=req.mode
    )

    # 마스크 시각화 (스파클 부분만 잘라서 확대해 반환)
    mask_crop = mask[local_by:local_by + bh, local_bx:local_bx + bw]
    # 투명 PNG 오버레이용 또는 흑백 마스크
    mask_preview_path = TEMP_DIR / session_id / "mask_preview.png"
    cv2.imwrite(str(mask_preview_path), mask_crop)

    return {
        "mask_url": f"/api/files/{session_id}/mask_preview.png"
    }


def run_processing_task(task_id, session_id, box, dilate_px, mode="auto"):
    print(f"\n==================================================")
    print(f"🎬 [TASK START] 워터마크 제거 작업 시작")
    print(f"   Task ID: {task_id}")
    print(f"   모드: {mode} (box=사각형박스, auto=스파클정밀)")
    print(f"   확장(dilate): {dilate_px}px")
    print(f"   지정 영역: (x={box.x}, y={box.y}, w={box.w}, h={box.h})")
    print(f"==================================================")
    try:
        sess = sessions[session_id]
        input_path = sess["input_video"]
        sess_dir = TEMP_DIR / session_id
        output_path = sess_dir / "cleaned.mp4"

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📹 [VIDEO INFO] 해상도: {width}x{height}, FPS: {fps:.2f}, 총 프레임: {total_frames}")

        # ROI 영역
        margin = 28
        bx, by, bw, bh = box.x, box.y, box.w, box.h
        rx = max(0, bx - margin)
        ry = max(0, by - margin)
        rw = min(width - rx, bw + 2 * margin)
        rh = min(height - ry, bh + 2 * margin)

        local_bx = bx - rx
        local_by = by - ry

        print(f"✂️ [ROI INFO] 워터마크: ({bx}, {by}, {bw}x{bh}) -> ROI: ({rx}, {ry}, {rw}x{rh})")

        # 기준 프레임 (사용자가 탐색한 현재 프레임이 있으면 그것을 우선 사용, 없으면 첫 프레임)
        base_frame = sess.get("frame_bgr")
        if base_frame is None:
            ret_first, base_frame = cap.read()
        base_roi = base_frame[ry:ry + rh, rx:rx + rw]

        print(f"🎭 [MASK] 마스크 생성 중 (모드: {mode})...")
        inpaint_mask = create_precise_shape_mask(
            base_roi, (local_bx, local_by, bw, bh), dilate_px=dilate_px, mask_mode=mode
        )
        # 실제 마스킹된 외곽 3px만 부드럽게 합성하여 주변 배경 왜곡을 최소화
        blend_mask = create_feathered_blend_mask(inpaint_mask, feather_radius=3)
        blend_mask_3ch = np.stack([blend_mask] * 3, axis=-1)
        print(f"✅ [MASK] 마스크 생성 완료 (크기: {inpaint_mask.shape})")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        temp_raw = sess_dir / "temp_raw.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(temp_raw), fourcc, fps, (width, height))

        prev_roi = None
        # 0.70으로 안정화 가중치를 높여 프레임 간 펄럭임/일렁거림 완화
        ema_alpha = 0.70
        idx = 0

        print(f"🤖 [INPAINT] AI 인페인팅 엔진 실행 시작...")
        tasks[task_id]["status"] = f"AI 복원 처리 중... (0/{total_frames})"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            roi = frame[ry:ry + rh, rx:rx + rw].copy()
            inpainted_roi = inpainter.inpaint(roi, inpaint_mask)

            # 시간축 스무딩 (시간적 연속성 보장)
            if prev_roi is not None:
                inpainted_roi = cv2.addWeighted(
                    inpainted_roi, ema_alpha, prev_roi, 1.0 - ema_alpha, 0
                )
            prev_roi = inpainted_roi.copy()

            # 정밀 합성: 마스크 자리만 복원치로 대치, 바깥 배경은 원본 100% 보존
            orig_roi_f = frame[ry:ry + rh, rx:rx + rw].astype(np.float32)
            inpaint_f = inpainted_roi.astype(np.float32)
            blended = inpaint_f * blend_mask_3ch + orig_roi_f * (1.0 - blend_mask_3ch)
            frame[ry:ry + rh, rx:rx + rw] = blended.astype(np.uint8)

            out.write(frame)
            idx += 1
            progress = idx / max(total_frames, 1)
            tasks[task_id]["progress"] = progress

            # 15프레임마다 또는 첫/마지막 프레임에 콘솔 및 상태창 로그 출력
            if idx == 1 or idx % 15 == 0 or idx == total_frames:
                percent = int(progress * 100)
                status_msg = f"AI 복원 처리 중... ({idx}/{total_frames} 프레임, {percent}%)"
                tasks[task_id]["status"] = status_msg
                print(f"⚡ [PROGRESS] {status_msg}")

        cap.release()
        out.release()

        print(f"🔊 [AUDIO] 오디오 합성 및 ffmpeg 최종 인코딩 시작...")
        tasks[task_id]["status"] = "오디오 합성 및 최종 인코딩 중..."
        merge_audio_ffmpeg(str(temp_raw), input_path, str(output_path), crf=18)

        if temp_raw.exists():
            temp_raw.unlink()

        tasks[task_id]["progress"] = 1.0
        tasks[task_id]["done"] = True
        tasks[task_id]["status"] = "완료!"
        tasks[task_id]["clean_url"] = f"/api/files/{session_id}/cleaned.mp4"
        tasks[task_id]["orig_url"] = f"/api/files/{session_id}/input.mp4"
        print(f"🎉 [SUCCESS] 워터마크 제거 완료! 결과 파일: {output_path}\n")

    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(f"\n❌ [ERROR] 처리 중 예외 발생:")
        print(err_detail)
        tasks[task_id]["status"] = f"오류 발생: {str(e)}"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["done"] = True


@app.post("/api/process")
async def process_video_endpoint(req: ProcessRequest):
    if not sessions:
        return JSONResponse({"error": "세션을 찾을 수 없습니다."}, status_code=400)

    session_id = list(sessions.keys())[-1]
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "progress": 0.0,
        "done": False,
        "status": "AI 복원 처리 중...",
    }

    t = threading.Thread(
        target=run_processing_task,
        args=(task_id, session_id, req.box, req.dilate, req.mode),
    )
    t.daemon = True
    t.start()

    return {"task_id": task_id}


@app.get("/api/progress/{task_id}")
@app.get("/api/tasks/{task_id}")
async def get_progress(task_id: str):
    if task_id not in tasks:
        return JSONResponse({"error": "작업을 찾을 수 없습니다."}, status_code=404)
    return tasks[task_id]


@app.get("/api/files/{session_id}/{filename}")
async def get_file(session_id: str, filename: str):
    file_path = TEMP_DIR / session_id / filename
    if not file_path.exists():
        return JSONResponse({"error": "파일을 찾을 수 없습니다."}, status_code=404)
    media_type = None
    if filename.endswith(".webm"):
        media_type = "video/webm"
    return FileResponse(file_path, media_type=media_type)


# ──────────────────────────────────────────────
# 배경 제거 (Background Removal) API
# ──────────────────────────────────────────────

class BgRemoveRequest(BaseModel):
    output_mode: str = "white"        # 'white' (순백색 MP4, 추천), 'transparent' (WebM alpha), 'green' (크로마키 MP4)
    quality: str = "fast"             # 'fast' (초고속 u2netp) 또는 'quality' (고정밀 isnet)


@app.post("/api/bg-upload")
async def bg_upload(video: UploadFile = File(...)):
    """배경 제거용 영상 업로드 — 첫 프레임 미리보기 반환"""
    session_id = "bg_" + str(uuid.uuid4())
    sess_path = TEMP_DIR / session_id
    sess_path.mkdir(exist_ok=True)

    input_path = sess_path / "input.mp4"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    cap = cv2.VideoCapture(str(input_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JSONResponse({"error": "영상의 첫 프레임을 읽지 못했습니다."}, status_code=400)

    frame_path = sess_path / "frame_first.png"
    cv2.imwrite(str(frame_path), frame)

    # rembg로 첫 프레임 배경 제거 미리보기 생성 (초고속 모델 사용)
    from PIL import Image
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pil_removed = rembg_remove(pil_frame, session=get_rembg_session("fast"))
    preview_path = sess_path / "preview_nobg.png"
    pil_removed.save(str(preview_path))

    sessions[session_id] = {
        "input_video": str(input_path),
        "frame_path": str(frame_path),
    }

    return {
        "session_id": session_id,
        "frame_url": f"/api/files/{session_id}/frame_first.png",
        "preview_url": f"/api/files/{session_id}/preview_nobg.png",
    }


def run_bg_remove_task(task_id, session_id, output_mode="white", quality="fast"):
    """
    프레임별 배경 제거 처리 (고속 메모리 파이프라인)
    - white: 순백색(RGB: 255, 255, 255) 배경의 웹 전용 고화질 MP4
    - green: 크로마키(초록) 배경의 편집용 MP4
    - transparent: 알파 채널이 포함된 웹용 투명 WebM
    """
    try:
        sess = sessions[session_id]
        input_path = sess["input_video"]
        sess_dir = TEMP_DIR / session_id

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        active_session = get_rembg_session(quality)
        from PIL import Image

        if output_mode in ("white", "green"):
            # 1. 단색 배경(순백색 또는 크로마키) 직접 VideoWriter 스트리밍 (디스크 쓰기 0회)
            bg_rgb = (255, 255, 255) if output_mode == "white" else (0, 177, 64)
            bg_rgba = (*bg_rgb, 255)

            temp_raw = sess_dir / "temp_raw_bg.mp4"
            output_path = sess_dir / "bg_removed.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(temp_raw), fourcc, fps, (width, height))

            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_removed = rembg_remove(pil_frame, session=active_session)  # RGBA

                # 배경색과 알파 합성
                bg = Image.new("RGBA", pil_removed.size, bg_rgba)
                composited = Image.alpha_composite(bg, pil_removed)
                out_frame = cv2.cvtColor(np.array(composited.convert("RGB")), cv2.COLOR_RGB2BGR)
                out.write(out_frame)

                idx += 1
                tasks[task_id]["progress"] = idx / max(total_frames, 1)

            cap.release()
            out.release()

            tasks[task_id]["status"] = "오디오 합성 및 최종 인코딩 중..."
            merge_audio_ffmpeg(str(temp_raw), input_path, str(output_path), crf=18)
            if temp_raw.exists():
                temp_raw.unlink()

            tasks[task_id]["result_url"] = f"/api/files/{session_id}/bg_removed.mp4"
            tasks[task_id]["result_filename"] = "bg_white.mp4" if output_mode == "white" else "bg_green.mp4"

        else:
            # 2. 투명 WebM (rawvideo 파이프 직접 스트리밍, 디스크 쓰기 0회)
            output_path = sess_dir / "bg_removed.webm"
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-i", input_path,
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuva420p",
                "-b:v", "2M",
                "-map", "0:v", "-map", "1:a?",
                "-c:a", "libopus",
                "-shortest",
                "-auto-alt-ref", "0",
                str(output_path)
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_removed = rembg_remove(pil_frame, session=active_session)  # RGBA

                # RGBA 바이트 스트림을 ffmpeg stdin으로 바로 전송
                proc.stdin.write(pil_removed.tobytes())

                idx += 1
                tasks[task_id]["progress"] = idx / max(total_frames, 1)

            cap.release()
            proc.stdin.close()
            tasks[task_id]["status"] = "WebM 알파 비디오 인코딩 중..."
            proc.wait()

            tasks[task_id]["result_url"] = f"/api/files/{session_id}/bg_removed.webm"
            tasks[task_id]["result_filename"] = "bg_removed.webm"

        tasks[task_id]["progress"] = 1.0
        tasks[task_id]["done"] = True
        tasks[task_id]["status"] = "완료!"
    except Exception as e:
        import traceback
        traceback.print_exc()
        tasks[task_id]["status"] = f"오류 발생: {str(e)}"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["done"] = True


@app.post("/api/bg-process")
async def bg_process(req: BgRemoveRequest):
    """배경 제거 처리 시작"""
    bg_sessions = [k for k in sessions if k.startswith("bg_")]
    if not bg_sessions:
        return JSONResponse({"error": "배경 제거 세션을 찾을 수 없습니다."}, status_code=400)

    session_id = bg_sessions[-1]
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "progress": 0.0,
        "done": False,
        "status": "AI 배경 제거 중...",
    }

    t = threading.Thread(
        target=run_bg_remove_task,
        args=(task_id, session_id, req.output_mode, req.quality),
    )
    t.daemon = True
    t.start()

    return {"task_id": task_id}


@app.post("/api/system/open-folder-dialog")
async def open_folder_dialog():
    """macOS Finder / 네이티브 폴더 선택 다이얼로그를 직접 호출"""
    import platform, subprocess
    selected_path = ""
    system = platform.system()

    if system == "Darwin":
        cmd = """
        osascript -e 'try
            set folderPath to POSIX path of (choose folder with prompt "워터마크를 제거할 이미지 폴더를 선택하세요:")
            return folderPath
        on error
            return ""
        end try'
        """
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        selected_path = res.stdout.strip()

    if selected_path:
        return {"folder_path": selected_path}
    else:
        return {"folder_path": "", "canceled": True}


# ──────────────────────────────────────────────
# 이미지 일괄 워터마크 제거 (Image Batch Processing)
# ──────────────────────────────────────────────

class ImageBatchFolderLoadRequest(BaseModel):
    folder_path: str


class ImageBatchFolderProcessRequest(BaseModel):
    folder_path: str
    box: BoxModel
    dilate: int = 4
    mode: str = "auto"
    save_mode: str = "new_folder"  # 'new_folder' or 'overwrite'


def read_image_utf8(path: str):
    """한글 및 특수문자 경로를 안전하게 지원하는 이미지 로드"""
    with open(path, "rb") as f:
        bytes_data = bytearray(f.read())
    np_arr = np.asarray(bytes_data, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def save_image_utf8(path: str, img_bgr, ext: str):
    """원본 해상도, 확장자 및 최고 화질을 100% 보존하며 안전하게 저장"""
    ext_lower = ext.lower()
    params = []
    if ext_lower in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 98]
    elif ext_lower == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif ext_lower == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 100]

    success, enc = cv2.imencode(ext_lower, img_bgr, params)
    if success:
        with open(path, "wb") as f:
            f.write(enc.tobytes())
    else:
        raise IOError(f"Failed to encode image to {ext}")


def inpaint_single_image(img_bgr, box: dict, dilate_px: int = 4, mask_mode: str = "auto"):
    """단일 이미지 워터마크 제거 및 1:1 고정밀 원본 합성"""
    bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
    h, w = img_bgr.shape[:2]

    # 주변 여백
    margin = 32
    rx = max(0, bx - margin)
    ry = max(0, by - margin)
    rw = min(w - rx, bw + margin * 2)
    rh = min(h - ry, bh + margin * 2)

    roi = img_bgr[ry:ry + rh, rx:rx + rw].copy()
    local_box = (bx - rx, by - ry, bw, bh)

    inpaint_mask = create_precise_shape_mask(roi, local_box, dilate_px=dilate_px, mask_mode=mask_mode)
    blend_mask = create_feathered_blend_mask(inpaint_mask, feather_radius=3)
    blend_mask_3ch = np.stack([blend_mask] * 3, axis=-1)

    inpainted_roi = inpainter.inpaint(roi, inpaint_mask)

    orig_roi_f = roi.astype(np.float32)
    inpaint_f = inpainted_roi.astype(np.float32)
    blended = inpaint_f * blend_mask_3ch + orig_roi_f * (1.0 - blend_mask_3ch)

    result = img_bgr.copy()
    result[ry:ry + rh, rx:rx + rw] = blended.astype(np.uint8)
    return result


def run_image_batch_folder_task(task_id: str, folder_path: str, box: dict, dilate: int, mode: str, save_mode: str):
    import unicodedata, re
    try:
        norm_folder = unicodedata.normalize("NFC", folder_path.strip().strip("'\""))
        p = Path(norm_folder).expanduser().resolve()
        if not p.is_dir():
            norm_folder_nfd = unicodedata.normalize("NFD", folder_path.strip().strip("'\""))
            p = Path(norm_folder_nfd).expanduser().resolve()
            if not p.is_dir():
                raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = [f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in valid_exts and not f.startswith(".")]
        files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])

        if not files:
            raise ValueError("해당 폴더에 지원되는 이미지 파일이 없습니다.")

        total_count = len(files)
        tasks[task_id]["total"] = total_count

        # 출력 폴더 결정
        if save_mode == "overwrite":
            backup_dir = p.parent / f"{p.name}_backup"
            if not backup_dir.exists():
                shutil.copytree(p, backup_dir)
            out_dir = p
        else:
            out_dir = p.parent / f"{p.name}_clean"
            out_dir.mkdir(parents=True, exist_ok=True)

        for i, fname in enumerate(files):
            tasks[task_id]["progress"] = i / total_count
            tasks[task_id]["status"] = f"[{i+1}/{total_count}] {fname} 처리 중..."

            src_file = p / fname
            dst_file = out_dir / fname

            img_bgr = read_image_utf8(str(src_file))
            if img_bgr is None:
                continue

            cleaned_img = inpaint_single_image(img_bgr, box, dilate_px=dilate, mask_mode=mode)
            ext = os.path.splitext(fname)[1]
            save_image_utf8(str(dst_file), cleaned_img, ext)

        tasks[task_id]["progress"] = 1.0
        tasks[task_id]["done"] = True
        tasks[task_id]["output_dir"] = str(out_dir)
        tasks[task_id]["status"] = f"총 {total_count}개 이미지 워터마크 제거 완료!"
    except Exception as e:
        import traceback
        traceback.print_exc()
        tasks[task_id]["status"] = f"오류 발생: {str(e)}"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["done"] = True


@app.post("/api/image-batch/load-folder")
async def load_image_folder(req: ImageBatchFolderLoadRequest):
    """지정된 폴더를 탐색하여 첫 번째 이미지를 로드하고 이미지 목록을 반환"""
    import unicodedata, re, base64
    folder_str = req.folder_path.strip().strip("'\"")
    if not folder_str:
        return JSONResponse({"error": "폴더 경로를 입력해 주세요."}, status_code=400)

    p = Path(unicodedata.normalize("NFC", folder_str)).expanduser().resolve()
    if not p.is_dir():
        p = Path(unicodedata.normalize("NFD", folder_str)).expanduser().resolve()
        if not p.is_dir():
            return JSONResponse({"error": f"존재하지 않는 폴더 경로입니다: {folder_str}"}, status_code=400)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = [f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in valid_exts and not f.startswith(".")]
    files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])

    if not files:
        return JSONResponse({"error": "해당 폴더에 이미지 파일(JPG, PNG, WebP)이 없습니다."}, status_code=400)

    first_file = p / files[0]
    first_bgr = read_image_utf8(str(first_file))
    if first_bgr is None:
        return JSONResponse({"error": f"첫 번째 이미지 파일을 읽을 수 없습니다: {files[0]}"}, status_code=400)

    h, w = first_bgr.shape[:2]

    # 세션 디렉토리에 첫 프레임 저장하여 URL 생성
    session_id = f"imgbatch_{uuid.uuid4()}"
    sess_dir = TEMP_DIR / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    preview_path = sess_dir / "first_preview.png"
    save_image_utf8(str(preview_path), first_bgr, ".png")

    sessions[session_id] = {
        "folder_path": str(p),
        "files": files,
        "width": w,
        "height": h,
    }

    return {
        "session_id": session_id,
        "first_image_url": f"/api/files/{session_id}/first_preview.png",
        "first_image_name": files[0],
        "width": w,
        "height": h,
        "count": len(files),
        "sample_files": files[:8],
        "all_files": files,
    }


class ImageBatchSelectImageRequest(BaseModel):
    session_id: str
    filename: str


@app.post("/api/image-batch/select-image")
async def select_image(req: ImageBatchSelectImageRequest):
    """폴더 내에서 사용자가 선택한 이미지를 기준 이미지로 로드"""
    sess = sessions.get(req.session_id)
    if not sess or "folder_path" not in sess:
        return JSONResponse({"error": "세션을 찾을 수 없습니다."}, status_code=400)

    p = Path(sess["folder_path"])
    target_file = p / req.filename
    if not target_file.is_file():
        return JSONResponse({"error": f"파일을 찾을 수 없습니다: {req.filename}"}, status_code=404)

    img_bgr = read_image_utf8(str(target_file))
    if img_bgr is None:
        return JSONResponse({"error": "이미지를 읽을 수 없습니다."}, status_code=400)

    h, w = img_bgr.shape[:2]
    sess_dir = TEMP_DIR / req.session_id
    preview_name = f"preview_{uuid.uuid4().hex[:8]}.png"
    preview_path = sess_dir / preview_name
    save_image_utf8(str(preview_path), img_bgr, ".png")

    return {
        "image_url": f"/api/files/{req.session_id}/{preview_name}",
        "filename": req.filename,
        "width": w,
        "height": h,
    }


@app.post("/api/image-batch/upload-single")
async def upload_single_image(file: UploadFile = File(...)):
    """단일 이미지를 직접 업로드하여 워터마크 영역 지정용으로 사용"""
    session_id = f"imgbatch_{uuid.uuid4()}"
    sess_dir = TEMP_DIR / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)

    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse({"error": "이미지 파일을 읽을 수 없습니다."}, status_code=400)

    h, w = img_bgr.shape[:2]
    preview_path = sess_dir / "first_preview.png"
    save_image_utf8(str(preview_path), img_bgr, ".png")

    sessions[session_id] = {
        "single_image": str(preview_path),
        "width": w,
        "height": h,
    }

    return {
        "session_id": session_id,
        "first_image_url": f"/api/files/{session_id}/first_preview.png",
        "first_image_name": file.filename,
        "width": w,
        "height": h,
        "count": 1,
    }


@app.post("/api/image-batch/process-folder")
async def process_image_folder(req: ImageBatchFolderProcessRequest):
    """폴더 내 이미지 일괄 워터마크 제거 작업 시작"""
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "progress": 0.0,
        "done": False,
        "status": "이미지 일괄 처리 준비 중...",
    }

    t = threading.Thread(
        target=run_image_batch_folder_task,
        args=(
            task_id,
            req.folder_path,
            req.box.model_dump(),
            req.dilate,
            req.mode,
            req.save_mode,
        ),
    )
    t.daemon = True
    t.start()

    return {"task_id": task_id}


@app.post("/api/image-batch/process-files")
async def process_image_files(
    files: list[UploadFile] = File(...),
    x: int = 0,
    y: int = 0,
    w: int = 70,
    h: int = 70,
    dilate: int = 4,
    mode: str = "auto",
):
    """웹 브라우저에서 다중 업로드된 이미지들을 일괄 처리 후 ZIP 파일 생성"""
    import zipfile
    session_id = f"imgbatch_{uuid.uuid4()}"
    sess_dir = TEMP_DIR / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    zip_path = sess_dir / "cleaned_images.zip"

    box = {"x": x, "y": y, "w": w, "h": h}

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            file_bytes = await file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            cleaned_img = inpaint_single_image(img_bgr, box, dilate_px=dilate, mask_mode=mode)
            ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
            success, enc = cv2.imencode(ext, cleaned_img)
            if success:
                zipf.writestr(file.filename, enc.tobytes())

    return {
        "zip_url": f"/api/files/{session_id}/cleaned_images.zip",
        "count": len(files),
    }


if __name__ == "__main__":
    print("✨ Video Edit Editor 웹 서버 시작 중...")
    print("👉 브라우저 주소: http://127.0.0.1:8888")
    uvicorn.run(app, host="127.0.0.1", port=8888)
