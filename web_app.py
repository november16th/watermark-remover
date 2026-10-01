#!/opt/anaconda3/bin/python3
"""
FastAPI 기반 고품질 AI 영상 워터마크 정밀 제거 웹 서버
"""

import os
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

# rembg 모델 세션 (fast: 초고속 u2netp ~0.1s/frame, quality: 고정밀 isnet)
rembg_sessions = {
    "fast": rembg_new_session("u2netp"),
    "quality": rembg_new_session("isnet-general-use"),
}


class BoxModel(BaseModel):
    x: int
    y: int
    w: int
    h: int


class ProcessRequest(BaseModel):
    box: BoxModel
    dilate: int = 4
    mode: str = "auto"  # 'auto' (다이아몬드 정밀), 'diamond' (강제 다이아몬드), 'box' (전체 사각형)


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
    }

    return {
        "session_id": session_id,
        "frame_url": f"/api/files/{session_id}/frame_first.png",
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

    # 정밀 스파클 마스크 생성
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
    sess = sessions[session_id]
    input_path = sess["input_video"]
    sess_dir = TEMP_DIR / session_id
    output_path = sess_dir / "cleaned.mp4"

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ROI 영역
    margin = 28
    bx, by, bw, bh = box.x, box.y, box.w, box.h
    rx = max(0, bx - margin)
    ry = max(0, by - margin)
    rw = min(width - rx, bw + 2 * margin)
    rh = min(height - ry, bh + 2 * margin)

    local_bx = bx - rx
    local_by = by - ry

    # 첫 프레임 기준 고정 정밀 마스크 생성
    ret_first, first_frame = cap.read()
    first_roi = first_frame[ry:ry + rh, rx:rx + rw]
    inpaint_mask = create_precise_shape_mask(
        first_roi, (local_bx, local_by, bw, bh), dilate_px=dilate_px, mask_mode=mode
    )
    # 실제 마스킹된 스파클 외곽 3px만 부드럽게 합성하여 주변 배경 왜곡을 0으로 만듦
    blend_mask = create_feathered_blend_mask(inpaint_mask, feather_radius=3)
    blend_mask_3ch = np.stack([blend_mask] * 3, axis=-1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    temp_raw = sess_dir / "temp_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_raw), fourcc, fps, (width, height))

    prev_roi = None
    # 0.70으로 안정화 가중치를 높여 프레임 간 펄럭임/일렁거림 완화
    ema_alpha = 0.70
    idx = 0

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

        # 정밀 합성: 스파클 픽셀 자리만 원본에 대치, 바깥 배경은 원본 100% 보존
        orig_roi_f = frame[ry:ry + rh, rx:rx + rw].astype(np.float32)
        inpaint_f = inpainted_roi.astype(np.float32)
        blended = inpaint_f * blend_mask_3ch + orig_roi_f * (1.0 - blend_mask_3ch)
        frame[ry:ry + rh, rx:rx + rw] = blended.astype(np.uint8)

        out.write(frame)
        idx += 1
        tasks[task_id]["progress"] = idx / max(total_frames, 1)

    cap.release()
    out.release()

    tasks[task_id]["status"] = "오디오 합성 및 최종 인코딩 중..."
    merge_audio_ffmpeg(str(temp_raw), input_path, str(output_path), crf=18)

    if temp_raw.exists():
        temp_raw.unlink()

    tasks[task_id]["progress"] = 1.0
    tasks[task_id]["done"] = True
    tasks[task_id]["status"] = "완료!"
    tasks[task_id]["clean_url"] = f"/api/files/{session_id}/cleaned.mp4"
    tasks[task_id]["orig_url"] = f"/api/files/{session_id}/input.mp4"


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
    pil_removed = rembg_remove(pil_frame, session=rembg_sessions["fast"])
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

        active_session = rembg_sessions.get(quality, rembg_sessions["fast"])
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


if __name__ == "__main__":
    print("✨ Video Edit Editor 웹 서버 시작 중...")
    print("👉 브라우저 주소: http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)
