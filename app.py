"""
합창 악보 추출기 - Streamlit 웹앱
악보 영상 업로드 → 악보 프레임 자동 추출 → 중복 제거 → 인터랙티브 크롭 → A4 PDF/PNG 내보내기
"""

import io
import os
import tempfile
import zipfile

import cv2
import imagehash
import numpy as np
import streamlit as st
from fpdf import FPDF
from PIL import Image, ImageDraw

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(page_title="악보 추출기", layout="wide")


# ──────────────────────────────────────────────
# Session state 초기화
# ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "extracted_frames": [],
        "timestamps": [],
        "selected": [],
        "individual_crops": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ──────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────
def extract_frames(video_path: str, interval: float) -> list[tuple[np.ndarray, float]]:
    """interval(초) 간격으로 프레임 추출. (frame, timestamp) 리스트 반환."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("영상 파일을 열 수 없습니다.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * interval))

    frames = []
    idx = 0
    while idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = idx / fps
        frames.append((frame, timestamp))
        idx += frame_interval

    cap.release()
    return frames


def is_score_frame(frame: np.ndarray, brightness_threshold: float) -> bool:
    """밝기 + 수평선 감지로 악보 프레임 여부 판단."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 밝기 분석
    bright_ratio = np.mean(gray > 200)
    if bright_ratio < brightness_threshold:
        return False

    # 수평선 감지
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    h, w = binary.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 5, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    # 수평선이 있는 행 카운트
    row_sums = np.sum(horizontal, axis=1)
    line_rows = np.where(row_sums > w * 0.3 * 255)[0]

    if len(line_rows) == 0:
        return False

    # distinct 수평선 그룹 카운트 (10px 이상 간격)
    groups = 1
    for i in range(1, len(line_rows)):
        if line_rows[i] - line_rows[i - 1] > 10:
            groups += 1

    return groups >= 3


def remove_duplicates(
    frames: list[tuple[np.ndarray, float]], threshold: int
) -> list[tuple[np.ndarray, float]]:
    """perceptual hash로 중복 프레임 제거."""
    unique = []
    hashes = []

    for frame, ts in frames:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h = imagehash.phash(pil_img)

        is_unique = all(abs(h - existing) >= threshold for existing in hashes)
        if is_unique:
            unique.append((frame, ts))
            hashes.append(h)

    return unique


def apply_crop(img: Image.Image, top: int, bottom: int, left: int, right: int) -> Image.Image:
    """퍼센트 기반 크롭 적용."""
    w, h = img.size
    x1 = int(w * left / 100)
    y1 = int(h * top / 100)
    x2 = int(w * (100 - right) / 100)
    y2 = int(h * (100 - bottom) / 100)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def create_crop_preview(img: Image.Image, top: int, bottom: int, left: int, right: int) -> Image.Image:
    """크롭 미리보기: 반투명 오버레이 + 빨간 테두리."""
    preview = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    x1 = int(w * left / 100)
    y1 = int(h * top / 100)
    x2 = int(w * (100 - right) / 100)
    y2 = int(h * (100 - bottom) / 100)

    # 반투명 마스크 (크롭 영역 밖)
    mask_color = (0, 0, 0, 120)
    if y1 > 0:
        draw.rectangle([0, 0, w, y1], fill=mask_color)
    if y2 < h:
        draw.rectangle([0, y2, w, h], fill=mask_color)
    if x1 > 0:
        draw.rectangle([0, y1, x1, y2], fill=mask_color)
    if x2 < w:
        draw.rectangle([x2, y1, w, y2], fill=mask_color)

    # 빨간 테두리
    border_width = max(2, min(w, h) // 200)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=border_width)

    preview = Image.alpha_composite(preview, overlay)
    return preview.convert("RGB")


def create_png_zip(images: list[Image.Image]) -> bytes:
    """PNG ZIP 생성."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img in enumerate(images):
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            zf.writestr(f"score_{i + 1:03d}.png", img_buf.getvalue())
    return buf.getvalue()


def create_auto_layout_pdf(
    images: list[Image.Image], margin_mm: float, spacing_mm: float
) -> bytes:
    """A4 PDF - 이미지를 위에서부터 순서대로 배치."""
    pdf = FPDF(unit="mm", format="A4")
    page_w, page_h = 210, 297
    content_w = page_w - 2 * margin_mm

    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    cursor_y = margin_mm

    for img in images:
        # 이미지를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp, format="PNG")
            tmp_path = tmp.name

        try:
            img_w, img_h = img.size
            # A4 콘텐츠 너비에 맞추어 비율 조정
            scale = content_w / (img_w * 0.264583)  # px → mm (approx)
            display_w = content_w
            display_h = img_h * 0.264583 * scale

            # 페이지 넘침 시 새 페이지
            if cursor_y + display_h > page_h - margin_mm:
                pdf.add_page()
                cursor_y = margin_mm

            # 수평 중앙 정렬
            x = margin_mm + (content_w - display_w) / 2
            pdf.image(tmp_path, x=x, y=cursor_y, w=display_w)
            cursor_y += display_h + spacing_mm
        finally:
            os.unlink(tmp_path)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def create_individual_pdf(images: list[Image.Image]) -> bytes:
    """개별 PDF - 1이미지 = 1 A4 페이지, 중앙 배치."""
    pdf = FPDF(unit="mm", format="A4")
    page_w, page_h = 210, 297
    margin = 10

    for img in images:
        pdf.add_page()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp, format="PNG")
            tmp_path = tmp.name

        try:
            img_w, img_h = img.size
            content_w = page_w - 2 * margin
            content_h = page_h - 2 * margin

            # 비율 유지하며 페이지에 맞춤
            scale_w = content_w / (img_w * 0.264583)
            scale_h = content_h / (img_h * 0.264583)
            scale = min(scale_w, scale_h)

            display_w = img_w * 0.264583 * scale
            display_h = img_h * 0.264583 * scale

            x = (page_w - display_w) / 2
            y = (page_h - display_h) / 2
            pdf.image(tmp_path, x=x, y=y, w=display_w)
        finally:
            os.unlink(tmp_path)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────
st.sidebar.title("🎵 악보 추출기")
st.sidebar.markdown("악보 영상을 업로드하면 악보 프레임을 자동 추출합니다.")

uploaded_video = st.sidebar.file_uploader(
    "영상 파일 업로드",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="악보가 포함된 영상 파일을 업로드하세요.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("추출 설정")

frame_interval = st.sidebar.slider(
    "프레임 추출 간격 (초)", min_value=0.5, max_value=5.0, value=1.0, step=0.5
)
brightness_thresh = st.sidebar.slider(
    "밝기 임계값 (%)", min_value=20, max_value=80, value=40, step=5
)
hash_threshold = st.sidebar.slider(
    "중복 제거 임계값 (해밍 거리)", min_value=1, max_value=25, value=10, step=1
)

extract_btn = st.sidebar.button("🎼 악보 추출 시작", width="stretch")

# ──────────────────────────────────────────────
# 메인 영역
# ──────────────────────────────────────────────
st.title("합창 악보 추출기")
st.caption("악보 영상 업로드 → 악보 프레임 추출 → 크롭 → PDF/PNG 내보내기")

# ── 추출 로직 ──
if extract_btn:
    if uploaded_video is None:
        st.error("사이드바에서 영상 파일을 업로드해주세요.")
    else:
        tmp_dir = tempfile.mkdtemp()
        try:
            with st.status("악보 추출 중...", expanded=True) as status:
                # 1) 업로드된 영상을 임시 파일로 저장
                st.write("영상 파일을 준비하고 있습니다...")
                video_ext = os.path.splitext(uploaded_video.name)[1] or ".mp4"
                video_path = os.path.join(tmp_dir, f"video{video_ext}")
                with open(video_path, "wb") as vf:
                    vf.write(uploaded_video.getvalue())

                # 2) 프레임 추출
                st.write("프레임을 추출하고 있습니다...")
                try:
                    raw_frames = extract_frames(video_path, frame_interval)
                    st.write(f"총 {len(raw_frames)}개 프레임 추출됨")
                except Exception as e:
                    st.error(f"프레임 추출 실패: {e}")
                    status.update(label="프레임 추출 실패", state="error")
                    st.stop()

                # 3) 악보 프레임 감지
                st.write("악보 프레임을 감지하고 있습니다...")
                progress_bar = st.progress(0)
                score_frames = []
                for i, (frame, ts) in enumerate(raw_frames):
                    if is_score_frame(frame, brightness_thresh / 100):
                        score_frames.append((frame, ts))
                    progress_bar.progress((i + 1) / len(raw_frames))

                st.write(f"악보 프레임 {len(score_frames)}개 감지됨")

                if not score_frames:
                    st.warning("악보 프레임을 찾을 수 없습니다. 밝기 임계값을 조절해보세요.")
                    status.update(label="악보 미감지", state="error")
                    st.stop()

                # 4) 중복 제거
                st.write("중복 프레임을 제거하고 있습니다...")
                unique_frames = remove_duplicates(score_frames, hash_threshold)
                st.write(f"고유 프레임 {len(unique_frames)}개 (중복 {len(score_frames) - len(unique_frames)}개 제거)")

                status.update(label=f"추출 완료! (고유 프레임 {len(unique_frames)}개)", state="complete")

            # session_state에 저장
            pil_frames = []
            timestamps = []
            for frame, ts in unique_frames:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_frames.append(pil_img)
                timestamps.append(ts)

            st.session_state.extracted_frames = pil_frames
            st.session_state.timestamps = timestamps
            st.session_state.selected = [True] * len(pil_frames)
            st.session_state.individual_crops = {}

        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

# ── 결과 표시 ──
if st.session_state.extracted_frames:
    frames = st.session_state.extracted_frames
    timestamps = st.session_state.timestamps

    st.markdown("---")
    st.subheader(f"추출된 악보 프레임 ({len(frames)}개)")

    # ── 전체 크롭 설정 ──
    st.markdown("### ✂️ 전체 크롭 설정")

    # 인터랙티브 슬라이더 (범위 슬라이더로 즉시 반영)
    crop_h_range = st.slider(
        "좌우 크롭 범위 (%)",
        min_value=0, max_value=100, value=(0, 100),
        key="crop_h_slider",
        help="왼쪽/오른쪽 크롭 경계를 드래그로 조정",
    )
    crop_v_range = st.slider(
        "상하 크롭 범위 (%)",
        min_value=0, max_value=100, value=(0, 100),
        key="crop_v_slider",
        help="위/아래 크롭 경계를 드래그로 조정",
    )

    crop_left = crop_h_range[0]
    crop_right = 100 - crop_h_range[1]
    crop_top = crop_v_range[0]
    crop_bottom = 100 - crop_v_range[1]

    # 숫자 입력으로도 미세 조정 가능
    with st.expander("숫자로 미세 조정"):
        col_crop1, col_crop2, col_crop3, col_crop4 = st.columns(4)
        with col_crop1:
            crop_top = st.number_input("위 (%)", 0, 49, crop_top, key="crop_top")
        with col_crop2:
            crop_bottom = st.number_input("아래 (%)", 0, 49, crop_bottom, key="crop_bottom")
        with col_crop3:
            crop_left = st.number_input("왼쪽 (%)", 0, 49, crop_left, key="crop_left")
        with col_crop4:
            crop_right = st.number_input("오른쪽 (%)", 0, 49, crop_right, key="crop_right")

    has_crop = any([crop_top, crop_bottom, crop_left, crop_right])

    # 크롭 미리보기 (항상 표시)
    st.markdown("**크롭 미리보기** (첫 번째 이미지 기준)")
    preview = create_crop_preview(frames[0], crop_top, crop_bottom, crop_left, crop_right)
    st.image(preview, width="stretch")

    # ── 프레임 선택 ──
    st.markdown("### 📋 프레임 선택")
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        if st.button("전체 선택", width="stretch"):
            st.session_state.selected = [True] * len(frames)
            st.rerun()
    with col_sel2:
        if st.button("전체 해제", width="stretch"):
            st.session_state.selected = [False] * len(frames)
            st.rerun()

    # ── 3열 그리드 표시 ──
    cols_per_row = 3
    for row_start in range(0, len(frames), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            i = row_start + col_idx
            if i >= len(frames):
                break
            with cols[col_idx]:
                # 개별 크롭 가져오기
                ind_crop = st.session_state.individual_crops.get(i)
                if ind_crop:
                    display_img = apply_crop(frames[i], **ind_crop)
                elif has_crop:
                    display_img = apply_crop(
                        frames[i], crop_top, crop_bottom, crop_left, crop_right
                    )
                else:
                    display_img = frames[i]

                st.image(display_img, caption=f"#{i + 1} ({timestamps[i]:.1f}초)", width="stretch")

                st.session_state.selected[i] = st.checkbox(
                    "선택", value=st.session_state.selected[i], key=f"sel_{i}"
                )

                # 개별 크롭 오버라이드
                with st.expander(f"개별 크롭 #{i + 1}"):
                    use_individual = st.checkbox(
                        "개별 크롭 사용", key=f"use_ind_{i}",
                        value=i in st.session_state.individual_crops,
                    )
                    if use_individual:
                        ind_top = st.number_input("위 (%)", 0, 49, 0, key=f"ind_top_{i}")
                        ind_bottom = st.number_input("아래 (%)", 0, 49, 0, key=f"ind_bot_{i}")
                        ind_left = st.number_input("왼쪽 (%)", 0, 49, 0, key=f"ind_left_{i}")
                        ind_right = st.number_input("오른쪽 (%)", 0, 49, 0, key=f"ind_right_{i}")
                        st.session_state.individual_crops[i] = {
                            "top": ind_top,
                            "bottom": ind_bottom,
                            "left": ind_left,
                            "right": ind_right,
                        }
                    else:
                        st.session_state.individual_crops.pop(i, None)

    # ── 내보내기 ──
    st.markdown("---")
    st.subheader("📤 내보내기")

    # 선택된 이미지에 크롭 적용
    selected_images = []
    for i, img in enumerate(frames):
        if not st.session_state.selected[i]:
            continue
        ind_crop = st.session_state.individual_crops.get(i)
        if ind_crop:
            selected_images.append(apply_crop(img, **ind_crop))
        elif has_crop:
            selected_images.append(
                apply_crop(img, crop_top, crop_bottom, crop_left, crop_right)
            )
        else:
            selected_images.append(img)

    if not selected_images:
        st.warning("내보낼 프레임을 선택해주세요.")
    else:
        st.info(f"선택된 프레임: {len(selected_images)}개")

        export_cols = st.columns(3)

        with export_cols[0]:
            st.markdown("**PNG ZIP**")
            png_data = create_png_zip(selected_images)
            st.download_button(
                "📦 PNG ZIP 다운로드",
                data=png_data,
                file_name="score_images.zip",
                mime="application/zip",
                width="stretch",
            )

        with export_cols[1]:
            st.markdown("**A4 PDF (자동 배치)**")
            margin_mm = st.number_input("여백 (mm)", 5, 30, 10, key="pdf_margin")
            spacing_mm = st.number_input("이미지 간격 (mm)", 0, 20, 5, key="pdf_spacing")
            pdf_data = create_auto_layout_pdf(selected_images, margin_mm, spacing_mm)
            st.download_button(
                "📄 자동 배치 PDF 다운로드",
                data=pdf_data,
                file_name="score_auto_layout.pdf",
                mime="application/pdf",
                width="stretch",
            )

        with export_cols[2]:
            st.markdown("**개별 PDF**")
            ind_pdf_data = create_individual_pdf(selected_images)
            st.download_button(
                "📄 개별 PDF 다운로드",
                data=ind_pdf_data,
                file_name="score_individual.pdf",
                mime="application/pdf",
                width="stretch",
            )

        # 최종 미리보기
        st.markdown("---")
        st.subheader("👀 최종 미리보기")
        for i, img in enumerate(selected_images):
            st.image(img, caption=f"score_{i + 1:03d}", width="stretch")
