import os
import re
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ========= 캐시 디렉토리(쓰기 가능한 경로) =========
BASE_DIR = os.getcwd()
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
HF_DIR = os.path.join(CACHE_DIR, "hf")
TORCH_DIR = os.path.join(CACHE_DIR, "torch")
EASYOCR_DIR = os.path.join(CACHE_DIR, "easyocr")
for d in [CACHE_DIR, HF_DIR, TORCH_DIR, EASYOCR_DIR]:
    os.makedirs(d, exist_ok=True)

os.environ["HF_HOME"] = HF_DIR
os.environ["TORCH_HOME"] = TORCH_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_DIR
os.environ["EASYOCR_MODULE_PATH"] = EASYOCR_DIR

# ========= 페이지 설정 =========
st.set_page_config(page_title="이미지 기반 PDP 자동화", layout="centered")
st.markdown("""
<style>
  .main .block-container { max-width: 1000px !important; margin: 0 auto !important; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 기반 PDP 생성 자동화 솔루션")
st.write("✅ App is up")  # 부팅 확인

# ========= 지연 로드 유틸 =========
def get_blip():
    """
    BLIP 모델을 첫 사용 시에만 로드 (지연 로드).
    무거운 import와 다운로드를 함수 내부로 넣어 부팅 타임아웃 방지.
    """
    if "blip_loaded" not in st.session_state:
        with st.spinner("BLIP 모델 로드 중... (최초 3~8분 소요)"):
            # 무거운 모듈 import를 여기서
            from transformers import BlipProcessor, BlipForConditionalGeneration
            proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            mdl = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            st.session_state["blip_processor"] = proc
            st.session_state["blip_model"] = mdl
            st.session_state["blip_loaded"] = True
    return st.session_state["blip_processor"], st.session_state["blip_model"]

def get_easyocr():
    """EasyOCR를 첫 사용 시에만 로드 (지연 로드)."""
    if "easyocr_loaded" not in st.session_state:
        with st.spinner("EasyOCR 로드 중... (최초 1~3분 소요)"):
            import easyocr  # 지연 import
            reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=EASYOCR_DIR)
            st.session_state["easyocr_reader"] = reader
            st.session_state["easyocr_loaded"] = True
    return st.session_state["easyocr_reader"]

# ========= 업로드 UI =========
uploaded = st.file_uploader("이미지를 업로드하세요", type=["png","jpg","jpeg"])
if not uploaded:
    st.info("이미지를 선택해주세요")
    st.stop()

from PIL import Image
img = Image.open(uploaded).convert("RGB")

# 미리보기(라벨 1 표시)
annotated = img.copy()
draw = ImageDraw.Draw(annotated)
font = ImageFont.load_default()
draw.text((10, 10), "1", fill="red", font=font)
st.image(annotated, use_container_width=True)

# ========= 기능 함수들 =========
def extract_text_via_easyocr(pil_img):
    reader = get_easyocr()
    arr = np.array(pil_img)
    lines = reader.readtext(arr, detail=0, paragraph=True)
    return "\n".join(lines)

def generate_blip_caption(pil_img):
    # torch도 지연 import (부팅 시 메모리/시간 절약)
    import torch
    processor, blip_model = get_blip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    blip_model.to(device)
    out = blip_model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)

def make_alt_candidates(pil_img):
    base = generate_blip_caption(pil_img)
    cands = [base, f"{base} in a modern environment", f"LG product - {base}"]
    # 중복 제거 후 최대 3개
    uniq = []
    for c in cands:
        if c not in uniq:
            uniq.append(c)
        if len(uniq) == 3:
            break
    return uniq

COMPONENT_DEFS = {
    "ST0001": {"name": "Hero banner", "has_image": True},
    "ST0002": {"name": "Tab Anchor", "has_image": False},
    "ST0003": {"name": "Title", "has_image": False},
    "ST0013": {"name": "Side Image", "has_image": True},
    "ST0014": {"name": "Layered Text", "has_image": True},
}
CTA_KEYWORDS = [
    "learn more","shop now","buy now","see more","read more",
    "click here","get started","try now","explore","discover","buy it now"
]

def classify_elements(lines):
    extracted, cleaned = [], []
    for L in lines:
        line = L
        for kw in CTA_KEYWORDS:
            if kw in line.lower():
                extracted.append(kw)
                import re as _re
                line = _re.sub(rf"\b{kw}\b", "", line, flags=_re.IGNORECASE).strip()
        if line:
            cleaned.append(line)
    disclaimers = [l for l in cleaned if l.startswith("*")]
    body = [l for l in cleaned if not l.startswith("*")]
    ey = body.pop(0) if body and body[0].isupper() else ""
    hl = body.pop(0) if body else ""
    bc = "\n".join(body)
    return {"Eyebrow": ey, "Headline": hl, "Bodycopy": bc,
            "Disclaimer": "\n".join(disclaimers),
            "CTA": (extracted[0] if extracted else "")}

def recommend_components(classified, has_image=True):
    return [cid for cid, comp in COMPONENT_DEFS.items() if comp["has_image"] == has_image]

# ========= UI 동작 =========
col1, col2 = st.columns(2)
with col1:
    if st.button("🖼️ Alt Text 생성"):
        st.session_state["candidates"] = make_alt_candidates(img)

with col2:
    if st.button("🚀 OCR 실행 (EasyOCR)"):
        txt = extract_text_via_easyocr(img)
        lines = txt.split("\n")
        st.session_state["ocr_done"] = True
        st.session_state["ocr_text"] = txt
        st.session_state["classified"] = classify_elements(lines)
        st.session_state["recs"] = recommend_components(st.session_state["classified"])

# 결과 표시
if "candidates" in st.session_state:
    choice = st.radio("Alt Text 후보 선택:", st.session_state["candidates"], key="alt_choice")
    st.subheader("🖼️ Selected Alt Text")
    st.code(choice)

if st.session_state.get("ocr_done"):
    st.subheader("📋 OCR 결과")
    st.text_area("", st.session_state["ocr_text"], height=200)
    st.subheader("📑 텍스트 분류 결과")
    for k, v in st.session_state["classified"].items():
        st.markdown(f"**{k}:** {v or '—'}")
    with st.expander("🧩 컴포넌트 추천"):
        recs = st.session_state["recs"]
        if recs:
            sel = st.selectbox("추천된 컴포넌트 중 하나를 선택하세요:", recs,
                               format_func=lambda cid: f"{cid} – {COMPONENT_DEFS[cid]['name']}")
            st.markdown(f"**Selected Component:** {sel} – {COMPONENT_DEFS[sel]['name']}")
            st.session_state["selected_component"] = sel
        else:
            st.write("추천할 컴포넌트가 없습니다.")
    df = pd.DataFrame.from_dict(st.session_state["classified"], orient='index', columns=['Content'])
    csv = df.to_csv(encoding='utf-8-sig')
    st.download_button("💾 엑셀 파일 다운로드", data=csv, file_name="ocr_result.csv", mime="text/csv")

if st.session_state.get("ocr_done") and st.button("📤 PDP생성하기 (WCM으로 전송하기)"):
    st.caption("※ 해당 기능은 기획 단계의 구현이며 실제 적용되어 있지 않습니다.")
