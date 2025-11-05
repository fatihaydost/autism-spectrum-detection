import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import cm
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import AutismDetector  # noqa: E402


@st.cache_resource(show_spinner=False)
def load_detector():
    # Modeli yalnızca ilk çağrıda belleğe alıp sonraki taleplerde tekrar kullanıyoruz.
    return AutismDetector()


st.set_page_config(page_title="Autism Detection", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(46,51,90,0.8), rgba(20,20,35,0.9));
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 12px 32px rgba(15, 17, 26, 0.35);
        color: #f9fafb;
    }
    .metric-card h2 {
        font-size: 2.1rem;
        margin-bottom: 0.4rem;
    }
    .metric-card span {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.85;
    }
    .image-frame img {
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(12, 15, 32, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MAIN_DISPLAY_SIZE = (360, 360)
GRADCAM_DISPLAY_SIZE = (320, 320)
CANVAS_COLOR = (17, 20, 34)
try:
    RESAMPLE_METHOD = Image.Resampling.LANCZOS  # Pillow >= 9
except AttributeError:  # pragma: no cover
    RESAMPLE_METHOD = Image.LANCZOS


def prettify_label(label: str) -> str:
    return label.replace("_", " ").title()


def prepare_display_image(image: Image.Image | None, *, size: tuple[int, int]) -> Image.Image | None:
    if image is None:
        return None
    rgb_image = image.convert("RGB")
    canvas = Image.new("RGB", size, CANVAS_COLOR)
    fitted = ImageOps.contain(rgb_image, size, method=RESAMPLE_METHOD)
    offset = (
        (size[0] - fitted.width) // 2,
        (size[1] - fitted.height) // 2,
    )
    canvas.paste(fitted, offset)
    return canvas


def build_gradcam_overlay(
    base_image: Image.Image,
    heatmap_img: Image.Image | None,
    alpha: float,
) -> Image.Image | None:
    if heatmap_img is None:
        return None
    base_rgb = base_image.convert("RGB")
    resized_heatmap = heatmap_img.resize(base_rgb.size, resample=Image.BILINEAR)
    heatmap_np = np.array(resized_heatmap, dtype=np.float32) / 255.0
    colored_heatmap = cm.get_cmap("jet")(heatmap_np)[..., :3]
    base_np = np.array(base_rgb, dtype=np.float32) / 255.0
    overlay_np = (1 - alpha) * base_np + alpha * colored_heatmap
    overlay_np = np.clip(overlay_np, 0, 1)
    return Image.fromarray((overlay_np * 255).astype(np.uint8))

st.title("Yüz Görsellerinden Otizm Tespiti")
st.write(
    "Bu arayüz, önceden eğitilmiş derin öğrenme modelini kullanarak yüklenen yüz "
    "görsellerini otistik / tipik olarak sınıflandırır ve Grad-CAM ile hangi bölgelerin "
    "kararı etkilediğini görselleştirir."
)

uploaded_file = st.file_uploader(
    "Bir yüz görseli yükleyin", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Model çalıştırılıyor..."):
        detector = load_detector()
        result = detector.predict(image)

    display_original = prepare_display_image(image, size=MAIN_DISPLAY_SIZE)
    raw_heatmap_img = result.heatmap.convert("RGB") if result.heatmap else None
    display_heatmap = prepare_display_image(raw_heatmap_img, size=GRADCAM_DISPLAY_SIZE)

    st.success(f"Tahmin edilen sınıf: **{prettify_label(result.label)}**")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("Girdi Görseli")
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(display_original, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    probabilities = pd.Series(result.probabilities, name="olasılık")
    pred_probability = probabilities[result.label]
    prob_display = (
        probabilities.rename(index=prettify_label)
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Sınıf", "olasılık": "Olasılık (%)"})
    )

    with col_right:
        st.subheader("Model Sonucu")
        st.markdown(
            f"""
            <div class="metric-card">
                <span>Tahmin Edilen Sınıf</span>
                <h2>{prettify_label(result.label)}</h2>
                <p>Güven skoru: <strong>{pred_probability:.1%}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        chart = (
            alt.Chart(prob_display)
            .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10)
            .encode(
                x=alt.X("Olasılık (%)", scale=alt.Scale(domain=(0, 100))),
                y=alt.Y("Sınıf", sort="-x"),
                color=alt.Color("Sınıf", legend=None, scale=alt.Scale(scheme="tealblues")),
                tooltip=["Sınıf", "Olasılık (%)"],
            )
            .properties(height=220)
        )
        # Hem görsel grafik hem tabloyu birlikte sunarak kullanıcıya hızlı bir karşılaştırma sağlıyoruz.
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(
            prob_display.style.format({"Olasılık (%)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    tab_gradcam, tab_heatmap = st.tabs(["Grad-CAM", "Isı Haritası"])

    with tab_gradcam:
        if result.heatmap:
            gradcam_opacity = st.slider(
                "Grad-CAM Opaklığı",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                key="gradcam_opacity_slider",
            )
            adjusted_overlay = build_gradcam_overlay(
                image, result.heatmap, gradcam_opacity
            )
            display_overlay = prepare_display_image(
                adjusted_overlay, size=GRADCAM_DISPLAY_SIZE
            )
            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
            st.image(display_overlay, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Grad-CAM çıktısı üretilemedi.")

    with tab_heatmap:
        if display_heatmap:
            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
            st.image(display_heatmap, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Isı haritası bulunmuyor.")

else:
    st.info("Başlamak için bir görsel yükleyin.")
