from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from inference import COLOR_OPTIONS, analyze_image, get_haircut_recommendations, load_models


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Style Strand AI",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)

hair_studio = components.declare_component(
    "hair_studio",
    path=str(APP_DIR / "frontend"),
)

try:
    models = load_models()
except Exception as error:
    st.error("The hair analysis models could not be loaded.")
    st.exception(error)
    st.stop()

upload = st.session_state.get("hair_studio")
result = None

if isinstance(upload, dict) and upload.get("imageData"):
    try:
        result = analyze_image(upload["imageData"], models)
        if result.get("hairDetected"):
            result["recommendations"] = get_haircut_recommendations(
                result["hairType"], models["segmentation"]
            )
    except Exception as error:
        result = {"error": str(error)}

hair_studio(
    result=result,
    colors=COLOR_OPTIONS,
    key="hair_studio",
    default=None,
)