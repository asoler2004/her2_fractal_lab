import streamlit as st

st.set_page_config(
    page_title="HER2 Fractal Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("HER2 Fractal Lab")

st.markdown("""
### Interactive Digital Pathology Workstation

Analyze HER2 immunohistochemistry images using:

- Color deconvolution (Ruifrok & Johnston)
- Membrane segmentation
- Fractal Dimension
- Lacunarity
- Multifractal Spectrum
- Batch analysis
- Statistical comparison
- Automatic clustering
""")

st.info(
    "Select a page from the sidebar to begin."
)