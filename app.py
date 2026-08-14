import streamlit as st

st.set_page_config(
    page_title="HER2 Fractal Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("HER2 Fractal Lab")

st.markdown("""
### Estación de Trabajo de Patología Digital Interactiva

Analizar imágenes de inmunohistoquímica HER2 usando:

- Deconvolución de color (Ruifrok & Johnston)
- Segmentación de membrana
- Dimensión Fractal 
- Lacunaridad
- Espectro Multifractal
- Análisis por lotes
- Comparación estadística
- Clustering automático
""")

st.info(
    "Seleccionar opción en el menú lateral."
)