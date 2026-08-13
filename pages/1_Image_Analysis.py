import streamlit as st
from PIL import Image

st.title("Image Analysis")

uploaded = st.file_uploader(
    "Upload HER2 image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width='stretch')

    with col2:
        st.subheader("Metadata")
        st.write(image.size)
        st.write(image.mode)