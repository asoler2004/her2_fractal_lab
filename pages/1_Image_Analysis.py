from pandas import col

from core.segmentation.cellpose_segmentation import CellposeSegmenter
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core.io.loader import load_image
from core.io.loader import SUPPORTED_FORMATS
from core.preprocessing.color_deconvolution import (color_deconvolution)
#from core.segmentation.membrane_segmentation import (segment_membrane,)
from skimage.color import label2rgb

@st.cache_resource
def load_cellpose(model_path=None):
    return CellposeSegmenter(
        model_path=model_path,
        gpu=False,
    )

st.title("HER2 Image Analysis")

uploaded = st.file_uploader(
    "Upload HER2 image",
    type=SUPPORTED_FORMATS,
)

if uploaded:
    
    image = load_image(uploaded)
    # Convert PIL → NumPy RGB
    image_array = np.array(image.convert("RGB"))
    st.subheader("Original Image")
    st.image(
        image,
        width="stretch",
    )
    st.divider()
    st.subheader("Color Deconvolution")
    st.caption(
            "Ruifrok–Johnston separation of the "
            "Hematoxylin and DAB stain components."
        )  

    deconv = color_deconvolution(image_array)
       
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(
            deconv.hematoxylin,
            caption="Hematoxylin",
            width="stretch"
        )
        st.caption("Counterstain component")
    with col2:
        st.image(
            deconv.dab,
            caption="DAB",
            width="stretch"
        )
        st.caption("HER2-associated chromogen component")

    with col3:
        st.image(
            deconv.eosin,
            caption="Eosin",
            width="stretch"
        )
        st.caption("Counterstain component")        
    st.markdown("### scikit-image (rgb2hed → hed2rgb)")
   
    with st.expander("About the stain channels"):

        st.markdown("""
            **Hematoxylin**
            - Counterstains cell nuclei.
            - Useful for assessing tissue morphology.

            **Eosin**
            - Highlights cytoplasm and extracellular structures.
            - In HER2 immunohistochemistry, this channel often contains relatively little information because eosin is typically absent or minimal.

            **DAB**
            - Brown chromogen indicating HER2 immunostaining.
            - This channel will be used for membrane segmentation and subsequent fractal analysis.
        """)

    st.divider()

    st.subheader("DAB Quantitative Analysis")

    dab = deconv.dab
    hem = deconv.hematoxylin
    eos = deconv.eosin

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Minimum", f"{dab.min():.3f}")

    with col2:
        st.metric("Maximum", f"{dab.max():.3f}")

    with col3:
        st.metric("Mean", f"{dab.mean():.3f}")

    with col4:
        st.metric("Median", f"{np.median(dab):.3f}")

    #---------------------------------------------
    # Segmentation
    #---------------------------------------------
    st.divider()

    st.subheader("Cell Segmentation")
    st.caption("Cell instance segmentation using Cellpose.")

    model_source = st.radio(
        "Cellpose model",
        [
        "Local checkpoint",
        "Pretrained model",
        ],
        horizontal=True,
    )

    if model_source == "Local checkpoint":
        checkpoint_path = st.text_input(
            "Cellpose checkpoint path",
            placeholder="/home/antonia/models/cellpose_model",
        )
    else:
        checkpoint_path = None

    cell_diameter = st.slider(
        "Estimated cell diameter (pixels)",
        min_value=5,
        max_value=150,
        value=48,
        step=1,
        help=(
            "Approximate diameter of cells in the image. "
            "Adjust this according to image magnification "
            "and cell size."
        ),
    )

    run_cellpose = st.button(
        "Run Cellpose",
        type="primary",
    )

    if run_cellpose:
        if model_source == "Pretrained model":
            st.warning(
                "The pretrained Cellpose model may need to be "
                "downloaded the first time it is used."
            )
        if model_source == "Local checkpoint":
            if not checkpoint_path:
                st.error("Please provide a Cellpose checkpoint path.")
                st.stop()
            model_path = checkpoint_path
        else:
            model_path = None            

        try:
            with st.spinner("Loading Cellpose model..."):
                segmenter = load_cellpose(model_path)
                st.success("Cellpose model loaded.")
            with st.spinner("Segmenting cells..."):
                result = segmenter.segment(image_array,diameter=cell_diameter,)
                st.success("Cell segmentation complete.")
        except Exception as e:
            st.error(f"Cellpose failed: {e}")
            st.stop()

        masks = result.masks
        n_cells = int(masks.max())
        st.metric( "Detected Cells", n_cells,)  

        cell_overlay = label2rgb(
            masks,
            image=image_array,
            bg_label=0,
        )
        
        st.subheader("Cellpose Results")  
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                cell_overlay,
                caption="Cell Instance Overlay",
                width="stretch",
            )
        with col2:
            mask_display = (
                masks.astype(np.float32)
                / max(n_cells, 1)
                * 255
            ).astype(np.uint8)

            st.image(
                mask_display,
                caption="Cell Instance Mask",
                width="stretch",
            )
    

    # --------------------------------------------------
    # Histograms
    # --------------------------------------------------
    
    st.subheader("DAB Intensity Distribution")
          
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(dab.ravel(), bins=100,)
    ax.set_xlabel("DAB optical density / concentration")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of DAB Concentration")
    ax.grid(alpha=0.2)
    """ax.axvline(
        threshold,
        color="red",
        linewidth=2,
        label="Threshold",
    )"""
    st.pyplot(fig,width="stretch")
    plt.close(fig)
    
    st.subheader("Hematoxylin Intensity Distribution")
         
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(hem.ravel(), bins=100,)
    ax.set_xlabel("Hematoxylin optical density / concentration")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of Hematoxylin Concentration")
    ax.grid(alpha=0.2)
    """ax.axvline(
            threshold,
            color="red",
            linewidth=2,
            label="Threshold",
        )"""
    st.pyplot(fig,width="stretch")
    plt.close(fig)
    
    st.subheader("Eosin Intensity Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(eos.ravel(), bins=100,)
    ax.set_xlabel("Eosin optical density / concentration")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of Eosin Concentration")
    ax.grid(alpha=0.2)
    """ax.axvline(
            threshold,
            color="red",
            linewidth=2,
            label="Threshold",
        )"""
    st.pyplot(fig,width="stretch")
    plt.close(fig)
    
    st.divider()