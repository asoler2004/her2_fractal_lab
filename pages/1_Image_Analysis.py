from unittest import result

from core.segmentation.cellpose_segmentation import CellposeSegmenter
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core.io.loader import load_image
from core.io.loader import SUPPORTED_FORMATS
from core.preprocessing.color_deconvolution import (color_deconvolution)
from core.segmentation.membrane_segmentation import (segment_membrane,)
from skimage.color import label2rgb

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

    @st.cache_resource
    def load_cellpose():
        return CellposeSegmenter(
            model_type="cyto3",
            gpu=False,
        )

    st.subheader("Membrane Segmentation")

    segmenter = load_cellpose()
    result = segmenter.segment(image_array, 48)

    overlay = label2rgb(
        result.masks,
        image=image_array,
        bg_label=0,
    )
    st.image(
        overlay,
        caption="Cellpose Segmentation",
        width="stretch",
    )
    n_cells = result.masks.max()

    st.metric(
        "Detected Cells",
        n_cells,
    )

    """
    st.caption(
        "Segmentation of DAB-positive regions using "
        "the quantitative DAB channel."
    )

    dab = deconv.hed[:, :, 2]
    hem = deconv.hed[:, :, 0]
    eos = deconv.hed[:, :, 1]

    st.write(
        "DAB statistics",
        dab.min(),
        dab.max(),
        dab.mean(),
    )

    threshold_method = st.radio(
        "Threshold method",
        ["Otsu", "Manual"],
        horizontal=True,
    )

    st.write("Threshold method:", threshold_method)

    if threshold_method == "Otsu":
        segmentation = segment_membrane(
            dab,
            threshold=None,
        )

        threshold = segmentation["threshold"]
        st.write("Threshold:", threshold)

    else:
        threshold = st.slider(
            "DAB threshold",
            min_value=float(dab.min()),
            max_value=float(dab.max()),
            value=float(np.median(dab)),
            step=0.01,
        )
        st.write("Threshold manual:", threshold)

   
    segmentation = segment_membrane(dab,threshold=threshold, )
    st.metric("DAB threshold", f"{threshold:.4f}",)

    """

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            segmentation["raw_mask"],
            caption="Raw DAB Mask",
            width="stretch",
        )
    with col2:
        st.image(
            segmentation["cleaned_mask"],
            caption="Cleaned Mask",
            width="stretch",
        )

    overlay = label2rgb(
        segmentation["cleaned_mask"],
        image_array,
        bg_label=0,
    )

    st.subheader("Segmentation Overlay")

    st.image(
        overlay,
        width="stretch",
    )

    """mask = segmentation["cleaned_mask"]
    total_pixels = mask.size
    positive_pixels = np.count_nonzero(mask)
    positive_fraction = (positive_pixels / total_pixels)

    col1, col2, col3 = st.columns(3)    

    with col1:
        st.metric(
            "Positive pixels",
            f"{positive_pixels:,}",
        )
    with col2:
        st.metric(
            "Total pixels",
            f"{total_pixels:,}",
        )
    with col3:
        st.metric(
            "Positive area",
            f"{positive_fraction * 100:.2f}%",
        )
    """
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
    ax.axvline(
        threshold,
        color="red",
        linewidth=2,
        label="Threshold",
    )
    st.pyplot(fig,width="stretch")
    plt.close(fig)
    
    st.subheader("Hematoxylin Intensity Distribution")
         
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(hem.ravel(), bins=100,)
    ax.set_xlabel("Hematoxylin optical density / concentration")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of Hematoxylin Concentration")
    ax.grid(alpha=0.2)
    ax.axvline(
            threshold,
            color="red",
            linewidth=2,
            label="Threshold",
        )
    st.pyplot(fig,width="stretch")
    plt.close(fig)
    
    st.subheader("Eosin Intensity Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(eos.ravel(), bins=100,)
    ax.set_xlabel("Eosin optical density / concentration")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of Eosin Concentration")
    ax.grid(alpha=0.2)
    ax.axvline(
            threshold,
            color="red",
            linewidth=2,
            label="Threshold",
        )
    st.pyplot(fig,width="stretch")
    plt.close(fig)
    
    st.divider()