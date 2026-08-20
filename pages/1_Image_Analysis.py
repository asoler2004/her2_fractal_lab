import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from core.io.loader import load_image
from core.io.loader import SUPPORTED_FORMATS
from core.preprocessing.color_deconvolution import (color_deconvolution)
from core.segmentation.membrane_segmentation import (
    segment_membrane,   
)   
from skimage.color import label2rgb
from cellpose import models
from core.fractals import (    
    calculate_lacunarity,
    calculate_multifractal,
)
from core.fractals.box_counting import (
    calculate_box_counting,
    get_box_grid,
)

@st.cache_resource
def load_cellpose(checkpoint_path):
    try:
        model = models.CellposeModel(gpu=False, pretrained_model=checkpoint_path)
        print("Cellpose model loaded successfully.")
        return model

    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

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

    dab = deconv.hed[:, :, 2]   # 2-D quantitative DAB concentration
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
    # Membrane Segmentation
    #---------------------------------------------
    st.divider()
    st.subheader("Membrane Segmentation")
    st.caption("Segmentation of HER2 membrane staining.")
    method = st.radio(
        "Segmentation Method",
        [
        "Otsu's Method",
        "Custom Threshold",
        ],
        horizontal=True,    
    )
    if method == "Otsu's Method":        
        threshold = None    
    else:
        threshold = st.slider(
            "DAB threshold",
            min_value=float(dab.min()),
            max_value=float(dab.max()),
            value = float(dab.mean()),
            step = 0.001,
        )
    memb_segm = segment_membrane(dab,threshold, min_size = 15, morphology_radius = 1)
    #memb_segm = {"threshold", "dab_binary_mask", "cleaned_mask"}
    binary_mask = memb_segm["dab_binary_mask"]
    st.caption(f"Otsu threshold: {memb_segm['threshold']:.4f}")
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            deconv.dab,
            caption="DAB channel",
            width="stretch"
        )
    with col2:
        st.image(
            memb_segm["cleaned_mask"] * 255,
            caption=f"DAB binary mask - Otsu threshold: {memb_segm['threshold']:.4f}",
            width="stretch"
        )    
    #---------------------------------------------
    # Cell Segmentation
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
            value="/home/antonia/.cellpose/models/cpsam_v2",
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
                masks, flows, styles = segmenter.eval(
                    image_array, 
                    channels=[0, 0],   # Adjust based on your channel mappings (e.g., [2, 3])
                    diameter=cell_diameter
                )
                # Convert Cellpose instance mask → binary mask
                binary_mask_cellpose = (masks > 0).astype(np.uint8)
                st.success("Cell segmentation complete.")
        except Exception as e:
            st.error(f"Cellpose failed: {e}")
            st.stop()

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
        
    st.divider()
    #--------------------------------------------------
    # Fractal Analysis
    #--------------------------------------------------
    st.subheader("Fractal Analysis")
    st.caption("Fractal analysis of the DAB channel.")
    with st.spinner("Calculating fractal metrics..."):
        if run_cellpose:
            box_result_cellpose = calculate_box_counting(binary_mask_cellpose, min_box_size=2, max_box_size=128, num_sizes=10)
            lac_result_cellpose = calculate_lacunarity(binary_mask_cellpose)           
            multifractal_result_cellpose = calculate_multifractal(binary_mask_cellpose)
            st.success("Fractal analysis cellpose complete.")
        box_result_threshold = calculate_box_counting(binary_mask, min_box_size=2, max_box_size=128, num_sizes=10)
        lac_result_threshold = calculate_lacunarity(binary_mask)
        # multifractal_result_threshold = calculate_multifractal(binary_mask)
        st.success("Fractal analysis threshold complete.")
        multifractal_result_threshold = calculate_multifractal(
            binary_mask,
            q_values=np.linspace(-3, 3, 25),
            min_box_size=4,
            max_box_size=None,
            min_scaling_points=4,
            min_r2=0.95,
            use_multiple_origins=True,
            smooth_tau=True,
            smoothing_order=3,
        )

        result = multifractal_result_threshold
        print("Box sizes:", result.box_sizes)
        print("Scaling region:",
            result.box_sizes[
                result.scaling_start:result.scaling_end
            ])
        print("tau(q):", result.tau_q)
        print("R²:", result.tau_r2)

        print("D0:", result.d0)
        print("D1:", result.d1)
        print("D2:", result.d2)

        print("Δα:", result.singularity_width)

        print("alpha:", result.alpha)
        print("f(alpha):", result.f_alpha)

        print("f(alpha0):", result.f_alpha0)

        st.subheader("Box Counting")
        if run_cellpose:
            st.write(f"Cellpose segmentation: {box_result_cellpose.fractal_dimension}")
            box_plot = st.empty()
            for box_size, box_count in zip(
                box_result_cellpose.box_sizes,
                box_result_cellpose.box_counts
            ):
                occupied = get_box_grid(
                    binary_mask_cellpose,
                    int(box_size)
                )
                fig,ax = plt.subplots(figsize=(8,8))
                ax.imshow(binary_mask_cellpose, cmap='gray', interpolation='nearest')
                height, width = binary_mask_cellpose.shape
                for x in range(0, width, int(box_size)):
                    ax.axvline(x, linewidth=0.5)                
                for y in range(0,height, int(box_size)):
                    ax.axhline(y, linewidth=0.5)    
                ax.set_title(
                    f"Box size = {int(box_size)} px  | "
                    f"Occupied boxes = {int(box_count)}"
                )
                ax.set_xlabel("X (pixels)")
                ax.set_ylabel("Y (pixels)")
                box_plot.pyplot(fig, width='stretch')
                plt.close(fig)
                st.subheader("Log-Log Scaling and Fractal Dimension")
                x = box_result_cellpose.log_inverse_box_sizes
                y = box_result_cellpose.log_box_counts
                y_fit = (box_result_cellpose.slope*x+box_result_cellpose.intercept)
                fig, ax= plt.subplots(figsize=(9,6))
                ax.scatter(x,y,s=60, label="Observed box counts")
                ax.plot(x,y_fit,linewidth=2,label=(f"Linear fit D= {box_result_cellpose.fractal_dimension:.3f})"))
                ax.set_title("Log-Log plot of box counting")    
                ax.set_xlabel("log(1/box size)")
                ax.set_ylabel("log(box count)")
                ax.legend()
                ax.grid(alpha=0.2)

                st.pyplot(fig, width='stretch')
                plt.close(fig)


        st.write(f"Threshold segmentation: {box_result_threshold.fractal_dimension}")
        box_plot = st.empty()

        
        for box_size, box_count in zip(
            box_result_threshold.box_sizes,
            box_result_threshold.box_counts
        ):
            occupied = get_box_grid(
                binary_mask,
                int(box_size)
            )
            fig,ax = plt.subplots(figsize=(8,8))
            ax.imshow(binary_mask, cmap='gray', interpolation='nearest')
            height, width = binary_mask.shape
            for x in range(0, width, int(box_size)):
                ax.axvline(x, linewidth=0.5)                
            for y in range(0,height, int(box_size)):
                ax.axhline(y, linewidth=0.5)    
            ax.set_title(
                f"Box size = {int(box_size)} px  | "
                f"Occupied boxes = {int(box_count)}"
            )
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            box_plot.pyplot(fig, width='stretch')
            plt.close(fig)
            time.sleep(1)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Box-Counting Scaling")
            box_sizes = np.asarray(box_result_threshold.box_sizes)
            box_counts = np.asarray(box_result_threshold.box_counts)
            fig, ax= plt.subplots(figsize=(9,6))
            ax.scatter(
                box_sizes,
                box_counts,
                s=60,
                label="Observed box counts"
            )
            ax.plot(
                box_sizes,
                box_counts,
                linewidth=1.5,
                alpha=0.7
            )
            ax.set_title(
                "Box Counting"
            )
            ax.set_xlabel(
                "Box size"
            )
            ax.set_ylabel(
                "Occupied boxes"
            )
            ax.grid(
                alpha=0.2
            )

            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        with col2:    
            st.subheader("Log-Log Scaling and Fractal Dimension")
            x = box_result_threshold.log_inverse_box_sizes
            y = box_result_threshold.log_box_counts
            y_fit = (box_result_threshold.slope*x+box_result_threshold.intercept)
            fig, ax= plt.subplots(figsize=(9,6))
            ax.scatter(x,y,s=60, label="Observed box counts")
            ax.plot(x,y_fit,linewidth=2,label=(f"Linear fit D= {box_result_threshold.fractal_dimension:.3f})"))
            ax.set_title("Log-Log plot of box counting")    
            ax.set_xlabel("log(1/box size)")
            ax.set_ylabel("log(box count)")
            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig, width='stretch')
            plt.close(fig)


        st.subheader("Lacunarity")
        #st.subheader("Threshold segmentation")
        col1, col2 = st.columns(2)
        with col1:
                st.write(lac_result_threshold.lacunarity)
        with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(
                    lac_result_threshold.box_sizes,
                    lac_result_threshold.lacunarity,
                    marker="o"
                )
                ax.set_xlabel("Box size (pixels)")
                ax.set_ylabel("Lacunarity")
                ax.set_title("Lacunarity")
                st.pyplot(fig, width="stretch")
                plt.close(fig)      
        

        st.write("Lacunarity quantifies the distribution of gaps and solid regions in a pattern.")  
        
        st.subheader("Multifractal")
        if run_cellpose:
            data_cellpose = {
                "q_values": multifractal_result_cellpose.q_values,
                "box_sizes": multifractal_result_cellpose.box_sizes,
                "tau_q": multifractal_result_cellpose.tau_q,
                "generalized_dimensions": multifractal_result_cellpose.generalized_dimensions,
                "alpha": multifractal_result_cellpose.alpha, 
                "f_alpha": multifractal_result_cellpose.f_alpha    
            }
            data_cellpose(pd.DataFrame({key: pd.Series(value) for key, value in data_cellpose.items()}  ))
        data_threshold = {
            "q_values": multifractal_result_threshold.q_values,
            "box_sizes": multifractal_result_threshold.box_sizes,
            "tau_q": multifractal_result_threshold.tau_q,
            "generalized_dimensions": multifractal_result_threshold.generalized_dimensions,
            "alpha": multifractal_result_threshold.alpha, 
            "f_alpha": multifractal_result_threshold.f_alpha,
            "tau_r2": multifractal_result_threshold.tau_r2    
        }

        data_threshold = pd.DataFrame({key: pd.Series(value) for key, value in data_threshold.items()})

        st.subheader("Threshold segmentation")
        st.write(f"singularity_width: {multifractal_result_threshold.singularity_width}")
        # st.write(f"spectrum_width: {multifractal_result_threshold.spectrum_width}")


        st.write(f"d0: {multifractal_result_threshold.d0}")
        st.write(f"d1: {multifractal_result_threshold.d1}")
        st.write(f"d2: {multifractal_result_threshold.d2}")
        st.write(f"alpha_min: {multifractal_result_threshold.alpha_min}")
        st.write(f"alpha_max: {multifractal_result_threshold.alpha_max}")
        st.write(f"f_max: {multifractal_result_threshold.f_max}")
        st.write(f"alpha_at_fmax: {multifractal_result_threshold.alpha_at_fmax}")
        st.write(f"scaling_start: {multifractal_result_threshold.scaling_start}")
        st.write(f"scaling_end: {multifractal_result_threshold.scaling_end}")
        st.dataframe(data_threshold)

        col1, col2 = st.columns(2)
        with col2:
            st.subheader("Multifractal spectrum")
            alpha = np.asarray(multifractal_result_threshold.alpha)
            f_alpha = np.asarray(multifractal_result_threshold.f_alpha)
            order = np.argsort(alpha)
            alpha_sorted = alpha[order]
            f_alpha_sorted = f_alpha[order]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                alpha_sorted,
                f_alpha_sorted,
                # multifractal_result_threshold.alpha,
                # multifractal_result_threshold.f_alpha,
                marker="o"
            )
            ax.set_xlabel("α (Hölder exponent)")
            ax.set_ylabel("f(α)")
            ax.set_title("Multifractal Spectrum")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            # st.subheader("Cellpose segmentation")
            # if run_cellpose:                
            #     st.write(f"singularity_width: {multifractal_result_cellpose.singularity_width}")
            #     st.write(f"spectrum_width: {multifractal_result_cellpose.spectrum_width}")
            #     st.dataframe(data_cellpose)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                multifractal_result_threshold.q_values,
                multifractal_result_threshold.tau_q,
                marker="o"
            )
            ax.set_xlabel("q")
            ax.set_ylabel("τ(q)")
            ax.set_title("Mass Exponent τ(q)")
            st.pyplot(fig, width="stretch")
            plt.close(fig)
                 
        

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
    st.pyplot(fig,width="stretch")
    plt.close(fig)
    
    st.subheader("Hematoxylin Intensity Distribution")
         
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(hem.ravel(), bins=100,)
    ax.set_xlabel("Hematoxylin optical density / concentration")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of Hematoxylin Concentration")
    ax.grid(alpha=0.2)
    st.pyplot(fig,width="stretch")
    plt.close(fig)
    
    st.subheader("Eosin Intensity Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(eos.ravel(), bins=100,)
    ax.set_xlabel("Eosin optical density / concentration")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of Eosin Concentration")
    ax.grid(alpha=0.2)
    st.pyplot(fig,width="stretch")
    plt.close(fig)
    
