import numpy as np

from skimage.filters import threshold_otsu
from skimage.morphology import (closing,opening,disk,remove_small_objects,)


def segment_membrane(
    dab: np.ndarray,
    threshold: float | None = None,
    min_size: int = 15,
    morphology_radius: int = 1,
) -> dict[str, np.ndarray | float]:
    """
    Segment DAB-positive regions from a quantitative
    DAB concentration image.

    Parameters
    ----------
    dab:
        2D quantitative DAB concentration image obtained
        from rgb2hed().

    threshold:
        DAB concentration threshold.

        If None, Otsu's method is used.

    min_size:
        Remove connected components smaller than this
        number of pixels.

    morphology_radius:
        Radius of the structuring element used for
        morphological cleanup.

    Returns
    -------
    dict
        Contains:

        threshold
        raw_mask
        cleaned_mask
    """

    if dab.ndim != 2:
        raise ValueError(
            "DAB image must be a 2D array."
        )

    dab = dab.astype(np.float32)

    # --------------------------------------------------
    # Threshold
    # --------------------------------------------------

    if threshold is None:
        threshold = threshold_otsu(dab)

    dab_binary_mask = (dab >= threshold).astype(np.uint8)

    # --------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------

    footprint = disk(morphology_radius)

    cleaned_mask = opening(
        dab_binary_mask,
        footprint,
    )

    cleaned_mask = closing(
        cleaned_mask,
        footprint,
    )

    # Remove small objects
    cleaned_mask = remove_small_objects(
        cleaned_mask,
        max_size=min_size,
    )

    print("Threshold:", threshold)
    print("dab min:", dab.min())
    print("dab max:", dab.max())
    print("Pixels >= threshold:", np.count_nonzero(dab >= threshold))
    print("Pixels <= threshold:", np.count_nonzero(dab <= threshold))

    return {
        "threshold": float(threshold),
        "dab_binary_mask": dab_binary_mask,
        "cleaned_mask": cleaned_mask,
    }
