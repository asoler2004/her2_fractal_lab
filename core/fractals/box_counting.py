"""
Box-counting fractal dimension analysis.

The module is intentionally independent of Streamlit.

It calculates:
    N(epsilon) = number of occupied boxes at scale epsilon

and estimates:

    N(epsilon) ~ epsilon^(-D)

therefore:

    log(N) = D * log(1/epsilon) + C

The returned result contains all intermediate information required
to visualize the box-counting process.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BoxCountingResult:
    """Complete result of a box-counting analysis."""

    fractal_dimension: float

    box_sizes: np.ndarray
    box_counts: np.ndarray

    log_inverse_box_sizes: np.ndarray
    log_box_counts: np.ndarray

    slope: float
    intercept: float
    r_squared: float


def _prepare_binary_mask(
    binary_mask: np.ndarray,
) -> np.ndarray:
    """
    Validate and normalize the binary mask.

    Parameters
    ----------
    binary_mask:
        2D binary image.

    Returns
    -------
    np.ndarray
        Boolean binary mask.
    """

    mask = np.asarray(binary_mask)

    if mask.ndim != 2:
        raise ValueError(
            f"binary_mask must be 2D. "
            f"Received shape {mask.shape}."
        )

    mask = mask > 0

    if not np.any(mask):
        raise ValueError(
            "The binary mask contains no foreground pixels."
        )

    return mask


def _pad_to_multiple(
    mask: np.ndarray,
    box_size: int,
) -> np.ndarray:
    """
    Pad image so that its dimensions are multiples of box_size.
    """

    height, width = mask.shape

    new_height = (
        int(np.ceil(height / box_size))
        * box_size
    )

    new_width = (
        int(np.ceil(width / box_size))
        * box_size
    )

    pad_height = new_height - height
    pad_width = new_width - width

    return np.pad(
        mask,
        (
            (0, pad_height),
            (0, pad_width),
        ),
        mode="constant",
        constant_values=False,
    )


def get_box_grid(
    binary_mask: np.ndarray,
    box_size: int,
):
    """
    Return the box occupancy grid for visualization.

    Each element corresponds to one box.

    True:
        box contains foreground pixels.

    False:
        box contains only background.
    """

    mask = _prepare_binary_mask(binary_mask)

    if box_size <= 0:
        raise ValueError(
            "box_size must be greater than zero."
        )

    padded = _pad_to_multiple(
        mask,
        box_size,
    )

    height, width = padded.shape

    reshaped = padded.reshape(
        height // box_size,
        box_size,
        width // box_size,
        box_size,
    )

    occupied = reshaped.any(axis=(1, 3))

    return occupied


def count_occupied_boxes(
    binary_mask: np.ndarray,
    box_size: int,
) -> int:
    """
    Count the number of boxes containing at least
    one foreground pixel.
    """

    occupied = get_box_grid(
        binary_mask,
        box_size,
    )

    return int(
        np.count_nonzero(occupied)
    )


def generate_box_sizes(
    shape: tuple[int, int],
    min_box_size: int = 2,
    max_box_size: Optional[int] = None,
    num_sizes: int = 10,
) -> np.ndarray:
    """
    Generate logarithmically distributed box sizes.
    """

    min_dim = min(shape)

    if max_box_size is None:
        max_box_size = min_dim // 2

    max_box_size = min(
        max_box_size,
        min_dim,
    )

    if min_box_size >= max_box_size:
        raise ValueError(
            "min_box_size must be smaller "
            "than max_box_size."
        )

    sizes = np.unique(
        np.round(
            np.logspace(
                np.log10(min_box_size),
                np.log10(max_box_size),
                num=num_sizes,
            )
        ).astype(int)
    )

    return sizes


def calculate_box_counting(
    binary_mask: np.ndarray,
    min_box_size: int = 2,
    max_box_size: Optional[int] = None,
    num_sizes: int = 10,
) -> BoxCountingResult:
    """
    Calculate the box-counting fractal dimension.

    Parameters
    ----------
    binary_mask:
        2D binary segmentation.

    min_box_size:
        Smallest box size.

    max_box_size:
        Largest box size.

    num_sizes:
        Number of scales.

    Returns
    -------
    BoxCountingResult
    """

    mask = _prepare_binary_mask(
        binary_mask
    )

    box_sizes = generate_box_sizes(
        mask.shape,
        min_box_size=min_box_size,
        max_box_size=max_box_size,
        num_sizes=num_sizes,
    )

    box_counts = np.array(
        [
            count_occupied_boxes(
                mask,
                int(size),
            )
            for size in box_sizes
        ],
        dtype=float,
    )

    valid = (
        (box_counts > 0)
        & np.isfinite(box_counts)
    )

    if np.count_nonzero(valid) < 2:
        raise ValueError(
            "Not enough valid scales for "
            "fractal-dimension estimation."
        )

    box_sizes = box_sizes[valid]
    box_counts = box_counts[valid]

    # -------------------------------------------------
    # Scaling variables
    # -------------------------------------------------

    # epsilon = box size
    #
    # N(epsilon) ~ epsilon^(-D)
    #
    # log(N) = D * log(1/epsilon) + C

    log_inverse_box_sizes = np.log(
        1.0 / box_sizes
    )

    log_box_counts = np.log(
        box_counts
    )

    # -------------------------------------------------
    # Linear regression
    # -------------------------------------------------

    slope, intercept = np.polyfit(
        log_inverse_box_sizes,
        log_box_counts,
        1,
    )

    predicted = (
        slope
        * log_inverse_box_sizes
        + intercept
    )

    residuals = (
        log_box_counts
        - predicted
    )

    ss_res = np.sum(
        residuals ** 2
    )

    ss_tot = np.sum(
        (
            log_box_counts
            - np.mean(log_box_counts)
        ) ** 2
    )

    if ss_tot > 0:
        r_squared = (
            1.0
            - ss_res / ss_tot
        )
    else:
        r_squared = np.nan

    fractal_dimension = float(
        slope
    )

    return BoxCountingResult(
        fractal_dimension=fractal_dimension,
        box_sizes=box_sizes,
        box_counts=box_counts,
        log_inverse_box_sizes=(
            log_inverse_box_sizes
        ),
        log_box_counts=log_box_counts,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
    )


def fractal_dimension(
    binary_mask: np.ndarray,
    **kwargs,
) -> float:
    """
    Convenience function returning only D.
    """

    result = calculate_box_counting(
        binary_mask,
        **kwargs,
    )

    return result.fractal_dimension