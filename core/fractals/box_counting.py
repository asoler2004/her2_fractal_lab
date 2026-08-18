"""
Box-counting fractal dimension analysis.

Input:
    Binary 2D mask:
        0 = background
        1 = foreground / DAB-positive structure

Output:
    BoxCountingResult containing:
        - fractal dimension
        - box sizes
        - occupied box counts
        - log-log regression information
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BoxCountingResult:
    fractal_dimension: float
    box_sizes: np.ndarray
    box_counts: np.ndarray
    log_box_sizes: np.ndarray
    log_box_counts: np.ndarray
    r_squared: float
    slope: float
    intercept: float


def _prepare_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Validate and normalize the binary mask.
    """
    mask = np.asarray(binary_mask)

    if mask.ndim != 2:
        raise ValueError(
            f"binary_mask must be a 2D array, got shape {mask.shape}"
        )

    mask = mask > 0

    if not np.any(mask):
        raise ValueError("binary_mask contains no foreground pixels.")

    return mask


def _pad_to_multiple(
    mask: np.ndarray,
    box_size: int,
) -> np.ndarray:
    """
    Pad image so both dimensions are divisible by box_size.
    """
    height, width = mask.shape

    new_height = int(np.ceil(height / box_size) * box_size)
    new_width = int(np.ceil(width / box_size) * box_size)

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


def count_occupied_boxes(
    binary_mask: np.ndarray,
    box_size: int,
) -> int:
    """
    Count boxes containing at least one foreground pixel.
    """
    mask = _prepare_binary_mask(binary_mask)

    if box_size <= 0:
        raise ValueError("box_size must be greater than zero.")

    padded = _pad_to_multiple(mask, box_size)

    height, width = padded.shape

    reshaped = padded.reshape(
        height // box_size,
        box_size,
        width // box_size,
        box_size,
    )

    occupied = reshaped.any(axis=(1, 3))

    return int(np.count_nonzero(occupied))


def generate_box_sizes(
    shape: tuple[int, int],
    min_box_size: int = 2,
    max_box_size: Optional[int] = None,
    num_sizes: int = 12,
) -> np.ndarray:
    """
    Generate approximately logarithmically spaced box sizes.
    """
    min_dim = min(shape)

    if max_box_size is None:
        max_box_size = min_dim // 2

    max_box_size = min(max_box_size, min_dim)

    if min_box_size >= max_box_size:
        raise ValueError(
            "min_box_size must be smaller than max_box_size."
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
    num_sizes: int = 12,
    min_occupied_boxes: int = 1,
) -> BoxCountingResult:
    """
    Calculate fractal dimension using box counting.

    The relationship is:

        N(s) ~ s^(-D)

    therefore:

        log(N(s)) = -D log(s) + C

    The fractal dimension is the negative slope.
    """
    mask = _prepare_binary_mask(binary_mask)

    box_sizes = generate_box_sizes(
        mask.shape,
        min_box_size=min_box_size,
        max_box_size=max_box_size,
        num_sizes=num_sizes,
    )

    counts = []

    for size in box_sizes:
        count = count_occupied_boxes(mask, int(size))

        if count >= min_occupied_boxes:
            counts.append(count)
        else:
            counts.append(np.nan)

    box_sizes = np.asarray(box_sizes, dtype=float)
    box_counts = np.asarray(counts, dtype=float)

    valid = (
        np.isfinite(box_counts)
        & (box_counts > 0)
        & (box_sizes > 0)
    )

    if np.count_nonzero(valid) < 2:
        raise ValueError(
            "Not enough valid box sizes for fractal-dimension estimation."
        )

    log_box_sizes = np.log(box_sizes[valid])
    log_box_counts = np.log(box_counts[valid])

    slope, intercept = np.polyfit(
        log_box_sizes,
        log_box_counts,
        1,
    )

    predicted = slope * log_box_sizes + intercept

    ss_res = np.sum(
        (log_box_counts - predicted) ** 2
    )

    ss_tot = np.sum(
        (log_box_counts - np.mean(log_box_counts)) ** 2
    )

    if ss_tot > 0:
        r_squared = 1 - ss_res / ss_tot
    else:
        r_squared = np.nan

    fractal_dimension = -float(slope)

    return BoxCountingResult(
        fractal_dimension=fractal_dimension,
        box_sizes=box_sizes[valid],
        box_counts=box_counts[valid],
        log_box_sizes=log_box_sizes,
        log_box_counts=log_box_counts,
        r_squared=float(r_squared),
        slope=float(slope),
        intercept=float(intercept),
    )


def fractal_dimension(
    binary_mask: np.ndarray,
    **kwargs,
) -> float:
    """
    Convenience function returning only the fractal dimension.
    """
    result = calculate_box_counting(
        binary_mask,
        **kwargs,
    )

    return result.fractal_dimension