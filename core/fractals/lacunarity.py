"""
Lacunarity analysis using the gliding-box method.

Input:
    Binary 2D mask.

Output:
    LacunarityResult containing lacunarity values across box sizes.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LacunarityResult:
    box_sizes: np.ndarray
    lacunarity: np.ndarray


def _prepare_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Validate and normalize the binary mask.
    """
    mask = np.asarray(binary_mask)

    if mask.ndim != 2:
        raise ValueError(
            f"binary_mask must be 2D, got shape {mask.shape}"
        )

    mask = mask > 0

    if not np.any(mask):
        raise ValueError("binary_mask contains no foreground pixels.")

    return mask


def _integral_image(mask: np.ndarray) -> np.ndarray:
    """
    Create integral image for fast rectangular sums.
    """
    return np.pad(
        mask.astype(np.float64).cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    )


def _gliding_box_masses(
    integral: np.ndarray,
    box_size: int,
) -> np.ndarray:
    """
    Calculate foreground pixel count for every possible
    gliding box position.
    """
    if box_size <= 0:
        raise ValueError("box_size must be positive.")

    height = integral.shape[0] - 1
    width = integral.shape[1] - 1

    if box_size > height or box_size > width:
        return np.array([], dtype=float)

    masses = (
        integral[box_size:, box_size:]
        - integral[:-box_size, box_size:]
        - integral[box_size:, :-box_size]
        + integral[:-box_size, :-box_size]
    )

    return masses.ravel()


def calculate_lacunarity(
    binary_mask: np.ndarray,
    box_sizes: Optional[np.ndarray] = None,
    min_box_size: int = 2,
    max_box_size: Optional[int] = None,
    num_sizes: int = 10,
) -> LacunarityResult:
    """
    Calculate lacunarity using the gliding-box method.

    Lacunarity is calculated as:

        Λ = E[M²] / E[M]²

    where M is the foreground mass contained in
    each gliding box.

    Higher lacunarity generally indicates greater
    spatial heterogeneity / larger gaps.
    """
    mask = _prepare_binary_mask(binary_mask)

    height, width = mask.shape
    min_dim = min(height, width)

    if box_sizes is None:
        if max_box_size is None:
            max_box_size = min_dim // 4

        max_box_size = min(max_box_size, min_dim)

        if min_box_size >= max_box_size:
            raise ValueError(
                "Invalid box-size range."
            )

        box_sizes = np.unique(
            np.round(
                np.logspace(
                    np.log10(min_box_size),
                    np.log10(max_box_size),
                    num=num_sizes,
                )
            ).astype(int)
        )

    box_sizes = np.asarray(box_sizes, dtype=int)

    integral = _integral_image(mask)

    lacunarity_values = []
    valid_sizes = []

    for size in box_sizes:

        masses = _gliding_box_masses(
            integral,
            int(size),
        )

        if masses.size == 0:
            continue

        mean_mass = np.mean(masses)

        if mean_mass <= 0:
            continue

        second_moment = np.mean(masses ** 2)

        lac = second_moment / (mean_mass ** 2)

        valid_sizes.append(size)
        lacunarity_values.append(lac)

    if not lacunarity_values:
        raise ValueError(
            "Unable to calculate lacunarity for the supplied mask."
        )

    return LacunarityResult(
        box_sizes=np.asarray(valid_sizes),
        lacunarity=np.asarray(lacunarity_values),
    )


def lacunarity(
    binary_mask: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Convenience function returning lacunarity values.
    """
    result = calculate_lacunarity(
        binary_mask,
        **kwargs,
    )

    return result.lacunarity