"""
Multifractal analysis of a binary DAB mask.

Uses the box-based partition function:

    Z(q, eps) = sum_i p_i(eps)^q

where p_i is the fraction of foreground mass
contained in box i.

The scaling relationship is:

    Z(q, eps) ~ eps^tau(q)

From tau(q), generalized dimensions Dq
and the multifractal spectrum f(alpha) are estimated.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MultifractalResult:
    q_values: np.ndarray
    box_sizes: np.ndarray

    tau_q: np.ndarray
    generalized_dimensions: np.ndarray

    alpha: np.ndarray
    f_alpha: np.ndarray

    singularity_width: float
    spectrum_width: float


def _prepare_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(binary_mask)

    if mask.ndim != 2:
        raise ValueError(
            f"binary_mask must be 2D, got shape {mask.shape}"
        )

    mask = mask > 0

    if not np.any(mask):
        raise ValueError(
            "binary_mask contains no foreground pixels."
        )

    return mask


def _box_masses(
    mask: np.ndarray,
    box_size: int,
) -> np.ndarray:
    """
    Calculate foreground mass of non-overlapping boxes.
    """
    height, width = mask.shape

    new_height = int(
        np.ceil(height / box_size) * box_size
    )
    new_width = int(
        np.ceil(width / box_size) * box_size
    )

    padded = np.pad(
        mask,
        (
            (0, new_height - height),
            (0, new_width - width),
        ),
        mode="constant",
    )

    reshaped = padded.reshape(
        new_height // box_size,
        box_size,
        new_width // box_size,
        box_size,
    )

    masses = reshaped.sum(axis=(1, 3))

    return masses.ravel()


def _generate_box_sizes(
    shape: tuple[int, int],
    min_box_size: int,
    max_box_size: Optional[int],
    num_sizes: int,
) -> np.ndarray:

    min_dim = min(shape)

    if max_box_size is None:
        max_box_size = min_dim // 2

    max_box_size = min(max_box_size, min_dim)

    if min_box_size >= max_box_size:
        raise ValueError(
            "min_box_size must be smaller than max_box_size."
        )

    return np.unique(
        np.round(
            np.logspace(
                np.log10(min_box_size),
                np.log10(max_box_size),
                num=num_sizes,
            )
        ).astype(int)
    )


def _partition_function(
    masses: np.ndarray,
    q: float,
) -> float:
    """
    Calculate Z(q, epsilon).

    q = 0:
        number of non-empty boxes

    q != 0:
        sum(p_i^q)
    """
    masses = masses[masses > 0]

    if masses.size == 0:
        return np.nan

    probabilities = masses / masses.sum()

    if q == 0:
        return float(len(probabilities))

    return float(np.sum(probabilities ** q))


def calculate_multifractal(
    binary_mask: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    min_box_size: int = 2,
    max_box_size: Optional[int] = None,
    num_sizes: int = 10,
) -> MultifractalResult:
    """
    Calculate multifractal properties of a binary mask.

    Parameters
    ----------
    binary_mask:
        2D binary DAB segmentation.

    q_values:
        Moment orders. Typical range:

            q = -5 ... +5

        excluding q=1 from direct Dq calculation.

    Returns
    -------
    MultifractalResult
    """
    mask = _prepare_binary_mask(binary_mask)

    if q_values is None:
        q_values = np.arange(
            -5,
            6,
            dtype=float,
        )

    q_values = np.asarray(q_values, dtype=float)

    box_sizes = _generate_box_sizes(
        mask.shape,
        min_box_size=min_box_size,
        max_box_size=max_box_size,
        num_sizes=num_sizes,
    )

    # -------------------------------------------------
    # Partition functions
    # -------------------------------------------------

    log_eps = []
    log_zq = {
        float(q): []
        for q in q_values
    }

    for size in box_sizes:

        masses = _box_masses(
            mask,
            int(size),
        )

        eps = float(size)

        log_eps.append(
            np.log(eps)
        )

        for q in q_values:

            z = _partition_function(
                masses,
                float(q),
            )

            if z > 0 and np.isfinite(z):
                log_zq[float(q)].append(
                    np.log(z)
                )
            else:
                log_zq[float(q)].append(
                    np.nan
                )

    log_eps = np.asarray(log_eps)

    tau_values = []

    # -------------------------------------------------
    # Estimate tau(q)
    # -------------------------------------------------

    for q in q_values:

        values = np.asarray(
            log_zq[float(q)]
        )

        valid = np.isfinite(values)

        if np.count_nonzero(valid) < 2:
            tau_values.append(np.nan)
            continue

        slope, _ = np.polyfit(
            log_eps[valid],
            values[valid],
            1,
        )

        tau_values.append(float(slope))

    tau_q = np.asarray(tau_values)

    # -------------------------------------------------
    # Generalized dimensions Dq
    # -------------------------------------------------

    generalized_dimensions = np.full_like(
        tau_q,
        np.nan,
        dtype=float,
    )

    for i, q in enumerate(q_values):

        if not np.isfinite(tau_q[i]):
            continue

        if np.isclose(q, 1.0):

            # Information dimension approximation.
            masses = []

            eps_values = []

            for size in box_sizes:

                box_mass = _box_masses(
                    mask,
                    int(size),
                )

                box_mass = box_mass[
                    box_mass > 0
                ]

                probabilities = (
                    box_mass /
                    box_mass.sum()
                )

                entropy = -np.sum(
                    probabilities *
                    np.log(probabilities)
                )

                masses.append(entropy)
                eps_values.append(np.log(size))

            if len(masses) >= 2:

                slope, _ = np.polyfit(
                    eps_values,
                    masses,
                    1,
                )

                generalized_dimensions[i] = -slope

        else:
            generalized_dimensions[i] = (
                tau_q[i] / (q - 1)
            )

    # -------------------------------------------------
    # Multifractal spectrum
    #
    # alpha = d tau / dq
    #
    # f(alpha) = q alpha - tau(q)
    # -------------------------------------------------

    valid_tau = np.isfinite(tau_q)

    q_valid = q_values[valid_tau]
    tau_valid = tau_q[valid_tau]

    if len(q_valid) >= 3:

        alpha = np.gradient(
            tau_valid,
            q_valid,
        )

        f_alpha = (
            q_valid * alpha
            - tau_valid
        )

    else:
        alpha = np.array([])
        f_alpha = np.array([])

    if alpha.size > 0:

        singularity_width = float(
            np.max(alpha) - np.min(alpha)
        )

        spectrum_width = float(
            np.max(f_alpha) - np.min(f_alpha)
        )

    else:

        singularity_width = np.nan
        spectrum_width = np.nan

    return MultifractalResult(
        q_values=q_values,
        box_sizes=box_sizes,
        tau_q=tau_q,
        generalized_dimensions=generalized_dimensions,
        alpha=alpha,
        f_alpha=f_alpha,
        singularity_width=singularity_width,
        spectrum_width=spectrum_width,
    )