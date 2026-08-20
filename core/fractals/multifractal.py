"""
Robust multifractal analysis of a binary DAB mask.

Box-based partition function:

    Z(q, eps) = sum_i p_i(eps)^q

where:

    p_i(eps) = m_i(eps) / sum_j m_j(eps)

and m_i is the foreground mass inside box i.

Scaling relationship:

    Z(q, eps) ~ eps^tau(q)

Therefore:

    tau(q) = d log Z(q, eps) / d log eps

Generalized dimensions:

    D_q = tau(q) / (q - 1), q != 1

    D_1 = lim_{eps->0}
          sum_i p_i log(p_i) / log(eps)

Multifractal spectrum:

    alpha(q) = d tau(q) / dq

    f(alpha) = q * alpha - tau(q)

The implementation includes:

- dyadic box sizes
- multiple grid origins
- scaling-quality R²
- optional automatic scaling-region selection
- robust handling of q < 0
- smoothing of tau(q) before numerical differentiation
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------

@dataclass
class MultifractalResult:
    q_values: np.ndarray
    box_sizes: np.ndarray

    tau_q: np.ndarray
    tau_r2: np.ndarray

    generalized_dimensions: np.ndarray

    alpha: np.ndarray
    f_alpha: np.ndarray

    singularity_width: float

    alpha_min: float
    alpha_max: float

    f_max: float
    alpha_at_fmax: float

    d0: float
    d1: float
    d2: float

    f_alpha0: float

    scaling_start: int
    scaling_end: int


# ---------------------------------------------------------------------
# Mask preparation
# ---------------------------------------------------------------------

def _prepare_binary_mask(
    binary_mask: np.ndarray,
) -> np.ndarray:
    """
    Convert input to a 2D boolean mask.
    """

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


# ---------------------------------------------------------------------
# Box sizes
# ---------------------------------------------------------------------

def _generate_box_sizes(
    shape: tuple[int, int],
    min_box_size: int = 4,
    max_box_size: Optional[int] = None,
) -> np.ndarray:
    """
    Generate dyadic box sizes.

    Example for a 512x512 image:

        4, 8, 16, 32, 64, 128, 256

    Only box sizes that fit completely inside the image are used.

    This avoids artificial zero-padding.
    """

    height, width = shape

    max_possible = min(height, width)

    if max_box_size is None:
        # Avoid using a box size equal to the whole image.
        max_possible = max_possible // 2
    else:
        max_possible = min(
            int(max_box_size),
            max_possible // 2,
        )

    min_box_size = int(min_box_size)

    if min_box_size < 1:
        raise ValueError(
            "min_box_size must be >= 1."
        )

    if min_box_size > max_possible:
        raise ValueError(
            "min_box_size is larger than the maximum "
            "usable box size."
        )

    # First power of two >= min_box_size
    first_power = int(
        2 ** np.ceil(np.log2(min_box_size))
    )

    sizes = []

    size = first_power

    while size <= max_possible:
        sizes.append(size)
        size *= 2

    if len(sizes) < 3:
        raise ValueError(
            "At least three box sizes are required "
            "for multifractal scaling analysis."
        )

    return np.asarray(sizes, dtype=int)


# ---------------------------------------------------------------------
# Box masses
# ---------------------------------------------------------------------

def _box_masses(
    mask: np.ndarray,
    box_size: int,
    offset_y: int = 0,
    offset_x: int = 0,
) -> np.ndarray:
    """
    Calculate foreground masses for non-overlapping boxes.

    Unlike the previous implementation, this function does NOT pad
    the image.

    Pixels outside the valid image area are ignored.

    Parameters
    ----------
    mask:
        Binary 2D image.

    box_size:
        Size of square box.

    offset_y, offset_x:
        Grid origin offsets.
    """

    height, width = mask.shape

    y0 = int(offset_y)
    x0 = int(offset_x)

    if y0 >= height or x0 >= width:
        return np.array([], dtype=float)

    cropped = mask[
        y0:height,
        x0:width,
    ]

    h, w = cropped.shape

    # Only use complete boxes.
    usable_h = (h // box_size) * box_size
    usable_w = (w // box_size) * box_size

    if usable_h == 0 or usable_w == 0:
        return np.array([], dtype=float)

    cropped = cropped[
        :usable_h,
        :usable_w,
    ]

    reshaped = cropped.reshape(
        usable_h // box_size,
        box_size,
        usable_w // box_size,
        box_size,
    )

    masses = reshaped.sum(
        axis=(1, 3)
    )

    return masses.ravel().astype(float)


# ---------------------------------------------------------------------
# Grid offsets
# ---------------------------------------------------------------------

def _get_grid_offsets(
    box_size: int,
    use_multiple_origins: bool = True,
) -> list[tuple[int, int]]:
    """
    Generate multiple grid origins.

    Four origins are used:

        (0, 0)
        (eps/2, 0)
        (0, eps/2)
        (eps/2, eps/2)

    This reduces sensitivity to the arbitrary image origin.
    """

    if not use_multiple_origins:
        return [(0, 0)]

    shift = box_size // 2

    return [
        (0, 0),
        (shift, 0),
        (0, shift),
        (shift, shift),
    ]


# ---------------------------------------------------------------------
# Partition function
# ---------------------------------------------------------------------

def _partition_function(
    masses: np.ndarray,
    q: float,
) -> float:
    """
    Calculate Z(q, epsilon).

    Only non-empty boxes are considered.

    q = 0:
        Z = number of non-empty boxes

    q != 0:
        Z = sum(p_i ** q)
    """

    masses = np.asarray(
        masses,
        dtype=float,
    )

    masses = masses[
        masses > 0
    ]

    if masses.size == 0:
        return np.nan

    total_mass = masses.sum()

    if total_mass <= 0:
        return np.nan

    probabilities = masses / total_mass

    if np.isclose(q, 0.0):
        return float(
            probabilities.size
        )

    # For negative q, extremely small probabilities can
    # produce numerical overflow.
    if q < 0:

        probabilities = np.maximum(
            probabilities,
            np.finfo(float).tiny,
        )

    with np.errstate(
        divide="ignore",
        over="ignore",
        invalid="ignore",
    ):
        z = np.sum(
            probabilities ** q
        )

    if not np.isfinite(z) or z <= 0:
        return np.nan

    return float(z)


# ---------------------------------------------------------------------
# Average partition function over grid origins
# ---------------------------------------------------------------------

def _partition_function_for_scale(
    mask: np.ndarray,
    box_size: int,
    q: float,
    use_multiple_origins: bool,
) -> float:
    """
    Calculate the partition function for one scale.

    Multiple grid origins are averaged in log-space.

    Averaging log(Z) is preferable here because multifractal
    scaling is performed in log-space.
    """

    offsets = _get_grid_offsets(
        box_size,
        use_multiple_origins,
    )

    log_z_values = []

    for offset_y, offset_x in offsets:

        masses = _box_masses(
            mask,
            box_size,
            offset_y,
            offset_x,
        )

        z = _partition_function(
            masses,
            q,
        )

        if np.isfinite(z) and z > 0:
            log_z_values.append(
                np.log(z)
            )

    if not log_z_values:
        return np.nan

    return float(
        np.mean(log_z_values)
    )


# ---------------------------------------------------------------------
# Linear regression with R²
# ---------------------------------------------------------------------

def _linear_fit(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """
    Linear regression:

        y = slope*x + intercept

    Returns:

        slope
        intercept
        R²
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if x.size < 2:
        return np.nan, np.nan, np.nan

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    predicted = (
        slope * x
        + intercept
    )

    ss_res = np.sum(
        (y - predicted) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if np.isclose(ss_tot, 0):
        r2 = 1.0
    else:
        r2 = 1.0 - (
            ss_res / ss_tot
        )

    return (
        float(slope),
        float(intercept),
        float(r2),
    )


# ---------------------------------------------------------------------
# Scaling region
# ---------------------------------------------------------------------

def _select_scaling_region(
    log_eps: np.ndarray,
    log_z: np.ndarray,
    min_points: int = 4,
    min_r2: float = 0.95,
) -> tuple[int, int]:
    """
    Select a contiguous scaling region.

    The region is chosen as the longest/high-quality interval
    with R² >= min_r2.

    If no region reaches min_r2, the region with the best R²
    is selected.
    """

    n = len(log_eps)

    if n < min_points:
        return 0, n

    candidates = []

    for start in range(
        0,
        n - min_points + 1,
    ):

        for end in range(
            start + min_points,
            n + 1,
        ):

            x = log_eps[start:end]
            y = log_z[start:end]

            slope, intercept, r2 = _linear_fit(
                x,
                y,
            )

            if np.isfinite(r2):

                candidates.append(
                    (
                        start,
                        end,
                        r2,
                    )
                )

    if not candidates:
        return 0, n

    # Prefer regions satisfying the R² threshold.
    good = [
        c for c in candidates
        if c[2] >= min_r2
    ]

    if good:

        # Longest region first, then highest R².
        good.sort(
            key=lambda c: (
                -(c[1] - c[0]),
                -c[2],
            )
        )

        return good[0][0], good[0][1]

    # Otherwise use the best R² region.
    candidates.sort(
        key=lambda c: (
            -c[2],
            -(c[1] - c[0]),
        )
    )

    return (
        candidates[0][0],
        candidates[0][1],
    )


# ---------------------------------------------------------------------
# D1
# ---------------------------------------------------------------------

def _calculate_d1(
    mask: np.ndarray,
    box_sizes: np.ndarray,
    scaling_start: int,
    scaling_end: int,
    use_multiple_origins: bool,
) -> float:
    """
    Calculate information dimension D1.

    D1 = - dS / d(log epsilon)

    where

        S = -sum(p_i log p_i)
    """

    log_eps = []
    entropies = []

    selected_sizes = box_sizes[
        scaling_start:scaling_end
    ]

    for size in selected_sizes:

        offsets = _get_grid_offsets(
            int(size),
            use_multiple_origins,
        )

        entropy_values = []

        for offset_y, offset_x in offsets:

            masses = _box_masses(
                mask,
                int(size),
                offset_y,
                offset_x,
            )

            masses = masses[
                masses > 0
            ]

            if masses.size == 0:
                continue

            probabilities = (
                masses / masses.sum()
            )

            entropy = -np.sum(
                probabilities
                * np.log(probabilities)
            )

            entropy_values.append(
                entropy
            )

        if entropy_values:

            log_eps.append(
                np.log(size)
            )

            entropies.append(
                np.mean(entropy_values)
            )

    if len(log_eps) < 2:
        return np.nan

    slope, _, _ = _linear_fit(
        np.asarray(log_eps),
        np.asarray(entropies),
    )

    return float(-slope)


# ---------------------------------------------------------------------
# Smooth tau(q)
# ---------------------------------------------------------------------

def _smooth_tau(
    q_values: np.ndarray,
    tau_values: np.ndarray,
    polynomial_order: int = 3,
) -> np.ndarray:
    """
    Smooth tau(q) using a low-order polynomial.

    This is intentionally conservative because alpha is obtained
    by differentiating tau(q).
    """

    valid = (
        np.isfinite(q_values)
        & np.isfinite(tau_values)
    )

    q = q_values[valid]
    tau = tau_values[valid]

    if q.size < polynomial_order + 1:
        return tau_values.copy()

    order = min(
        polynomial_order,
        q.size - 1,
    )

    coefficients = np.polyfit(
        q,
        tau,
        order,
    )

    smoothed = np.full_like(
        tau_values,
        np.nan,
        dtype=float,
    )

    smoothed[valid] = np.polyval(
        coefficients,
        q,
    )

    return smoothed


# ---------------------------------------------------------------------
# Main multifractal calculation
# ---------------------------------------------------------------------

def calculate_multifractal(
    binary_mask: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    min_box_size: int = 4,
    max_box_size: Optional[int] = None,
    min_scaling_points: int = 4,
    min_r2: float = 0.95,
    use_multiple_origins: bool = True,
    smooth_tau: bool = True,
    smoothing_order: int = 3,
) -> MultifractalResult:
    """
    Calculate multifractal properties of a binary DAB mask.

    Parameters
    ----------
    binary_mask:
        2D binary DAB segmentation.

    q_values:
        Moment orders.

        Recommended starting point:

            np.linspace(-3, 3, 25)

        Negative q values emphasize low-mass regions.

    min_box_size:
        Minimum box size.

    max_box_size:
        Maximum box size.

    min_scaling_points:
        Minimum number of scales used for regression.

    min_r2:
        Minimum R² for automatic scaling-region selection.

    use_multiple_origins:
        Average partition functions over multiple grid origins.

    smooth_tau:
        Smooth tau(q) before numerical differentiation.

    smoothing_order:
        Polynomial order used for smoothing tau(q).

    Returns
    -------
    MultifractalResult
    """

    # -------------------------------------------------------------
    # Prepare mask
    # -------------------------------------------------------------

    mask = _prepare_binary_mask(
        binary_mask
    )

    # -------------------------------------------------------------
    # q values
    # -------------------------------------------------------------

    if q_values is None:

        q_values = np.linspace(
            -3.0,
            3.0,
            25,
        )

    q_values = np.asarray(
        q_values,
        dtype=float,
    )

    if q_values.ndim != 1:
        raise ValueError(
            "q_values must be a 1D array."
        )

    if len(q_values) < 5:
        raise ValueError(
            "At least five q values are recommended."
        )

    # -------------------------------------------------------------
    # Box sizes
    # -------------------------------------------------------------

    box_sizes = _generate_box_sizes(
        mask.shape,
        min_box_size=min_box_size,
        max_box_size=max_box_size,
    )

    log_eps = np.log(
        box_sizes.astype(float)
    )

    # -------------------------------------------------------------
    # Partition functions
    # -------------------------------------------------------------

    log_zq = np.full(
        (
            len(q_values),
            len(box_sizes),
        ),
        np.nan,
        dtype=float,
    )

    for iq, q in enumerate(q_values):

        for ie, size in enumerate(box_sizes):

            log_zq[iq, ie] = (
                _partition_function_for_scale(
                    mask,
                    int(size),
                    float(q),
                    use_multiple_origins,
                )
            )

    # -------------------------------------------------------------
    # Tau(q)
    # -------------------------------------------------------------

    tau_q = np.full(
        len(q_values),
        np.nan,
        dtype=float,
    )

    tau_r2 = np.full(
        len(q_values),
        np.nan,
        dtype=float,
    )

    # First determine a common scaling region.
    #
    # q = 0 is generally a good reference because it corresponds
    # to box counting.
    # -------------------------------------------------------------

    q0_index = int(
        np.argmin(
            np.abs(q_values)
        )
    )

    q0_log_z = log_zq[
        q0_index
    ]

    valid_q0 = np.isfinite(
        q0_log_z
    )

    if np.count_nonzero(valid_q0) >= min_scaling_points:

        scaling_start, scaling_end = (
            _select_scaling_region(
                log_eps[valid_q0],
                q0_log_z[valid_q0],
                min_points=min_scaling_points,
                min_r2=min_r2,
            )
        )

        # Convert indices from the valid subset back to the
        # original box-size array.
        valid_indices = np.where(
            valid_q0
        )[0]

        scaling_start = int(
            valid_indices[scaling_start]
        )

        scaling_end = int(
            valid_indices[scaling_end - 1] + 1
        )

    else:

        scaling_start = 0
        scaling_end = len(box_sizes)

    # -------------------------------------------------------------
    # Fit tau(q)
    # -------------------------------------------------------------

    for iq, q in enumerate(q_values):

        values = log_zq[
            iq,
            scaling_start:scaling_end,
        ]

        eps_values = log_eps[
            scaling_start:scaling_end
        ]

        valid = np.isfinite(
            values
        )

        if np.count_nonzero(valid) < 2:
            continue

        slope, _, r2 = _linear_fit(
            eps_values[valid],
            values[valid],
        )

        tau_q[iq] = slope
        tau_r2[iq] = r2

    # -------------------------------------------------------------
    # Generalized dimensions
    # -------------------------------------------------------------

    generalized_dimensions = np.full(
        len(q_values),
        np.nan,
        dtype=float,
    )

    for i, q in enumerate(q_values):

        if not np.isfinite(
            tau_q[i]
        ):
            continue

        if np.isclose(
            q,
            1.0,
        ):

            generalized_dimensions[i] = (
                _calculate_d1(
                    mask,
                    box_sizes,
                    scaling_start,
                    scaling_end,
                    use_multiple_origins,
                )
            )

        else:

            generalized_dimensions[i] = (
                tau_q[i]
                / (q - 1.0)
            )

    # -------------------------------------------------------------
    # Smooth tau(q)
    # -------------------------------------------------------------

    if smooth_tau:

        tau_for_derivative = _smooth_tau(
            q_values,
            tau_q,
            polynomial_order=smoothing_order,
        )

    else:

        tau_for_derivative = tau_q.copy()

    # -------------------------------------------------------------
    # Multifractal spectrum
    # -------------------------------------------------------------

    valid_tau = (
        np.isfinite(q_values)
        & np.isfinite(
            tau_for_derivative
        )
    )

    q_valid = q_values[
        valid_tau
    ]

    tau_valid = tau_for_derivative[
        valid_tau
    ]

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

        alpha = np.array(
            [],
            dtype=float,
        )

        f_alpha = np.array(
            [],
            dtype=float,
        )

    # -------------------------------------------------------------
    # Spectrum metrics
    # -------------------------------------------------------------

    if alpha.size > 0:

        singularity_width = float(
            np.max(alpha)
            - np.min(alpha)
        )

        alpha_min = float(
            np.min(alpha)
        )

        alpha_max = float(
            np.max(alpha)
        )

        f_max_index = int(
            np.argmax(f_alpha)
        )

        f_max = float(
            f_alpha[f_max_index]
        )

        alpha_at_fmax = float(
            alpha[f_max_index]
        )

    else:

        singularity_width = np.nan
        alpha_min = np.nan
        alpha_max = np.nan
        f_max = np.nan
        alpha_at_fmax = np.nan

    # -------------------------------------------------------------
    # D0, D1, D2
    # -------------------------------------------------------------

    def _get_dimension(
        target_q: float,
    ) -> float:

        index = int(
            np.argmin(
                np.abs(
                    q_values
                    - target_q
                )
            )
        )

        if np.isclose(
            q_values[index],
            target_q,
            atol=1e-8,
        ):
            return float(
                generalized_dimensions[index]
            )

        return np.nan

    d0 = _get_dimension(0.0)
    d1 = _get_dimension(1.0)
    d2 = _get_dimension(2.0)

    # -------------------------------------------------------------
    # f(alpha_0) consistency check
    # -------------------------------------------------------------

    f_alpha0 = np.nan

    q0_spectrum_index = np.where(
        np.isclose(
            q_valid,
            0.0,
            atol=1e-8,
        )
    )[0]

    if len(q0_spectrum_index) > 0:

        index = int(
            q0_spectrum_index[0]
        )

        f_alpha0 = float(
            f_alpha[index]
        )

    # -------------------------------------------------------------
    # Result
    # -------------------------------------------------------------

    return MultifractalResult(

        q_values=q_values,

        box_sizes=box_sizes,

        tau_q=tau_q,

        tau_r2=tau_r2,

        generalized_dimensions=(
            generalized_dimensions
        ),

        alpha=alpha,

        f_alpha=f_alpha,

        singularity_width=(
            singularity_width
        ),

        alpha_min=alpha_min,

        alpha_max=alpha_max,

        f_max=f_max,

        alpha_at_fmax=(
            alpha_at_fmax
        ),

        d0=d0,

        d1=d1,

        d2=d2,

        f_alpha0=f_alpha0,

        scaling_start=scaling_start,

        scaling_end=scaling_end,
    )