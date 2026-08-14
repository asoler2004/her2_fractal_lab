import numpy as np
from skimage.color import rgb2hed, hed2rgb
from dataclasses import dataclass

@dataclass
class DeconvolutionResult:
    hematoxylin: np.ndarray
    dab: np.ndarray
    eosin: np.ndarray

    hed: np.ndarray


def _normalize_for_display(
    channel: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """
    Normalize a quantitative stain channel for visualization.

    Percentile normalization prevents a small number of extreme
    pixels from making the entire image appear dark.
    """

    low = np.percentile(channel, low_percentile)
    high = np.percentile(channel, high_percentile)

    if high <= low:
        return np.zeros_like(channel, dtype=np.uint8)

    normalized = (channel - low) / (high - low)

    normalized = np.clip(normalized, 0.0, 1.0)

    return (normalized * 255).astype(np.uint8)


def color_deconvolution(image: np.ndarray) -> DeconvolutionResult:
    """
    Perform Ruifrok–Johnston color deconvolution using
        scikit-image.    
        Parameters
        ----------
        image
            RGB uint8 image.    
        Returns
        -------
        DeconvolutionResult
    """
    hed = rgb2hed(image)
    rgb_channels = []
    
    for i in range(3):    
        isolated = np.zeros_like(hed)
        isolated[:, :, i] = hed[:, :, i]
        rgb = hed2rgb(isolated)
        rgb_channels.append((rgb * 255).astype(np.uint8) )
    
    return DeconvolutionResult(
            hematoxylin=rgb_channels[0],
            eosin=rgb_channels[1],
            dab=rgb_channels[2],
            hed=hed,
        )
    