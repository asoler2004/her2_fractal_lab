from pathlib import Path
from PIL import Image

SUPPORTED_FORMATS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}

def load_image(path_or_buffer):
    return Image.open(path_or_buffer)