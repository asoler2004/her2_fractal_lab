from PIL import Image
import numpy as np
from cellpose import models
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATH_TO_TEST_IMAGE = ROOT / "examples" / "EZEHER 002 (5) Cero.jpg"


image = np.array(
    Image.open(PATH_TO_TEST_IMAGE).convert("RGB")
)

print("Image shape:", image.shape)

print("Loading model...")
model = models.CellposeModel(gpu=False)

print("Starting segmentation...")
start = time.time()

masks, flows, styles = model.eval(
    image,
    diameter=48,
)

print("Finished!")
print("Time:", time.time() - start)
print("Number of cells:", masks.max())