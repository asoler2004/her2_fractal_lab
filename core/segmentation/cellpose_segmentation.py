from dataclasses import dataclass

import numpy as np
from cellpose import models


@dataclass
class CellposeResult:
    masks: np.ndarray
    flows: list
    styles: np.ndarray | None
    diameters: float | None


class CellposeSegmenter:

    def __init__(
        self,
        gpu: bool = False,
    ):
        """
        Cellpose 4.x segmenter.
        Uses the Cellpose-SAM model through CellposeModel.
        """
        self.model = models.CellposeModel(
            gpu=gpu,
        )

    def segment(
        self,
        image: np.ndarray,
        diameter: float | None = None,
    ) -> CellposeResult:

        masks, flows, styles = self.model.eval(
            image,
            diameter=diameter,
        )

        return CellposeResult(
            masks=masks,
            flows=flows,
            styles=styles,
            diameters=diameter,
        )
