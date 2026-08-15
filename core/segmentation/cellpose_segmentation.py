from dataclasses import dataclass
from pathlib import Path

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
        model_path: str | None = None,
        gpu: bool = False,
    ):
        """
        Cellpose 4.x segmenter.

        Parameters
        ----------
        model_path:
            Optional path to a local Cellpose model/checkpoint.
            If None, Cellpose uses its default pretrained model.

        gpu:
            Whether to use GPU.
        """

        if model_path is not None:

            model_path = Path(model_path).expanduser()

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Cellpose model not found: {model_path}"
                )

            self.model = models.CellposeModel(
                pretrained_model=str(model_path),
                gpu=gpu,
            )

        else:

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

