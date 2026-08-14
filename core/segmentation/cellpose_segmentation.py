from dataclasses import dataclass
import streamlit as st
import numpy as np
from cellpose import models


@dataclass
class CellposeResult:
    masks: np.ndarray
    flows: list
    styles: np.ndarray
    diameters: float


class CellposeSegmenter:

    def __init__(
        self,
        model_type: str = "cyto3",
        gpu: bool = False,
    ):

        self.model = models.CellposeModel(
            gpu=gpu,
            model_type=model_type,
        )

    def segment(
        self,
        image: np.ndarray,
        diameter: float | None = None,
    ) -> CellposeResult:

        masks, flows, styles, diams = self.model.eval(
            image,
            diameter=diameter,
            channels=[0, 0],
        )

        return CellposeResult(
            masks=masks,
            flows=flows,
            styles=styles,
            diameters=diams,
        )



