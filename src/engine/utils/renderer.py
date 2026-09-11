from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Renderer(metaclass=ABCMeta):
    @abstractmethod
    def draw(
            self,
            view_mat: NDArray[np.float32],
            proj_mat: NDArray[np.float32],
            scene: "Scene"
    ) -> None:
        pass