from typing import Optional

import numpy as np
from OpenGL.GL import glUseProgram
from numpy.typing import NDArray

from utils.component import Component
from utils.material import Material
from utils.mesh import Mesh

class MeshRenderer(Component):
    def __init__(
        self,
        mesh: Mesh,
        material: Optional[Material] = None
    ) -> None:
        self.mesh: Mesh = mesh
        self.material: Material = material or Material()

    def draw(
        self,
        view_mat: NDArray[np.float32],
        proj_mat: NDArray[np.float32],
        scene: "Scene"
    ) -> None:
        model = self.owner.transform.matrix()
        self.material.bind(model, view_mat, proj_mat, scene)
        self.mesh.draw()
        glUseProgram(0)

    def on_destroy(self) -> None:
        self.mesh.destroy()