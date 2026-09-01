import numpy as np
from OpenGL.GL import GL_FALSE, GL_TRUE, glDepthMask, glUseProgram
from numpy.typing import NDArray
from typing_extensions import override

from utils.component import Component
from utils.simplematerial import SimpleMaterial, Material
from utils.mesh import Mesh
from utils.renderer import Renderer


class MeshRenderer(Component, Renderer):
    def __init__(
        self,
        mesh: Mesh,
        material: Material | None = None
    ) -> None:
        self.mesh: Mesh = mesh
        self.material: Material = material or SimpleMaterial()

    @override
    def draw(
        self,
        view_mat: NDArray[np.float32],
        proj_mat: NDArray[np.float32],
        scene: "Scene"
    ) -> None:
        model = self.owner.transform.matrix()
        transparent = self.material.is_transparent
        if transparent:
            glDepthMask(GL_FALSE)
        self.material.bind(model, view_mat, proj_mat, scene)
        self.mesh.draw()
        glUseProgram(0)
        if transparent:
            glDepthMask(GL_TRUE)

    def on_destroy(self) -> None:
        self.mesh.destroy()