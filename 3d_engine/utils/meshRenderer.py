from typing import Optional

from OpenGL.GL import glUseProgram

from utils.component import Component
from utils.material import Material
from utils.mesh import Mesh

class MeshRenderer(Component):
    def __init__(self, mesh: Mesh, material: Optional[Material] = None):
        self.mesh     = mesh
        self.material = material or Material()

    def draw(self, view_mat, proj_mat, scene):
        model = self.owner.transform.matrix()
        self.material.bind(model, view_mat, proj_mat, scene)
        self.mesh.draw()
        glUseProgram(0)

    def on_destroy(self):
        self.mesh.destroy()