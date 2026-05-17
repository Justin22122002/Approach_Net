import ctypes
import math

import numpy as np

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glDrawElements,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glVertexAttribPointer,
)

from utils.vec3 import Vec3


class Mesh:
    """
    Vertex layout: [x,y,z,  nx,ny,nz,  u,v]  → 8 floats per vertex
    Indices are unsigned ints.
    """
    def __init__(self, vertices: np.ndarray, indices: np.ndarray):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices  = np.array(indices,  dtype=np.uint32)
        self._vao = None
        self._vbo = None
        self._ebo = None

    # GPU upload
    def _upload(self):
        if self._vao is not None:
            return
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        self._ebo = glGenBuffers(1)

        glBindVertexArray(self._vao)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        stride = 8 * 4  # 8 floats × 4 bytes
        # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # normal
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        # uv
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

        glBindVertexArray(0)

    def draw(self):
        self._upload()
        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def destroy(self):
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
            glDeleteBuffers(1, [self._vbo])
            glDeleteBuffers(1, [self._ebo])
            self._vao = None

    # Built-in shapes
    @staticmethod
    def cube():
        """Unit cube centred at origin."""
        # fmt: x y z  nx ny nz  u v
        v = np.array([
            # Front  +Z
            -0.5,-0.5, 0.5,  0, 0, 1,  0,0,
             0.5,-0.5, 0.5,  0, 0, 1,  1,0,
             0.5, 0.5, 0.5,  0, 0, 1,  1,1,
            -0.5, 0.5, 0.5,  0, 0, 1,  0,1,
            # Back   -Z
             0.5,-0.5,-0.5,  0, 0,-1,  0,0,
            -0.5,-0.5,-0.5,  0, 0,-1,  1,0,
            -0.5, 0.5,-0.5,  0, 0,-1,  1,1,
             0.5, 0.5,-0.5,  0, 0,-1,  0,1,
            # Right  +X
             0.5,-0.5, 0.5,  1, 0, 0,  0,0,
             0.5,-0.5,-0.5,  1, 0, 0,  1,0,
             0.5, 0.5,-0.5,  1, 0, 0,  1,1,
             0.5, 0.5, 0.5,  1, 0, 0,  0,1,
            # Left   -X
            -0.5,-0.5,-0.5, -1, 0, 0,  0,0,
            -0.5,-0.5, 0.5, -1, 0, 0,  1,0,
            -0.5, 0.5, 0.5, -1, 0, 0,  1,1,
            -0.5, 0.5,-0.5, -1, 0, 0,  0,1,
            # Top    +Y
            -0.5, 0.5, 0.5,  0, 1, 0,  0,0,
             0.5, 0.5, 0.5,  0, 1, 0,  1,0,
             0.5, 0.5,-0.5,  0, 1, 0,  1,1,
            -0.5, 0.5,-0.5,  0, 1, 0,  0,1,
            # Bottom -Y
            -0.5,-0.5,-0.5,  0,-1, 0,  0,0,
             0.5,-0.5,-0.5,  0,-1, 0,  1,0,
             0.5,-0.5, 0.5,  0,-1, 0,  1,1,
            -0.5,-0.5, 0.5,  0,-1, 0,  0,1,
        ], dtype=np.float32)

        idx = []
        for f in range(6):
            b = f * 4
            idx += [b,b+1,b+2, b,b+2,b+3]
        return Mesh(v, np.array(idx, dtype=np.uint32))

    @staticmethod
    def plane(size=1.0, divisions=1):
        """Flat XZ plane centred at origin."""
        half = size / 2
        n = divisions
        verts = []
        for j in range(n+1):
            for i in range(n+1):
                x = -half + i * size / n
                z = -half + j * size / n
                u = i / n
                v = j / n
                verts += [x, 0.0, z,  0,1,0,  u,v]
        idx = []
        for j in range(n):
            for i in range(n):
                tl = j*(n+1)+i
                tr = tl+1
                bl = tl+(n+1)
                br = bl+1
                idx += [tl,bl,tr, tr,bl,br]
        return Mesh(np.array(verts, dtype=np.float32),
                    np.array(idx,   dtype=np.uint32))

    @staticmethod
    def sphere(radius=0.5, rings=16, sectors=16):
        """UV sphere."""
        verts = []
        idx   = []
        for r in range(rings+1):
            lat = math.pi/2 - r * math.pi / rings
            y   = radius * math.sin(lat)
            xz  = radius * math.cos(lat)
            for s in range(sectors+1):
                lon = s * 2*math.pi / sectors
                x   = xz * math.cos(lon)
                z   = xz * math.sin(lon)
                nx, ny, nz = x/radius, y/radius, z/radius
                u = s / sectors
                v = r / rings
                verts += [x, y, z,  nx, ny, nz,  u, v]
        for r in range(rings):
            for s in range(sectors):
                cur  = r*(sectors+1)+s
                nxt  = cur + sectors+1
                idx += [cur, nxt, cur+1, cur+1, nxt, nxt+1]
        return Mesh(np.array(verts, dtype=np.float32),
                    np.array(idx,   dtype=np.uint32))

    @staticmethod
    def from_obj(path: str) -> "Mesh":
        """Load a .obj file (positions + normals + UVs)."""
        pos, nrm, uvs = [], [], []
        faces = []
        with open(path) as f:
            for line in f:
                tok = line.split()
                if not tok: continue
                if tok[0] == "v":
                    pos.append([float(t) for t in tok[1:4]])
                elif tok[0] == "vn":
                    nrm.append([float(t) for t in tok[1:4]])
                elif tok[0] == "vt":
                    uvs.append([float(t) for t in tok[1:3]])
                elif tok[0] == "f":
                    face = []
                    for t in tok[1:]:
                        parts = t.split("/")
                        pi = int(parts[0])-1
                        ti = int(parts[1])-1 if len(parts)>1 and parts[1] else 0
                        ni = int(parts[2])-1 if len(parts)>2 and parts[2] else 0
                        face.append((pi,ti,ni))
                    # fan-triangulate
                    for i in range(1, len(face)-1):
                        faces.append((face[0], face[i], face[i+1]))

        verts = []
        indices = []
        cache: dict[tuple,int] = {}
        for tri in faces:
            for (pi,ti,ni) in tri:
                key = (pi,ti,ni)
                if key not in cache:
                    p = pos[pi]
                    n = nrm[ni] if nrm else [0,1,0]
                    u = uvs[ti] if uvs else [0,0]
                    verts += p + n + u
                    cache[key] = len(cache)
                indices.append(cache[key])
        return Mesh(np.array(verts, dtype=np.float32),
                    np.array(indices, dtype=np.uint32))
