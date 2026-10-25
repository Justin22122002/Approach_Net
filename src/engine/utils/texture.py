import numpy as np
from PIL import Image

from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_NEAREST,
    GL_REPEAT,
    GL_RGBA,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    glActiveTexture,
    glBindTexture,
    glDeleteTextures,
    glGenTextures,
    glGenerateMipmap,
    glTexImage2D,
    glTexParameteri,
)

class Texture:
    def __init__(self, path: str) -> None:
        self._id: int | None  = None
        self._path: str       = path
        self._filter_min: int = GL_LINEAR_MIPMAP_LINEAR
        self._filter_mag: int = GL_LINEAR
        self._repeat_x: bool  = GL_REPEAT
        self._repeat_y: bool  = GL_REPEAT

    def _upload(self) -> None:
        if self._id is not None:
            return

        # img  = Image.open(self._path).transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("RGBA")
        img  = Image.open(self._path).convert("RGBA")
        data = img.tobytes()
        self._id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,GL_REPEAT if self._repeat_x else GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,GL_REPEAT if self._repeat_y else GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, self._filter_min)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, self._filter_mag)
        glBindTexture(GL_TEXTURE_2D, 0)

    def bind(self, unit: int = 0) -> None:
        self._upload()
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self._id)

    def destroy(self) -> None:
        if self._id:
            glDeleteTextures(1, [self._id])
            self._id = None

    @staticmethod
    def solid_color(
        r: float = 1.0,
        g: float = 1.0,
        b: float = 1.0,
        a: float = 1.0,
    ) -> "Texture":
        return _SolidTexture(r, g, b, a)

    @staticmethod
    def from_array(rgb: np.ndarray) -> "Texture":
        return _ArrayTexture(rgb)


class _SolidTexture(Texture):
    def __init__(
        self,
        r: float,
        g: float,
        b: float,
        a: float
    ) -> None:
        super().__init__("")
        self._r: float = r
        self._g: float = g
        self._b: float = b
        self._a: float = a

    def _upload(self) -> None:
        if self._id is not None:
            return
        data = bytes([
            int(self._r * 255), int(self._g * 255),
            int(self._b * 255), int(self._a * 255),
        ])
        self._id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)


class _ArrayTexture(Texture):
    def __init__(self, rgb: np.ndarray) -> None:
        super().__init__("")
        self._img = Image.fromarray(rgb, mode="RGB").convert("RGBA")

    def _upload(self) -> None:
        if self._id is not None:
            return
        data = self._img.tobytes()
        self._id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self._img.width, self._img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT if self._repeat_x else GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT if self._repeat_y else GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, self._filter_min)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, self._filter_mag)
        glBindTexture(GL_TEXTURE_2D, 0)