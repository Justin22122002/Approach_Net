from PIL import Image

from OpenGL.GL import (
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
        self._id: int | None = None
        self._path: str      = path

    def _upload(self) -> None:
        if self._id is not None:
            return
        img  = Image.open(self._path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
        data = img.tobytes()
        self._id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
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
        a: float = 1.0
    ) -> "Texture":
        """Create a 1×1 pixel texture with the given colour."""
        t = _SolidTexture(r, g, b, a)
        return t


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
            int(self._b * 255), int(self._a * 255)
        ])
        self._id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)
