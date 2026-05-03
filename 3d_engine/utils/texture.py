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
    def __init__(self, path: str):
        self._id = None
        self._path = path

    def _upload(self):
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

    def bind(self, unit=0):
        self._upload()
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self._id)

    def destroy(self):
        if self._id:
            glDeleteTextures(1, [self._id])
            self._id = None

    @staticmethod
    def solid_color(r=1.0, g=1.0, b=1.0, a=1.0) -> "Texture":
        """Create a 1×1 pixel texture with the given colour."""
        t = _SolidTexture(r, g, b, a)
        return t


class _SolidTexture(Texture):
    def __init__(self, r, g, b, a):
        super().__init__("")
        self._r, self._g, self._b, self._a = r, g, b, a

    def _upload(self):
        if self._id is not None:
            return
        data = bytes([
            int(self._r*255), int(self._g*255),
            int(self._b*255), int(self._a*255)
        ])
        self._id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)
