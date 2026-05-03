from dataclasses import dataclass


@dataclass
class Color:
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0
    def to_list(self): return [self.r, self.g, self.b, self.a]
    def to_rgb(self):  return [self.r, self.g, self.b]

    @staticmethod
    def white():  return Color(1,1,1,1)
    @staticmethod
    def black():  return Color(0,0,0,1)
    @staticmethod
    def red():    return Color(1,0,0,1)
    @staticmethod
    def green():  return Color(0,1,0,1)
    @staticmethod
    def blue():   return Color(0,0,1,1)