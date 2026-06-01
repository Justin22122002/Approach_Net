from dataclasses import dataclass


@dataclass
class Color:
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0
    
    def to_list(self) -> list[float]: return [self.r, self.g, self.b, self.a]
    def to_rgb(self)-> list[float]:  return [self.r, self.g, self.b]

    @staticmethod
    def white() -> "Color":  return Color(1,1,1,1)
    @staticmethod
    def black() -> "Color":  return Color(0,0,0,1)
    @staticmethod
    def red() -> "Color":    return Color(1,0,0,1)
    @staticmethod
    def green() -> "Color":  return Color(0,1,0,1)
    @staticmethod
    def blue() -> "Color":   return Color(0,0,1,1)
    @staticmethod
    def gray() -> "Color":   return Color(0.7, 0.7, 0.7, 1)