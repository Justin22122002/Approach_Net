from abc import ABCMeta, abstractmethod
from collections.abc import Callable

import pygame
from pygame import Surface

type GeoToScreenFunction = Callable[[float, float], tuple[float, float]]

class Renderable(pygame.sprite.Sprite, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    def update(
        self,
        geo_to_screen: GeoToScreenFunction,
    ) -> None:
        """
        Calculates the screen coordinates (optional)
        """
        pass

    @abstractmethod
    def render(
            self,
            screen: Surface,
            geo_to_screen: GeoToScreenFunction,
            **kwargs,
    ) -> None:
        """
        every class needs to implement this method for rendering
        :param screen: pygame.Surface
        :param geo_to_screen: function for converting (lat, lon) -> (x, y)
        """
        raise NotImplementedError


class JsonSerializable(metaclass=ABCMeta):

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_dict[T : JsonSerializable](cls: type[T], data: dict) -> T:
        pass