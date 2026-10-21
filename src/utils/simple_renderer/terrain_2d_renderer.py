from typing import override

import numpy as np
import pygame
from pygame import Surface

from rendering.renderable import Renderable, GeoToScreenFunction
from utils.terrain import Terrain


class Terrain2dRenderer(Renderable):
    def __init__(self, data: Terrain, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terrain: Terrain = data

    @override
    def render(self, screen: Surface, geo_to_screen: GeoToScreenFunction, **kwargs) -> None:
        if self.terrain.elevation_grid is None:
            return

        grid = self.terrain.elevation_grid

        # 1. normalize height values 0 - 255
        norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-6)
        img = (norm * 255).astype(np.uint8)

        # 2. grey values → RGB
        rgb = np.stack([img] * 3, axis=-1)

        # 3. numpy → pygame surface
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

        # 4. Geo → Screen Bounding Box
        x_min, y_min = geo_to_screen(self.terrain.lat_min, self.terrain.lon_min)
        x_max, y_max = geo_to_screen(self.terrain.lat_max, self.terrain.lon_max)

        top = min(y_min, y_max)
        bottom = max(y_min, y_max)
        left = min(x_min, x_max)
        right = max(x_min, x_max)

        width = int(right - left)
        height = int(bottom - top)

        if width <= 0 or height <= 0:
            return

        surface = pygame.transform.scale(surface, (width, height))

        screen.blit(surface, (left, top))
