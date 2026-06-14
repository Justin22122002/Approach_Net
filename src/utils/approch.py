from dataclasses import dataclass
from typing import override

import pygame
from pygame import Surface

from src.rendering.renderable import GeoToScreenFunction, Renderable


@dataclass
class FixIdentifier:
    fix_identifier: str
    path_terminator: str # e.g -> IF, TF etc, note: this is not IF, IAF, FAF, or MAP ! -> look for Description_Code -> 4th position describes the type of the waypoint -> A -> IAF, B -> IF, C -> FAF, M -> MAP, etc
    latitude: float
    longitude: float
    altitude: float
    sequence_number: str

    course: float | None
    speed_kts: float | None
    turn_direction: str | None
    altitude_restriction: str | None


@dataclass
class Approach(Renderable):
    procedure_identifier: str
    route_type: str  # R, A, I
    fixes: list[FixIdentifier]

    def _split_into_segments(self) -> list[list[FixIdentifier]]:
        segments: list[list[FixIdentifier]] = []
        current: list[FixIdentifier] = []

        for fix in self.fixes:
            seq = float(fix.sequence_number) if str(fix.sequence_number).replace('.', '', 1).isdigit() else 0.0
            if current:
                prev_seq = float(current[-1].sequence_number) if str(current[-1].sequence_number).replace('.', '', 1).isdigit() else 0.0
                if seq <= prev_seq:
                    segments.append(current)
                    current = []
            current.append(fix)

        if current:
            segments.append(current)

        return segments

    def get_final_approach_fixes(self) -> tuple[list[FixIdentifier], FixIdentifier | None, FixIdentifier | None]:
        segments = self._split_into_segments()

        final_seg_idx, rw_idx = None, None
        for i, segment in enumerate(segments):
            indices = [j for j, f in enumerate(segment) if f.fix_identifier.startswith("RW")]
            if indices:
                final_seg_idx, rw_idx = i, indices[0]
                break

        if final_seg_idx is None:
            return [], None, None

        final_segment = segments[final_seg_idx]
        rw_fix = final_segment[rw_idx]
        fix_before = final_segment[rw_idx - 1] if rw_idx >= 1 else None

        entry_fixes: list[FixIdentifier] = []

        if rw_idx >= 2:
            entry_fixes.append(final_segment[rw_idx - 2])

        if fix_before:
            for i, segment in enumerate(segments):
                if i == final_seg_idx:
                    continue
                for j, f in enumerate(segment):
                    if f.fix_identifier == fix_before.fix_identifier and j >= 1:
                        entry_fixes.append(segment[j - 1])

        return entry_fixes, fix_before, rw_fix

    @override
    def render(self, screen: Surface, geo_to_screen: GeoToScreenFunction, **kwargs) -> None:
        color = (255, 0, 0) if self.route_type == "R" else (0, 255, 0) if self.route_type == "A" else (0, 0, 255)

        sorted_fixes = sorted(
            self.fixes,
            key=lambda f: float(f.sequence_number) if isinstance(f.sequence_number, (int, float, str)) and str(f.sequence_number).replace('.', '', 1).isdigit() else 0.0
        )

        font = pygame.font.SysFont("Arial", 12)
        for f in sorted_fixes:
            if f.latitude is not None and f.longitude is not None:
                x, y = geo_to_screen(f.latitude, f.longitude)
                pygame.draw.circle(screen, color, (x, y), 3)

                text_surf = font.render(f.fix_identifier, True, color)
                screen.blit(text_surf, (x + 5, y - 5))