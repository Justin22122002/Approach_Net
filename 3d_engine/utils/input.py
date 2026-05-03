import pygame


class Input:
    def __init__(self):
        self._keys_down: set = set()
        self._keys_just: set = set()
        self._mouse_buttons: set = set()
        self._mouse_pos = (0, 0)
        self._mouse_delta = (0, 0)

    def _process_event(self, event):
        if event.type == pygame.KEYDOWN:
            self._keys_down.add(event.key)
            self._keys_just.add(event.key)
        if event.type == pygame.KEYUP:
            self._keys_down.discard(event.key)
        if event.type == pygame.MOUSEBUTTONDOWN:
            self._mouse_buttons.add(event.button)
        if event.type == pygame.MOUSEBUTTONUP:
            self._mouse_buttons.discard(event.button)
        if event.type == pygame.MOUSEMOTION:
            self._mouse_pos   = event.pos
            self._mouse_delta = event.rel

    def _end_frame(self):
        self._keys_just.clear()
        self._mouse_delta = (0, 0)

    def get_key(self, key) -> bool:        return key in self._keys_down
    def get_key_down(self, key) -> bool:   return key in self._keys_just
    def get_mouse_button(self, btn) -> bool: return btn in self._mouse_buttons
    @property
    def mouse_pos(self): return self._mouse_pos
    @property
    def mouse_delta(self): return self._mouse_delta