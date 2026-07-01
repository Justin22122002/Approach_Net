from typing import Callable, Optional

from utils.component import Component


class Interactable(Component):
    def __init__(
        self,
        on_hover_enter: Optional[Callable[["GameObject"], None]] = None,
        on_hover_exit:  Optional[Callable[["GameObject"], None]] = None,
        on_click:       Optional[Callable[["GameObject"], None]] = None,
    ) -> None:
        self.on_hover_enter = on_hover_enter
        self.on_hover_exit  = on_hover_exit
        self.on_click       = on_click
        self.hovered: bool  = False

    def _set_hovered(self, value: bool) -> None:
        if value == self.hovered:
            return
        self.hovered = value
        cb = self.on_hover_enter if value else self.on_hover_exit
        if cb:
            cb(self.owner)

    def _handle_click(self) -> None:
        if self.on_click:
            self.on_click(self.owner)