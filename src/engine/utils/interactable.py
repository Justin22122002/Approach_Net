from typing import Callable, Optional

from typing_extensions import override

from engine.utils.component import Component
from engine.utils.gameObject import GameObject


class Interactable(Component):
    def __init__(
        self,
        on_hover_enter: Optional[Callable[["GameObject"], None]] = None,
        on_hover_exit:  Optional[Callable[["GameObject"], None]] = None,
        on_click:       Optional[Callable[["GameObject"], None]] = None,
    ) -> None:
        self.on_hover_enter: Optional[Callable[["GameObject"], None]] | None = on_hover_enter
        self.on_hover_exit:  Optional[Callable[["GameObject"], None]] | None = on_hover_exit
        self.on_click:       Optional[Callable[["GameObject"], None]] | None = on_click
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

    @override
    def on_update(self, dt: float):
        pass

    @override
    def on_start(self) -> None:
        pass

    @override
    def on_destroy(self) -> None:
        pass