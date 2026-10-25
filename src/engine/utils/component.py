from abc import abstractmethod, ABCMeta


class Component(metaclass=ABCMeta):
    """Base class for all components attached to a GameObject."""
    owner: "GameObject" = None  # set when added

    @abstractmethod
    def on_start(self) -> None: pass

    @abstractmethod
    def on_update(self, dt: float): pass

    @abstractmethod
    def on_destroy(self) -> None: pass