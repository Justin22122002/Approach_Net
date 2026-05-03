
class Component:
    """Base class for all components attached to a GameObject."""
    owner: "GameObject" = None  # set when added

    def on_start(self): pass
    def on_update(self, dt: float): pass
    def on_destroy(self): pass