from abc import ABC, abstractmethod
import numpy as np


class BaseRasterizer(ABC):
    """
    Base class for all rasterizers.

    Subclasses must implement the `render` method.
    """

    def __init__(self, width: int, height: int, background_color=(0.0, 0.0, 0.0)):
        self.width = width
        self.height = height
        self.background_color = np.array(background_color, dtype=np.float32)


    @abstractmethod
    def render(self, *args, **kwargs):
        raise NotImplementedError("Rasterizer must implement render()")
