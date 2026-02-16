import numpy as np
from PIL import Image
import math
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod


# ------------------------------------------------------------
# Base Class
# ------------------------------------------------------------

class EncoderAlgorithms(ABC):
    """
    Base class for all encoder algorithms.
    """

    @abstractmethod
    def render(self):
        """
        Must return rendered image tensor (H, W, 3) in [0,1].
        """
        pass
