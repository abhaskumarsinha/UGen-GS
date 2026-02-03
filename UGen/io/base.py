from abc import ABC, abstractmethod
from typing import Tuple, Any


class ImporterBase(ABC):
    """
    Base class for importing Gaussians and Cameras from any format.

    Contract:
      - input: a single string (path, folder, uri, etc.)
      - output: (gaussians, cameras)
    """

    def __init__(self, source: str):
        self.source = source

    @abstractmethod
    def load(self) -> Tuple[Any, Any]:
        """
        Returns:
            gaussians: list | dict | json-like structure
            cameras:   list | dict | json-like structure
        """
        pass
