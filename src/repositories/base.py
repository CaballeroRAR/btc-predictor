from abc import ABC, abstractmethod
from src.utils.logger import setup_logger

class BaseRepository(ABC):
    """
    Abstract Base Class for the Repository Pattern.
    Ensures a consistent interface for all persistence layers.
    """
    def __init__(self, name: str):
        self.logger = setup_logger(name)
        self.logger.info(f"Initializing {name}")

    @abstractmethod
    def save(self, data: dict, target: str):
        pass

    @abstractmethod
    def get(self, target: str):
        pass

    @abstractmethod
    def delete(self, target: str):
        pass
