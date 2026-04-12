from abc import ABC, abstractmethod
from src.utils.logger import setup_logger

class PredictionStrategy(ABC):
    """
    Abstract Strategy for different prediction algorithms.
    """
    def __init__(self, name: str):
        self.logger = setup_logger(name)

    @abstractmethod
    def predict(self, model, scaler, data, **kwargs):
        """
        Execute the prediction logic.
        """
        pass
