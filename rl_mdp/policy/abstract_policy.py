from abc import ABC, abstractmethod
from typing import List


class AbstractPolicy(ABC):
    """
    An abstract base class for a simple policy for discrete state-action spaces.
    """

    @abstractmethod
    def set_action_probabilities(self, state: int, action_probabilities: List[float]) -> None:
        pass

    @abstractmethod
    def action_prob(self, state: int, action: int) -> float:
        pass
