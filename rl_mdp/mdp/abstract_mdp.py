from abc import ABC, abstractmethod
from typing import List
import numpy as np


class AbstractMDP(ABC):
    """
    Simple MDP abstract base class.
    Assumes: discrete state and action space (represented as integers).
    """

    @abstractmethod
    def transition_prob(self, new_state: int, state: int, action: int) -> float | np.ndarray:
        """
        An MDP should have a transition function. In this case modeled as p(s'|s,a).
        :param new_state:
        :param state:
        :param action:
        :return: Probability p(s'|s,a).
        """
        pass

    @abstractmethod
    def reward(self, state: int, action: int) -> float:
        """
        An MDP should have a reward function. In this case modeled as r(s,a).
        :param state:
        :param action:
        :return: a numerical reward value for taking action a in state s.
        """
        pass

    @property
    @abstractmethod
    def states(self) -> List[int]:
        """
        Getter for states.
        :return: The list of all possible states represented by a list of integers.
        """
        pass

    @property
    @abstractmethod
    def actions(self) -> List[int]:
        """
        Getter for actions.
        :return: The list of all possible states represented by a list of actions.
        """
        pass

    @property
    @abstractmethod
    def discount_factor(self) -> float:
        """
        Getter for discount factor (gamma).
        :return: The discount factor.
        """
        pass
