import numpy as np
from typing import List, Dict, Callable
from assignment2_mdp.mdp.abstract_mdp import AbstractMDP
from assignment2_mdp.mdp.reward_function import RewardFunction
from assignment2_mdp.mdp.transition_function import TransitionFunction


class MDP(AbstractMDP):
    def __init__(
        self,
        states: List[int],
        actions: List[int],
        transition_function: TransitionFunction,
        reward_function: RewardFunction,
        discount_factor: float = 0.9
    ):
        """
        Initializes the Markov Decision Process (MDP).

        :param states: A list of states in the MDP.
        :param actions: A list of actions in the MDP.
        :param transition_function: A TransitionFunction object that provides transition probabilities.
        :param reward_function: A RewardFunction object that provides rewards.
        :param discount_factor: A discount factor for future rewards.
        """
        self._states = states
        self._actions = actions
        self.transition_function = transition_function
        self.reward_function = reward_function
        self._discount_factor = discount_factor

    def transition_prob(self, state: int, action: int, new_state: int) -> float:
        """
        Returns the transition probabilities for a given state and action by calling the transition function.

        :param state: Current state
        :param action: Action taken
        :param new_state: New state

        :return: A NumPy array of transition probabilities to the next states.
        """
        return self.transition_function(state, action)[new_state]

    def reward(self, state: int, action: int) -> float:
        """
        Returns the reward for a given state and action by calling the reward function.

        :param state: Current state
        :param action: Action taken

        :return: A float representing the reward for the given (state, action) pair.
        """
        return self.reward_function(state, action)

    @property
    def states(self) -> List[int]:
        """
        Returns the list of states.

        :return: A list of states.
        """
        return self._states

    @property
    def actions(self) -> List[int]:
        """
        Returns the list of actions.

        :return: A list of actions.
        """
        return self._actions

    @property
    def discount_factor(self) -> float:
        """
        Returns the discount factor.

        :return: The discount factor.
        """
        return self._discount_factor
