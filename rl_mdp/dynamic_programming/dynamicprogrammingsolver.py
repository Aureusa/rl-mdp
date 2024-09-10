import numpy as np
from rl_mdp.mdp.abstract_mdp import AbstractMDP


class DynamicProgrammingSolver:
    """"
    Implement this class
    """

    def __init__(self, mdp: AbstractMDP, theta: float = 1e-6):
        """
        Initializes the Dynamic Programming Solver for the given MDP.

        :param mdp: An instance of a class that implements AbstractMDP.
        :param theta: Convergence threshold for iterative methods.
        """
        self.mdp = mdp
        self.theta = theta

    def value_iteration(self) -> np.ndarray:
        """
        Performs value iteration to find the optimal policy.

        :return: A NumPy array representing the optimal policy for each state.
        """
        pass

    def policy_iteration(self) -> np.ndarray:
        """
        Performs policy iteration to find the optimal policy.

        :return: A NumPy array representing the optimal policy for each state.
        """
        pass
