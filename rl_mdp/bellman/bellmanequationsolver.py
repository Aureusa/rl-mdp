import numpy as np
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.policy.abstract_policy import AbstractPolicy


class BellmanEquationSolver:
    """"
    Implement this class
    """

    def __init__(self, mdp: AbstractMDP):
        """
        Initializes the Bellman Equation Solver for the given MDP.

        :param mdp: An instance of a class that implements AbstractMDP.
        """
        self.mdp = mdp

    def policy_evaluation(self, policy: AbstractPolicy) -> np.ndarray:
        """
        Evaluates the value function for a given policy.

        :param policy: An instance of the Policy class, which provides the action probabilities for each state.
        :return: A NumPy array representing the value function for the given policy.
        """
        pass
