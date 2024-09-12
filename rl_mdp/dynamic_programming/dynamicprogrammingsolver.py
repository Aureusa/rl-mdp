from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.policy.policy import Policy


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

    def value_iteration(self) -> Policy:
        """
        Performs value iteration to find the optimal policy.

        :return: An optimal policy.
        """
        pass

    def policy_iteration(self) -> Policy:
        """
        Performs policy iteration to find the optimal policy.

        :return: An optimal policy.
        """
        pass
