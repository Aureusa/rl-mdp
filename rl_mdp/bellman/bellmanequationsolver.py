from assignment2_mdp.mdp.abstract_mdp import AbstractMDP


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
