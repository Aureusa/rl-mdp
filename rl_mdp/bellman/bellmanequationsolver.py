import numpy as np
from mdp.abstract_mdp import AbstractMDP
from policy.abstract_policy import AbstractPolicy


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

        self._states = self.mdp.states
        self._actions = self.mdp.actions

    def policy_evaluation(self, policy: AbstractPolicy) -> np.ndarray:
        """
        Evaluates the value function for a given policy.

        :param policy: An instance of the Policy class, which provides the action probabilities for each state.
        :return: A NumPy array representing the value function for the given policy.
        """
        Reward_Vector = self._reward_vector_constructor(policy)

        P_matrix = self._P_matrix_constructor(policy)

        Inverted_Matrix = self._inverting_the_joined_matrix(P_matrix)

        state_vec = Inverted_Matrix @ Reward_Vector

        return state_vec


    def _reward_vector_constructor(self, policy: AbstractPolicy
                                  ) -> np.ndarray:
        Vector = []

        for state in self._states:
            element = 0
            policy_of_state = policy.action_dist[state]
            for count, action in enumerate(self._actions):
                reward = self.mdp.reward(state=state, action=action)
                element += policy_of_state[count] * reward
            Vector.append(element)

        return np.array(Vector)

    def _P_matrix_constructor(self, policy: AbstractPolicy
                              ) -> np.ndarray:
        Matrix = []

        number_of_states = len(self._states)

        for i in range(number_of_states):
            row = []
            for j in range(number_of_states):
                element = 0
                for action in self._actions:
                    trans_prob = self.mdp.transition_prob(new_state = i,
                                                          state = j,
                                                          action = action)
                    action_prob = policy.action_prob(state = j,
                                                     action = action)
                    element += trans_prob * action_prob
                row.append(element)
            Matrix.append(row)

        return np.array(Matrix)
    
    def _inverting_the_joined_matrix(self, P_matrix: np.ndarray):
        Identity_matrix = np.identity(P_matrix[0].size)

        Matrix = Identity_matrix - self.mdp.discount_factor * P_matrix
        
        try:
            Inverse = np.linalg.inv(Matrix)
        except:
            Inverse = np.linalg.pinv(Matrix)

        return Inverse
    