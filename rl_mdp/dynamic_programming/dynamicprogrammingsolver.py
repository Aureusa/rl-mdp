import numpy as np
from mdp.abstract_mdp import AbstractMDP
from policy.abstract_policy import AbstractPolicy
from policy.policy import Policy


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
        # Finding the optimal value function
        value_function = self._optimal_value_function()

        # Once value function converges, extract the policy
        policy = Policy()

        for state in self.mdp.states:
            action_values = []
            for action in self.mdp.actions:
                expected_value = self._value_calculater(state, action, value_function)
                action_values.append(expected_value)

            best_action = np.argmax(action_values)

            action_prob_array = np.zeros(len(self.mdp.actions))

            action_prob_array[best_action] = 1

            action_prob_list = action_prob_array.tolist()
            policy.set_action_probabilities(state, action_prob_list)
        
        return policy

    def policy_iteration(self) -> Policy:
        """
        Performs policy iteration to find the optimal policy.

        :return: An optimal policy.
        """
        # Initialize random policy
        policy = self._initialize_random_policy()
        
        for _ in range(1000):
            # Step 1: Policy Evaluation
            value_function = self.iterative_policy_evaluation(policy)
            
            # Step 2: Policy Improvement
            policy_stable = self.policy_improvement(policy, value_function)
            
            # Terminate if policy is stable
            if policy_stable:
                break
        
        return policy

    def iterative_policy_evaluation(self, policy: AbstractPolicy) -> np.ndarray:
        """
        Evaluates iteratively the value function for a given policy.

        :param policy: An instance of the Policy class, which provides the action probabilities for each state.
        :return: A NumPy array representing the value function for the given policy.
        """
        # Initialising a state which is all 0's
        state_values = np.zeros(len(self.mdp.states))
        
        for _ in range(1000):
            delta = 0

            # Looping over the states
            for state in self.mdp.states:
                v = state_values[state]
                new_value = 0
                for action in self.mdp.actions:
                    action_value = self._value_calculater(state, action, state_values)
                    action_prob = policy.action_prob(state, action)
                    new_value += action_prob * action_value
                state_values[state] = new_value
                delta = max(delta, abs(v - new_value))
            
            if delta < self.theta:
                break
        
        return state_values

    def policy_improvement(self, policy: AbstractPolicy, value_function: np.ndarray) -> Policy:
        """
        Performs policy improvement on a given policy.

        :return: policy stability. (True or False)
        """
        policy_stable = True
        for state in self.mdp.states:
            old_action = []

            # Extracting all the action probabilities
            for action in self.mdp.actions:
                old_action.append(policy.action_prob(state, action))

            old_action = np.argmax(old_action)
                
            action_values = []
            for action in self.mdp.actions:
                expected_value = self._value_calculater(state, action, value_function)
                action_values.append(expected_value)
                
            best_action = np.argmax(action_values)
            if old_action != best_action:
                policy_stable = False

            # Make the policy deterministic for the best action
            policy.set_action_probabilities(state, [1.0 if a == best_action else 0.0 for a in range(len(self.mdp.actions))])

        return policy_stable
    
    def _value_calculater(self, state: AbstractMDP, action: AbstractMDP, state_values: np.ndarray):
        """
        This function calculates the value of a state-action pair based on transition probabilities,
        rewards, discount factor, and current state values.
        
        :param state: the current state in MDP
        :param action: the action taken in the current state
        :param state_values: dictionary that contains the estimated values of each state in the MDP
        :return: calculated value based on the state, action, and state values provided as input
        """
        value = 0
        for next_state in self.mdp.states:
            transition_prob = self.mdp.transition_prob(new_state=next_state, state=state, action=action)
            reward = self.mdp.reward(state=state, action=action)
            value += transition_prob * (reward + self.mdp.discount_factor * state_values[next_state])
        return value

    def _initialize_random_policy(self) -> Policy:
        """
        Initializes a random policy where each state's action distribution is random.

        :return: A random Policy object.
        """
        random_policy = Policy()

        for state in range(len(self.mdp.states)):
            # Generate a random probability distribution for each state's actions
            action_probabilities = np.random.rand(len(self.mdp.actions))
            action_probabilities /= action_probabilities.sum()  # Normalize
            
            # Setting the action/probabilities for this state
            random_policy.set_action_probabilities(state, action_probabilities.tolist())
        
        return random_policy

    def _optimal_value_function(self):
        """
        This helper method iteratively calculates the optimal value function for a
        Markov Decision Process (MDP) using the value iteration algorithm.

        :return: the optimal value function
        """
        value_function = np.zeros(len(self.mdp.states))
        
        for _ in range(1000):
            delta = 0
            for state in self.mdp.states:
                v = value_function[state]
                action_values = []
                for action in self.mdp.actions:
                    expected_value = self._value_calculater(state, action, value_function)
                    action_values.append(expected_value)
                value_function[state] = max(action_values)
                delta = max(delta, abs(v - value_function[state]))
            
            if delta < self.theta:
                break

        return value_function
