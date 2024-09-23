import numpy as np
from mdp.reward_function import RewardFunction
from mdp.transition_function import TransitionFunction
from policy.policy import Policy
from mdp.mdp import MDP
from bellman.bellmanequationsolver import BellmanEquationSolver
from dynamic_programming.dynamicprogrammingsolver import DynamicProgrammingSolver

def Create_mdp_and_policy():
    # S0 = 0; S1 = 1; S2 = 2
    states = [0, 1, 2]    # Set of states actions represented as a list of integers.
    actions = [0, 1]    # a1 = 0; a2 = 1
    
    # Define rewards using a dictionary
    rewards = {
        (0, 0): -0.9,           # state 0, action 0 gets reward -1.
        (0, 1): -0.9,
        (1, 0): -0.1,
        (1, 1): -0.1,
        (2, 0): -0.9,
        (2, 1): -0.1
    }

    # Create the RewardFunction object
    reward_function = RewardFunction(rewards)

    # Define transition probabilities using a dictionary
    transitions = {
        (0, 0): np.array([0.1, 0, 0.9]),      # For state one, action one we get probability vector (0.7, 0.2, 0.1) representing the probability to transition to state 0, 1, 2 respectively.
        (0, 1): np.array([0.1, 0.9, 0]),
        (1, 0): np.array([0.9, 0.1, 0]),
        (1, 1): np.array([0.9, 0.1, 0]),
        (2, 0): np.array([0, 0.9, 0.1]),
        (2, 1): np.array([0.9, 0, 0.1])
    }

    # Create the TransitionFunction object
    transition_function = TransitionFunction(transitions)

    # Creating the MDP
    mdp = MDP(states, actions, transition_function, reward_function, discount_factor=0.9)

    # Setting the policies
    policy = Policy()
    policy.set_action_probabilities(0, [0.4, 0.6])
    policy.set_action_probabilities(1, [0.4, 0.6])
    policy.set_action_probabilities(2, [0.4, 0.6])

    return mdp, policy

def main() -> None:

    mdp, policy = Create_mdp_and_policy()

    Solver = BellmanEquationSolver(mdp)
    Policy_evaluation = Solver.policy_evaluation(policy)

    print(f"Policy Evaluation: {Policy_evaluation}")

    DPSolver = DynamicProgrammingSolver(mdp)

    Optimal_Policy_PI = DPSolver.policy_iteration()

    print("Optimal Policy arrived at by Policy iteration:")
    print(Optimal_Policy_PI.action_dist)

    Optimal_Policy_VI = DPSolver.value_iteration()

    print("Optimal Policy arrived at by Value iteration:")
    print(Optimal_Policy_VI.action_dist)

if __name__ == "__main__":
    main()
