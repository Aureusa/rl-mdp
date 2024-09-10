from typing import Dict

class DeterministicPolicy:
    """
    A class representing a deterministic policy for an MDP, where each state is mapped to a specific action.
    """

    def __init__(self):
        """
        Initializes an empty deterministic policy where no state-action mapping is set initially.
        """
        self.policy: Dict[int, int] = {}  # Dictionary to store state-action mappings

    def set_action(self, state: int, action: int) -> None:
        """
        Sets the action for a given state in the policy.

        :param state: The state for which the action should be set.
        :param action: The action to be taken in the given state.
        """
        self.policy[state] = action

    def get_action(self, state: int) -> int:
        """
        Gets the action for a given state as per the current policy.

        :param state: The state for which the action is requested.
        :return: The action assigned to the given state.
        :raises ValueError: If the state does not have an action set.
        """
        if state in self.policy:
            return self.policy[state]
        else:
            raise ValueError(f"No action defined for state {state}.")
