from typing import Dict, Tuple


class RewardFunction:
    def __init__(self, rewards: Dict[Tuple[int, int], float]):
        """
        Initializes the reward function with a dictionary.

        :param rewards: A dictionary where keys are (state, action) tuples and values are floats
                        representing the rewards for each (state, action) pair.
        """
        self.rewards = rewards

    def __call__(self, state: int, action: int) -> float:
        """
        Returns the reward for a given state and action.

        :param state: Current state
        :param action: Action taken

        :return: A float representing the reward for the given (state, action) pair.
        """
        if (state, action) in self.rewards:
            return self.rewards[(state, action)]
        else:
            raise ValueError(f"No reward defined for state {state} and action {action}.")
