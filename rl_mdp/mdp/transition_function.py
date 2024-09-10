import numpy as np
from typing import Dict, Tuple


class TransitionFunction:
    def __init__(self, transitions: Dict[Tuple[int, int], np.ndarray]):
        """
        Initializes the transition function with a dictionary.

        :param transitions: A dictionary where keys are (state, action) tuples and values are NumPy arrays
                       representing the probabilities of transitioning to each possible next state.
        """
        self.transitions = transitions

    def __call__(self, state: int, action: int) -> np.ndarray:
        """
        Returns the transition probabilities for a given state and action.

        :param state: Current state
        :param action: Action taken

        :return: A NumPy array of transition probabilities to the next states.
        """
        if (state, action) in self.transitions:
            return self.transitions[(state, action)]
        else:
            raise ValueError(f"No transition probabilities defined for state {state} and action {action}.")
