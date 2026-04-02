import copy
import numpy as np

from dataclasses import dataclass, asdict, field
from typing import List, Optional

import model.model_defaults
import model.enums
import model.entropy


@dataclass
class Attributes:
    model_params: model.model_defaults.Parameters

    # The internal memory of agents
    # This is for "long-term" priming effects
    memory: np.ndarray = field(
        default_factory=lambda: np.array([], np.int64)
    )

    # For posterity: the base rate that the agents were initialised with
    starting_base_rate: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    # Activation rate
    # This is for "short-term" priming
    # Each value has a legal range from 0 to 1
    activation: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    # A "log" of entropy, needed so we can compute the delta
    entropy: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    # A "log" of base rate entropy
    base_rate_entropy: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def __post_init__(self):
        # We need the parent model parameters in order to be able to initialise
        # the probabilities and starting probabilities and such
        if self.model_params is None:
            raise ValueError("model_parameters cannot be None")
        
        # Set the initial probabilities
        self.init_construction_probs()
        self.init_memory()

        # Now that the initial probabilities have been set, do some housekeeping
        # (copying the starting probs, computing entropy etc.)
        self.starting_base_rate = copy.deepcopy(self.base_rate)

        # Since we start from a neutral position, copy the base rate to activation levels
        self.activation = copy.deepcopy(self.base_rate)

        # We multiply by two because else the delta computation will fail
        self.entropy = np.array([ model.entropy.compute_entropy(self.activation) ] * 2)
        self.base_rate_entropy = np.array([ model.entropy.compute_entropy(self.base_rate) ] * 2)

    def init_construction_probs(self):
        """Initialies the base rate starting probabilities based on the starting probability mode.

        Raises:
            NotImplementedError: Randomised starting probabilities are not implemented yet
        """

        # Assign starting base rate to the constructions
        self.base_rate_probs = np.zeros(self.model_params.num_constructions)

        # If all agents start with an equal probability distribution, adhere to this distribution
        if self.model_params.starting_probabilities_type == model.enums.StartingProbabilities.EQUAL:
            # If we start without predetermined probabilities, do equal probabilities
            if self.model_params.starting_probabilities is None:
                self.base_rate_probs = np.ones(self.model_params.num_constructions) / self.model_params.num_constructions
            # Else, adopt the given starting probabilities
            else:
                self.base_rate_probs = np.array(self.model_params.starting_probabilities)
        elif self.model_params.starting_probabilities_type == model.enums.StartingProbabilities.RANDOM:
            raise NotImplementedError
            random_numbers = self.model.nprandom.random(self.model_params.num_constructions)
            # Normalise
            self.base_rate_probs = random_numbers / random_numbers.sum()

    def init_memory(self):
        """Initialise the memory counts of this agent.
        This happens based on the base rate initialisation probability, and memory size.
        """

        # Fill memory
        self.memory_counts = np.array(self.base_rate_probs) * self.model_params.memory_size
    
    @property
    def base_rate(self):
        """The normalised base frequency. All values sum to one.

        Returns:
            np.array(float): A numpy array containing the base rate, normalised to sum to one.
        """
        
        return np.divide(self.memory_counts, self.model_params.memory_size)

    @property
    def activation_norm(self):
        """The normalised activation levels. All values sum to one.

        Returns:
            np.array(float): A numpy array containing the activation levels, normalised to sum to one.
        """
        return np.divide(self.activation, np.sum(self.activation))