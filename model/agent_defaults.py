import numpy as np

from dataclasses import dataclass, asdict, field
from typing import List, Optional

import model.model_defaults
import model.enums
import model.entropy


@dataclass
class Attributes:
    model_params: model.model_defaults = None

    # The internal probability distribution of the agents
    probs: List[float] = None

    # The initial probabilities
    starting_probs: List[float] = None

    # A "log" of entropy, needed so we can compute the delta
    entropy: List[float] = None

    def __post_init__(self):
        # We need the parent model parameters in order to be able to initialise
        # the probabilities and starting probabilities and such
        if self.model_params is None:
            raise ValueError("model_parameters cannot be None")
        
        # Set the initial probabilities
        self.init_construction_probs()

        # Now that the initial probabilities have been set, do some housekeeping
        # (copying the starting probs, computing entropy etc.)
        self.starting_probs = self.probs.copy()
        # We multiply by two because else the delta computation will fail
        self.entropy = np.array([ model.entropy.compute_entropy(self.probs) ] * 2)

    def init_construction_probs(self):
        # Assign starting probabilities to the constructions
        self.probs = np.zeros(self.model_params.num_constructions)

        # If all agents start with an equal probability distribution, adhere to this distribution
        if self.model_params.starting_probabilities_type == model.enums.StartingProbabilities.EQUAL:
            # If we start without predetermined probabilities, do equal probabilities
            if self.model_params.starting_probabilities is None:
                self.probs = np.ones(self.model_params.num_constructions) / self.model_params.num_constructions
            # Else, adopt the given starting probabilities
            else:
                self.probs = np.array(self.model_params.starting_probabilities)
        elif self.model_params.starting_probabilities_type == model.enums.StartingProbabilities.RANDOM:
            random_numbers = self.model.nprandom.random(self.model_params.num_constructions)
            # Normalise
            self.probs = random_numbers / random_numbers.sum()