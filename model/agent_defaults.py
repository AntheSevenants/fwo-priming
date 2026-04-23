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
    is_innovator: bool

    def __post_init__(self):
        # We need the parent model parameters in order to be able to initialise
        # the probabilities and starting probabilities and such
        if self.model_params is None:
            raise ValueError("Model parameters cannot be None")
        
        # Set base rate and initial levels
        self.base_rate = BaseRate(
            self.model_params,
            update_mechanism=self.model_params.base_rate_update_mechanism,
            is_innovator=self.is_innovator,
        )

        # Set activation and initial levels
        self.activation = Activation(
            self.model_params, copy.deepcopy(self.base_rate.level)
        )


class Entropy:
    def __init__(self, level: np.ndarray):
        # A "logbook" of entropy, needed so we can compute the delta
        self.history: np.ndarray = np.array([model.entropy.compute_entropy(level)] * 2)

    def update(self, new_value):
        new_entropy_value = model.entropy.compute_entropy(new_value)
        self.history = np.concatenate((self.history[1:], [new_entropy_value]))


class Activation:
    def __init__(
        self, model_params: model.model_defaults.Parameters, init_level: np.ndarray
    ):
        self.model_params = model_params

        # Each value has a legal range from 0 to infinity
        self.level = init_level
        self.entropy = Entropy(self.level)
        

    @property
    def norm(self):
        """The normalised activation levels. All values sum to one.
        If perception is logarithmic, a log10 pass is applied first.

        Returns:
            np.array(float): A numpy array containing the activation levels, normalised to sum to one.
        """
        to_normalise = self.level
        if self.model_params.logarithmic_perception:
            # Prevent negative log through addition of 1
            to_normalise = np.log10(1 + self.level)

        return np.divide(to_normalise, np.sum(to_normalise))


class BaseRate:
    def __init__(
        self,
        model_params: model.model_defaults.Parameters,
        update_mechanism: int,
        is_innovator: bool,
    ):
        self.model_params = model_params
        self.update_mechanism = update_mechanism
        self.is_innovator = is_innovator

        self.level: np.ndarray = field(
            default_factory=lambda: np.array([], dtype=np.float64)
        )
        self.init_construction_probs()

        self.entropy = Entropy(self.level)

    def init_construction_probs(self):
        """Initialies the base rate starting probabilities based on the starting probability mode.

        Raises:
            NotImplementedError: Randomised starting probabilities are not implemented yet
        """

        # Assign starting base rate to the constructions
        if self.is_innovator:
            self.level = np.array(
                [
                    self.model_params.innovator_innovation_share,
                    1 - self.model_params.innovator_innovation_share,
                ]
            )
        else:
            self.level = np.array(
                [
                    self.model_params.conservator_innovation_share,
                    1 - self.model_params.conservator_innovation_share,
                ]
            )

    def __post_init__(self):
        # Now that the initial probabilities have been set, do some housekeeping
        # (copying the starting probs, computing entropy etc.)
        self.starting_level = copy.deepcopy(self.level)
