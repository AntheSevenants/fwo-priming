from typing import TYPE_CHECKING
import numpy as np
import copy

import model.entropy

if TYPE_CHECKING:
    import model.model_defaults

class BaseRate:
    def __init__(
        self,
        model_params: 'model.model_defaults.Parameters',
        update_mechanism: int,
        is_innovator: bool,
    ):
        self.model_params = model_params
        self.update_mechanism = update_mechanism
        self.is_innovator = is_innovator

        self.level: np.ndarray = np.array([], dtype=np.float64)
        self.init_construction_probs()

        self.entropy = model.entropy.Entropy(self.level)

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
