from typing import TYPE_CHECKING
import numpy as np

import model.entropy

if TYPE_CHECKING:
    import model.model_defaults


class Activation:
    def __init__(
        self, model_params: "model.model_defaults.Parameters", init_level: np.ndarray
    ):
        self.model_params = model_params

        # Each value has a legal range from 0 to infinity
        self.level = init_level
        self.entropy = model.entropy.Entropy(self.level)

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

        to_normalise[
            self.model_params.innovation_index
        ] += self.model_params.replicator_selection_sway

        return np.divide(to_normalise, np.sum(to_normalise))
