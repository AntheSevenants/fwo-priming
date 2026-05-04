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

    def __compute_norm__(self, with_replicator_selection: bool = True) -> np.ndarray:
        """Internal function which normalises activation levels. All values sum to one.
        If perception is logarithmic, a log10 pass is applied first.

        Args:
            with_replicator_selection (bool, optional): Whether to add the replicator selection sway to the innovative variant. Defaults to True.

        Returns:
            np.ndarray: A numpy array containing the activation levels, normalised to sum to one.
        """

        to_normalise = self.level
        if self.model_params.logarithmic_perception:
            # Prevent negative log through addition of 1
            to_normalise = np.log10(1 + self.level)

        if with_replicator_selection:
            to_normalise[
                self.model_params.innovation_index
            ] += self.model_params.replicator_selection_sway

        return np.divide(to_normalise, np.sum(to_normalise))

    @property
    def norm(self) -> np.ndarray:
        """The normalised activation levels. All values sum to one.
        If perception is logarithmic, a log10 pass is applied first.
        If replicator selection is active, this bonus is added to the innovative variant.

        Returns:
            np.ndarray(float): A numpy array containing the activation levels, normalised to sum to one.
        """

        return self.__compute_norm__(with_replicator_selection=True)

    @property
    def neutral_norm(self):
        """The normalised activation levels. All values sum to one.
        This property does *not* add the replicator selection bonus.

        Returns:
            np.ndarray(float): A numpy array containing the activation levels, normalised to sum to one.
            The replicator selection bonus is not omitted.
        """

        return self.__compute_norm__(with_replicator_selection=False)
