from typing import TYPE_CHECKING
import numpy as np
import copy

import model.entropy

from model.enums import BaseRateUpdateMechanism

if TYPE_CHECKING:
    import model.model_defaults


class BaseRate:
    def __init__(
        self,
        model_params: "model.model_defaults.Parameters",
        update_mechanism: int,
        is_innovator: bool,
        init_probs: np.ndarray | None = None
    ):
        self.model_params = model_params
        self.update_mechanism = update_mechanism
        self.is_innovator = is_innovator

        self._level: np.ndarray = np.array([], dtype=np.float64)
        self.init_construction_probs(init_probs)
        self.init_memory()

        self.entropy = model.entropy.Entropy(self.level)

    def init_construction_probs(self, init_probs: np.ndarray | None = None):
        """Initialies the base rate starting probabilities based on the starting probability mode.

        Raises:
            NotImplementedError: Randomised starting probabilities are not implemented yet
        """

        # When a child agent is born
        if init_probs is not None:
            self._level = init_probs
            return

        # Assign starting base rate to the constructions
        if self.is_innovator:
            self._level = np.array(
                [
                    self.model_params.innovator_innovation_share,
                    1 - self.model_params.innovator_innovation_share,
                ]
            )
        else:
            self._level = np.array(
                [
                    self.model_params.conservator_innovation_share,
                    1 - self.model_params.conservator_innovation_share,
                ]
            )

    @property
    def level(self):
        if self.update_mechanism == BaseRateUpdateMechanism.DEKKER or BaseRateUpdateMechanism.RENORMALISE:
            return self._level
        elif self.update_mechanism == BaseRateUpdateMechanism.COUNT or BaseRateUpdateMechanism.LATERAL_INHIBITION:
            return np.divide(self.memory_counts, self.model_params.memory_size)
        else:
            raise ValueError("Base rate update mechanism not recognised")

    def init_memory(self):
        """Initialise the memory counts of this agent.
        This happens based on the base rate initialisation probability, and memory size.
        """

        # Fill memory
        self.memory_counts = (
            np.array(self._level.tolist()) * self.model_params.memory_size
        )

    def __post_init__(self):
        # Now that the initial probabilities have been set, do some housekeeping
        # (copying the starting probs, computing entropy etc.)
        self.starting_level = copy.deepcopy(self.level)

    def update(self, construction_index: int, deletion_index: int):
        if self.update_mechanism == BaseRateUpdateMechanism.DEKKER:
            self.update_dekker(construction_index)
        elif self.update_mechanism == BaseRateUpdateMechanism.RENORMALISE:
            self.update_renormalise(construction_index, deletion_index)
        elif (
            self.update_mechanism == BaseRateUpdateMechanism.COUNT
            or self.update_mechanism == BaseRateUpdateMechanism.LATERAL_INHIBITION
        ):
            self.update_count(construction_index, deletion_index)
        elif self.update_mechanism == BaseRateUpdateMechanism.INFINITE:
            self.update_count_infinite(construction_index)
        else:
            raise ValueError("Base rate update mechanism not recognised")

    def update_count(self, construction_index: int, deletion_index: int):
        self.memory_counts[deletion_index] = max(
            self.memory_counts[deletion_index] - 1, 0
        )

        # Then, we add to the chosen index
        self.memory_counts[construction_index] = min(
            self.model_params.memory_size,
            self.memory_counts[construction_index] + 1,
        )

    def update_count_infinite(self, construction_index: int):
        self.memory_counts[construction_index] += 1

    def update_dekker(self, construction_index: int):
        base_rate_change_strength = self.model_params.base_rate_change_strength

        self._level[construction_index] = np.divide(
            self._level[construction_index] + base_rate_change_strength,
            1 + base_rate_change_strength,
        )
        other_index = np.abs(construction_index - 1)
        self._level[other_index] = np.divide(
            self._level[other_index],
            1 + base_rate_change_strength,
        )

    def update_renormalise(self, construction_index: int, deletion_index: int):
        self._level[construction_index] = min(
            1,
            self._level[construction_index]
            + self.model_params.base_rate_change_strength,
        )
        self._level[deletion_index] = max(
            0, self._level[deletion_index] - self.model_params.base_rate_change_strength
        )
