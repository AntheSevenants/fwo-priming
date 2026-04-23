import copy
import numpy as np

from dataclasses import dataclass, asdict, field
from typing import List, Optional

import model.model_defaults
import model.enums
import model.entropy
import model.activation
import model.base_rate


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
        self.base_rate = model.base_rate.BaseRate(
            self.model_params,
            update_mechanism=self.model_params.base_rate_update_mechanism,
            is_innovator=self.is_innovator,
        )

        # Set activation and initial levels
        self.activation = model.activation.Activation(
            self.model_params, copy.deepcopy(self.base_rate.level)
        )