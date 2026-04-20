import model.enums
import numpy as np

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Type


# Mapping of parameter names to their enum classes
PARAMETER_ENUM_MAPPING: Dict[str, Type] = {
    "starting_probabilities_type": model.enums.StartingProbabilities,
    "decay_to": model.enums.DecayTo,
    "affects_base_rate": model.enums.AffectsBaseRate,
}


@dataclass
class Parameters:
    # ----
    # Model housekeeping
    # ----

    num_agents: int = 50
    seed: Optional[int] = None
    # Allow model to stop early if consensus is reached
    early_stop: bool = False
    # After how many steps do we collect data?
    datacollector_step_size: int = 1

    # ----
    # Base rate-related frequencies
    # ----
    memory_size: int = 1000
    base_rate_update_mechanism: int = model.enums.BaseRateUpdateMechanism.COUNT
    base_rate_change_strength: float = 0.01


    # ----
    # Priming-related probabilities
    # ----

    # How should the starting probabilities be initialised?
    # Can be EQUAL (everyone the same) or RANDOM (everyone different)
    starting_probabilities_type: int = model.enums.StartingProbabilities.EQUAL
    # If EQUAL, we can provide a custom distribution
    starting_probabilities: Optional[List[float]] = None
    priming_strength: float = 0.4
    # Make priming strength dependent on surprisal
    inverse_frequency_exponent: float = 0
    inverse_frequency_max_multiplier: float = 2
    # Whether to use activation at all
    use_activation: bool = True
    # Whether there should be an activation cap
    activation_cap: bool = True
    # Is activation logarithmically perceived?
    logarithmic_perception: bool = False

    # The hardcoded increase for the least popular construction
    # can be used to emulate an innovation becoming more popular
    linear_increase: float = 0.0

    # What probability does an agent have to be primed in every time step?
    priming_opportunity: float = 1.0

    # ----
    # Decay
    # ----

    # How quickly does priming decay if ctx not used?
    decay_strength: float = 0.0
    # Decay to uniform distribution or starting distribution, or somewhere else?
    decay_to: int = model.enums.DecayTo.UNIFORM_DIST
    # What processes affect the base rate of an agent?
    affects_base_rate: int = model.enums.AffectsBaseRate.NOTHING
    # Allow decay to stop when a construction has "won"
    allow_decay_stop: bool = True

    # Constructions in alternation and eligible for priming
    constructions: List[str] = field(default_factory=lambda: ["ACT", "PASS"])

    def __post_init__(self):
        # After the fact, compute the number of constructions
        self.num_constructions = len(self.constructions)
        # Populate the construction indices
        self.construction_indices = list(range(self.num_constructions))

        # Uniform distribution
        self.uniform_dist = np.ones(self.num_constructions) / self.num_constructions