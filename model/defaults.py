import model.enums
import numpy as np

from dataclasses import dataclass, asdict, field
from typing import List, Optional


@dataclass
class Parameters:
    # ----
    # Model housekeeping
    # ----

    num_agents: int = 50
    seed: Optional[int] = None
    # Allow model to stop early if consensus is reached
    early_stop: bool = False

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

    # What probability does an agent have to be primed in every time step?
    priming_opportunity: float = 1.0

    # ----
    # Decay
    # ----

    # How quickly does priming decay if ctx not used?
    decay_strength: float = 0.0
    # Decay to uniform distribution or starting distribution?
    decay_to_starting_probabilities: bool = False
    # Allow decay to stop when a construction has "won"
    allow_decay_stop: bool = True

    # Constructions in alternation and eligible for priming
    constructions: List[str] = field(default_factory=lambda: ["ACT", "PASS"])

    def __post_init__(self):
        # After the fact, compute the number of constructions
        self.num_constructions = len(self.constructions)

        # Uniform distribution
        self.uniform_dist = np.ones(self.num_constructions) / self.num_constructions