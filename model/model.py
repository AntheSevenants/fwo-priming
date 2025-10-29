import mesa
import math
import numpy as np

import model.agent
import model.types
import model.tracker

from typing import List, Optional


class PrimingModel(mesa.Model):
    """A model of syntactic priming"""

    def __init__(
        self,
        num_agents: int = 50,
        starting_probabilities_type: int = model.types.StartingProbabilities.EQUAL,
        starting_probabilities: List[float] = None,
        constructions: List[str] = ["ACT", "PASS"],
        priming_strength: int = 0.4,
        decay_strength: float = 0,
        priming_opportunity: float = 1,
        seed: int = None,
    ):
        
        # Number of agents
        self.num_agents = num_agents

        # Random
        super().__init__(seed=seed)
        self.nprandom = np.random.default_rng(seed)

        # Priming probabilities
        self.init_starting_probabilities(starting_probabilities_type, starting_probabilities)
        self.priming_strength = priming_strength
        self.decay_strength = decay_strength # how quickly does priming decay if ctx not used?

        # Priming opportunity = how often does a priming context appear?
        self.priming_opportunity = priming_opportunity

        # Constructions
        self.init_constructions(constructions)

        # Agents
        agents = model.agent.PrimingAgent.create_agents(model=self, n=num_agents)

        # Model data collection
        self.tracker = model.tracker.Tracker(self)

        model_reporters = {
            "ctx_probs_per_agent": lambda model: model.tracker.get_property_per_agent("probs"),
            "ctx_probs_mean": lambda model: model.tracker.get_property_mean_across_agents("probs"),
        }
        self.datacollector = mesa.DataCollector(model_reporters=model_reporters)
        self.datacollector.collect(self)

    def init_constructions(self,
                           constructions: List[str]):
        self.constructions = constructions
        self.num_constructions = len(self.constructions)

    def init_starting_probabilities(self,
                                    starting_probabilities_type: int,
                                    starting_probabilities: List[float]=None):
        self.starting_probabilities_type = starting_probabilities_type
        self.starting_probabilities = starting_probabilities

    def step(self):
        self.agents.shuffle_do("interact_do")
        self.datacollector.collect(self)
