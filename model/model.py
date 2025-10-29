import mesa
import math
import numpy as np

import model.agent
import model.types
import model.tracker


class PrimingModel(mesa.Model):
    """A model of syntactic priming"""

    def __init__(
        self,
        num_agents=50,
        starting_probabilities=model.types.StartingProbabilities.EQUAL,
        priming_strength = 0.4,
        seed=None,
    ):
        
        # Number of agents
        self.num_agents = num_agents

        # Random
        super().__init__(seed=seed)
        self.nprandom = np.random.default_rng(seed)

        # Priming probabilities
        self.starting_probabilities = starting_probabilities
        self.priming_strength = priming_strength

        # Constructions
        self.init_constructions()

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

    def init_constructions(self):
        self.constructions = ["ACT", "PASS"]
        self.num_constructions = len(self.constructions)

    def step(self):
        self.agents.shuffle_do("interact_do")
        self.datacollector.collect(self)
