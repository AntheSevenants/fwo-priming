import mesa
import math
import numpy as np

import model.agent
import model.types


class PrimingModel(mesa.Model):
    """A model of syntactic priming"""

    def __init__(
        self,
        num_agents=50,
        starting_probabilities=model.types.StartingProbabilities.EQUAL,
        priming_strength = 0.4,
        seed=None,
    ):

        super().__init__(seed=seed)
        print("Seed is", seed)

        self.nprandom = np.random.default_rng(seed)

        self.starting_probabilities = starting_probabilities
        self.priming_strength = priming_strength

        self.init_constructions()
        agents = model.agent.PrimingAgent.create_agents(model=self, n=num_agents)

        model_reporters = {}
        self.datacollector = mesa.DataCollector(model_reporters=model_reporters)
        self.datacollector.collect(self)

    def init_constructions(self):
        self.constructions = ["ACT", "PASS"]
        self.num_constructions = len(self.constructions)

    def step(self):
        pass
