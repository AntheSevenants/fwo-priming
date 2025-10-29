import mesa
import math
import numpy as np

import model.agent

class PrimingModel(mesa.Model):
    """A model of syntactic priming"""

    def __init__(self, num_agents=50, seed=None):
        super().__init__(seed=seed)

        print("Seed is", seed)

        agents = model.agent.PrimingAgent.create_agents(model=self, n=num_agents)

        model_reporters = {}

        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters
        )

        self.datacollector.collect(self)

    def step(self):
        pass