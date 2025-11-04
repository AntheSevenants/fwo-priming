import mesa
import math
import numpy as np

import model.agent
import model.enums
import model.tracker
import model.defaults

from dataclasses import dataclass, asdict
from typing import List, Optional


class PrimingModel(mesa.Model):
    """A model of syntactic priming"""

    def __init__(
        self,
        params: model.defaults.Parameters
    ):
        # Load parameters
        self.params = params

        # Load parent class, set random and seed
        super().__init__(seed=self.params.seed)
        self.nprandom = np.random.default_rng(self.params.seed)

        # Agents
        agents = model.agent.PrimingAgent.create_agents(model=self, n=self.params.num_agents)

        # Model data collection
        self.tracker = model.tracker.Tracker(self)

        model_reporters = {
            "ctx_probs_per_agent": lambda model: model.tracker.get_property_per_agent("probs"),
            "starting_probs_per_agent": lambda model: model.tracker.get_property_per_agent("starting_probs"),
            "ctx_probs_mean": lambda model: model.tracker.get_property_mean_across_agents("probs"),
            "consensus_reached": lambda model: False if len(model.datacollector.model_vars["ctx_probs_mean"]) == 0 else (np.any(np.isclose(model.datacollector.model_vars["ctx_probs_mean"][-1], 1)) is np.True_)
        }
        self.datacollector = mesa.DataCollector(model_reporters=model_reporters)
        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("interact_do")
        self.datacollector.collect(self)

        if self.datacollector.model_vars["consensus_reached"][-1] and self.params.early_stop:
            self.running = False