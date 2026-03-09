import mesa
import math
import numpy as np

import model.agent
import model.enums
import model.tracker
import model.model_defaults

from dataclasses import dataclass, asdict
from typing import List, Optional


class PrimingModel(mesa.Model):
    """A model of syntactic priming"""

    def __init__(self, params: model.model_defaults.Parameters):
        """Initialise the Priming Model

        Args:
            params (model.model_defaults.Parameters): 
            Parameters object detailing what parameters the simulation should use
        """

        # Load parameters
        self.params = params

        # Load parent class, set random and seed
        super().__init__(seed=self.params.seed)
        self.nprandom = np.random.default_rng(self.params.seed)

        # Agents
        agents = model.agent.PrimingAgent.create_agents(
            model=self, n=self.params.num_agents
        )

        # Model data collection
        self.tracker = model.tracker.Tracker(self)

        model_reporters = {
            "ctx_activation_per_agent": lambda model: model.tracker.get_property_per_agent(
                "activation"
            ),
            "ctx_probs_per_agent": lambda model: model.tracker.get_property_per_agent(
                "activation_norm"
            ),
            "ctx_base_rate_per_agent": lambda model: model.tracker.get_property_per_agent(
                "base_rate"
            ),
            "starting_probs_per_agent": lambda model: model.tracker.get_property_per_agent(
                "starting_base_rate"
            ),
            "ctx_activation_mean": lambda model: model.tracker.get_property_mean_across_agents(
                "activation"
            ),
            "ctx_probs_mean": lambda model: model.tracker.get_property_mean_across_agents(
                "activation_norm"
            ),
            "ctx_base_rate_mean": lambda model: model.tracker.get_property_mean_across_agents(
                "base_rate"
            ),
            "ctx_entropy_per_agent": lambda model: model.tracker.get_property_per_agent(
                "entropy", index=-1
            ),
            "ctx_entropy_mean": lambda model: model.tracker.get_property_mean_across_agents(
                "entropy", index=-1
            ),
            "consensus_reached": lambda model: (
                False
                if len(model.datacollector.model_vars["ctx_probs_mean"]) == 0
                else (
                    np.any(
                        np.isclose(
                            model.datacollector.model_vars["ctx_probs_mean"][-1], 1
                        )
                    )
                    is np.True_
                )
            ),
        }
        self.datacollector = mesa.DataCollector(model_reporters=model_reporters)
        self.datacollector.collect(self)

    def step(self):
        """Routine run at every step in the simulation"""

        # Make all agents interact in a random order
        self.agents.shuffle_do("interact_do")
        # Now do decay for all agents
        self.agents.do("do_decay")
        # Collect information about this specific model step
        self.datacollector.collect(self)

        # Stop the simulation if consensus is reached and early stopping is allowed
        if (
            self.datacollector.model_vars["consensus_reached"][-1]
            and self.params.early_stop
        ):
            self.running = False
