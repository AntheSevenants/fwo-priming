import mesa
import math
import numpy as np

import model.agent
import model.enums
import model.tracker
import model.reporters
import model.model_defaults

from dataclasses import dataclass, asdict
from typing import List, Optional, Callable


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
        super().__init__(rng=self.params.seed)
        self.nprandom = np.random.default_rng(self.params.seed)

        # Calculate the number of innovators and imitators
        num_innovators = round(self.params.num_agents * self.params.innovators_share)
        num_imitators = self.params.num_agents - num_innovators
        agent_types = [True] * num_innovators + [False] * num_imitators

        # Agents
        agents = model.agent.PrimingAgent.create_agents(
            model=self, n=self.params.num_agents, is_innovator=agent_types
        )

        # Model data collection
        self.tracker = model.tracker.Tracker(self)

        model_reporters = model.reporters.get_model_reporters()
        consensus_lambda: Callable[[PrimingModel], bool] = lambda model: (
            False
            if len(model.datacollector.model_vars["ctx_base_rate_mean"]) == 0
            else (
                np.any(
                    np.isclose(
                        model.datacollector.model_vars["ctx_base_rate_mean"][-1], 1
                    )
                )
                is np.True_
            )
        )
        model_reporters["consensus_reached"] = consensus_lambda

        self.datacollector = mesa.DataCollector(model_reporters=model_reporters)
        self.datacollector.collect(self)

    def step(self):
        """Routine run at every step in the simulation"""

        # Make all agents interact in a random order
        self.agents.shuffle_do("interact_do")

        if self.time % self.params.datacollector_step_size == 0:
            # Collect information about this specific model step
            self.datacollector.collect(self)

        if self.params.use_activation:
            # Now do decay for all agents
            self.agents.do("do_decay")

        # Stop the simulation if consensus is reached and early stopping is allowed
        if (
            self.datacollector.model_vars["consensus_reached"][-1]
            and self.params.early_stop
        ):
            self.running = False
