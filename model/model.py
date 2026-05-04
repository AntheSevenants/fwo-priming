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

        # Is there an innovator / conservator difference?
        for_all_types = self.params.innovators_share > 0
        # If not, don't make superfluous model reporters which will slow down performance
        model_reporters = model.reporters.get_model_reporters(for_all_types=for_all_types)

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

        # Check if agents need to be replaced
        dead_agents = []
        for agent in self.agents.copy():
            if not agent.is_dead:
                continue
            
            dead_agents.append(agent)
            
            # Get a random agent that is not the agent being replaced
            parent_agent = self.get_random_agent(agent)
            # Birth new agent
            model.agent.PrimingAgent.create_agents(
                model=self, n=1, is_innovator=parent_agent.is_innovator, preset_probs=parent_agent.atts.base_rate.level
            )
        
        for dead_agent in dead_agents:
            dead_agent.remove()

        # Stop the simulation if consensus is reached and early stopping is allowed
        if (
            self.datacollector.model_vars["consensus_reached"][-1]
            and self.params.early_stop
        ):
            self.running = False

    def get_random_agent(self, speaker_agent):
        # Choose a random other agent that is not the agent itself
        while True:
            hearer_agent = self.random.choice(self.agents)
            if speaker_agent != hearer_agent and not hearer_agent.is_dead:
                break

        return hearer_agent