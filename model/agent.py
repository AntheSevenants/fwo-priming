from __future__ import annotations

import mesa
import model.enums
import model.entropy
import model.agent_defaults
import numpy as np
import copy

from typing import Self, TYPE_CHECKING
if TYPE_CHECKING:
    from model.model import PrimingModel

class PrimingAgent(mesa.Agent):
    """A speaker in the model"""

    def __init__(self, priming_model: PrimingModel):
        """Initialise a PrimingModel agent

        Args:
            priming_model (PrimingModel): 
            The PrimingModel for which the agent is initialised
        """

        # Pass the parameters to the parent class.
        super().__init__(priming_model)

        # Populate the agent's parameters based off the model parameters
        self.atts = model.agent_defaults.Attributes(priming_model.params)

    @property
    def construction_probs_norm(self):
        # If activation levels are enabled, we just use the model as normal
        if self.model.params.use_activation:
            return self.atts.activation_norm
        else:
        # Else, we're only working with entrenchment. So only return the base rate
        # (the base rate is always normalised)
            return self.atts.base_rate

    def update_entropy_history(self):
        """We keep a log of the entropy of the activation levels for each agent.
        This function computes the entropy of the current distribution and adds it to the log.
        """

        new_entropy_value = model.entropy.compute_entropy(self.atts.activation_norm)
        self.atts.entropy = np.concatenate((self.atts.entropy[1:], [ new_entropy_value ]))

    
    def update_base_rate_entropy_history(self):
        """We keep a log of the entropy of the base rates for each agent.
        This function computes the entropy of the current base rate and adds it to the log.
        """

        new_entropy_value = model.entropy.compute_entropy(self.atts.base_rate)
        self.atts.base_rate_entropy = np.concatenate(
            (self.atts.base_rate_entropy[1:], [new_entropy_value])
        )

    def interact_do(self):
        """Each timestep, each agent has the opportunity to interact with another random agent.
        This is an outer wrapper script that decides whether the interaction occurs, 
        and/or whether there needs to be decay instead.
        """

        # First, establish whether there is a priming opportunity
        # If no chance at priming anyway, do not have a conversation
        priming_chance = self.model.nprandom.random()
        if priming_chance > self.model.params.priming_opportunity:
            return

        # Choose a random other agent that is not the agent itself
        while True:
            hearer_agent = self.random.choice(self.model.agents)
            if self != hearer_agent:
                break

        # If there *is* an interaction, do interact
        self.interact(hearer_agent)

    def interact(self, hearer_agent: Self):
        """This function describes the routine that every agent goes through when they interact.

        Args:
            hearer_agent (Self): The other agent with which the current agent interacts.
        """

        # We choose what construction the agent will utter based on the current activation levels
        # "More activated" constructions are more likely to be chosen.
        chosen_construction_index = self.model.nprandom.choice(
            self.model.params.construction_indices,
            p=self.construction_probs_norm
        )

        # If production is set to affect base rate, update the base rate for the chosen construction
        if model.enums.AffectsBaseRate.affects_production(self.model.params.affects_base_rate):
            self.update_base_rate(chosen_construction_index)

        # In the model tracker, register that this construction was chosen
        self.model.tracker.register_construction_chosen(chosen_construction_index)

        # Now, the other agent "hears" the construction that we just chose.
        hearer_agent.receive_construction(chosen_construction_index)

    def update_base_rate(self, construction_index: int):
        """This function describes what happens when the base rate needs to be increased for
        a specific construction. This will cause the construction to be more likely to be chosen
        in the future. Currently, base rate updates are implemented by adding a constant value and
        then renormalising.

        Args:
            construction_index (int): The index of the chosen construction
        """

        # TODO !!!

    def compute_priming_strength(self, construction_index: int):
        """Computes the priming strength (= the float that will be added to the current activation level).

        Args:
            construction_index (int): The index of the chosen construction

        Returns:
            float: The computed priming strength that will be added to the chosen construction's activation level.
        """

        # If disabled, return default strength to save on resources
        if self.model.params.inverse_frequency_exponent == 0:
            return self.model.params.priming_strength

        # The priming strength we will use as a baseline to attenuate / boost
        priming_strength_base = self.model.params.priming_strength
        # What is the highest probability currently?
        compare_probability = np.max(self.atts.base_rate)
        # What is the current probability of this construction?
        # Depending on whether it is high or low, we will adjust the priming strength
        current_probability = self.atts.activation[construction_index]

        # Safety for division, like Laplace
        epsilon = 0.001

        multiplier = np.power(
            np.divide(priming_strength_base + epsilon,
                      current_probability + epsilon),
            self.model.params.inverse_frequency_exponent
        )

        # Cap the multiplier to a predetermined maximum
        multiplier = min(multiplier, self.model.params.inverse_frequency_max_multiplier)

        # Apply the multiplier and return
        priming_strength = priming_strength_base * multiplier

        return priming_strength

    def receive_construction(self, construction_index: int):
        """Runs whenever an agent "hears" a construction uttered by another agent.
        This function will reshuffle their internal distributions depending on the model parameters.

        Args:
            construction_index (int): The index of the chosen construction
        """

        # When activation levels are activated
        if self.model.params.use_activation:
            # Now the hearer has to adjust their internal distribution
            # Maximum activation level is one!
            new_activation_level = self.atts.activation[construction_index] + self.compute_priming_strength(construction_index)
            if self.model.params.activation_cap:
                new_activation_level = min(new_activation_level, 1)
            self.atts.activation[construction_index] = new_activation_level

        # If reception is set to affect base rate, update base rate
        # This happens regardless whether activation levels are activated or not
        if model.enums.AffectsBaseRate.affects_reception(self.model.params.affects_base_rate):
            self.update_base_rate(construction_index)

        # Renormalise all probabilities
        # self.atts.activation = np.divide(self.atts.activation, self.atts.activation.sum())

        # Update internal entropy log
        self.update_entropy_history()
        self.update_base_rate_entropy_history()

        # That's it!

    # $$
    # p'_k = (1 - d) \cdot p_n + d \cdot \frac{1}{N}
    # $$
    def do_decay(self):
        """This function "decays" the current activation levels.
        Depending on model parameters, the activation levels decay towards a uniform distribution,
        the starting distribution or the current base rate.
        """

        # No need to do maths if decay strength is zero
        if self.model.params.decay_strength == 0:
            return

        # Where should the activation levels decay to?
        if self.model.params.decay_to == model.enums.DecayTo.UNIFORM_DIST:
            uniform_dist = self.model.params.uniform_dist
        # The distribution to decay to is the starting probabilities
        elif self.model.params.decay_to == model.enums.DecayTo.STARTING_DIST:
            uniform_dist = self.atts.starting_base_rate.copy()
        # The distribution to decay to is the base rate
        elif self.model.params.decay_to == model.enums.DecayTo.BASE_RATE:
            uniform_dist = self.atts.base_rate.copy()

        # self.atts.activation = (1 - self.model.params.decay_strength) * self.atts.activation + self.model.params.decay_strength * uniform_dist

        # Apply decay for each construction
        for construction_index in range(self.model.params.num_constructions):
            # Compute difference with the uniform distribution (or base rate)
            activation_delta = self.atts.activation[construction_index] - uniform_dist[construction_index]

            # Multiply by decay strength to attenuate
            decay = activation_delta * self.model.params.decay_strength

            # Now subtract the decay from the current activation level for this construction index
            self.atts.activation[construction_index] -= decay

            # Snap to decay goal if within a certain range
            RANGE = 0.05
            if self.atts.activation[construction_index] - RANGE <= self.atts.base_rate[construction_index]:
                self.atts.activation[construction_index] = self.atts.base_rate[construction_index]