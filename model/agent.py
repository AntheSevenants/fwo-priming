from __future__ import annotations

import mesa
import model.enums
import model.entropy
import model.agent_defaults
import numpy as np

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

    def update_entropy_history(self):
        """We keep a log of the entropy of the activation levels for each agent.
        This function computes the entropy of the current distribution and adds it to the log.
        """

        new_entropy_value = model.entropy.compute_entropy(self.atts.activation_norm)
        self.atts.entropy = np.concatenate((self.atts.entropy[1:], [ new_entropy_value ]))

    def interact_do(self):
        """Each timestep, each agent has the opportunity to interact with another random agent.
        This is an outer wrapper script that decides whether the interaction occurs, 
        and/or whether there needs to be decay instead.
        """

        # First, establish whether there is a priming opportunity
        # If no chance at priming anyway, do not have a conversation
        priming_chance = self.model.nprandom.random()
        if priming_chance > self.model.params.priming_opportunity:
            # Stop decaying if maximum preference was reached
            # (and this is allowed)
            max_prob = np.max(self.atts.activation)
            if max_prob < 1 and not self.model.params.allow_decay_stop:
                self.do_decay()
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
        construction_indices = list(range(len(self.model.params.constructions)))
        chosen_construction_index = self.model.nprandom.choice(construction_indices, p=self.atts.activation_norm)

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

        # Add a constant rate to the current base rate, and then ...
        self.atts.base_rate[construction_index] += self.model.params.base_rate_change_strength
        
        # ... renormalise so the sum sums to one.
        self.atts.base_rate = np.divide(self.atts.base_rate, self.atts.base_rate.sum())

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
        # We need the uniform distribution to know where the mean is
        mean_probability = self.model.params.uniform_dist[0]
        # What is the current probability of this construction?
        # Depending on whether it is high or low, we will adjust the priming strength
        current_probability = self.atts.activation_norm[construction_index]

        # Safety for division, like Laplace
        epsilon = 0.001

        multiplier = np.power(
            # Very improbable outcomes will lead to a much higher multiplier
            np.divide(mean_probability, current_probability + epsilon),
            # The exponent tries to adjust how strong the multiplier is
            self.model.params.inverse_frequency_exponent)
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

        # Now the hearer has to adjust their internal distribution
        # Maximum activation level is one!
        self.atts.activation[construction_index] = min(
            self.atts.activation[construction_index] + self.compute_priming_strength(construction_index),
            1
        )

        # If reception is set to affect base rate, update base rate
        if model.enums.AffectsBaseRate.affects_reception(self.model.params.affects_base_rate):
            self.update_base_rate(construction_index)

        # Renormalise all probabilities
        # self.atts.activation = np.divide(self.atts.activation, self.atts.activation.sum())

        # Update internal entropy log
        self.update_entropy_history()

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

        self.atts.activation = (1 - self.model.params.decay_strength) * self.atts.activation + self.model.params.decay_strength * uniform_dist