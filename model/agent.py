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
        # Pass the parameters to the parent class.
        super().__init__(priming_model)

        self.atts = model.agent_defaults.Attributes(priming_model.params)

    def update_entropy_history(self):
        new_entropy_value = model.entropy.compute_entropy(self.atts.activation)
        self.atts.entropy = np.concatenate((self.atts.entropy[1:], [ new_entropy_value ]))

    def interact_do(self):
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

        while True:
            hearer_agent = self.random.choice(self.model.agents)
            if self != hearer_agent:
                break

        self.interact(hearer_agent)

    def interact(self, hearer_agent: Self):
        construction_indices = list(range(len(self.model.params.constructions)))
        chosen_construction_index = self.model.nprandom.choice(construction_indices, p=self.atts.activation)

        self.model.tracker.register_construction_chosen(chosen_construction_index)

        hearer_agent.receive_construction(chosen_construction_index)

    def compute_priming_strength(self, construction_index: int):
        # If disabled, return default strength to save on resources
        if self.model.params.inverse_frequency_exponent == 0:
            return self.model.params.priming_strength

        # The priming strength we will use as a baseline to attenuate / boost
        priming_strength_base = self.model.params.priming_strength
        # We need the uniform distribution to know where the mean is
        mean_probability = self.model.params.uniform_dist[0]
        # What is the current probability of this construction?
        # Depending on whether it is high or low, we will adjust the priming strength
        current_probability = self.atts.activation[construction_index]

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
        # Now the hearer has to adjust their internal distribution
        self.atts.activation[construction_index] += self.compute_priming_strength(construction_index)

        # Renormalise all probabilities
        self.atts.activation = np.divide(self.atts.activation, self.atts.activation.sum())

        # Update internal entropy log
        self.update_entropy_history()

        # That's it!

    # $$
    # p'_k = (1 - d) \cdot p_n + d \cdot \frac{1}{N}
    # $$
    def do_decay(self):
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