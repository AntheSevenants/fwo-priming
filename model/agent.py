import mesa
import model.enums
import model.entropy
import numpy as np

from typing import Self

class PrimingAgent(mesa.Agent):
    """A speaker in the model"""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.init_construction_probs()

    def init_construction_probs(self):
        # Assign starting probabilities to the constructions
        self.probs = np.zeros(self.model.params.num_constructions)

        # If all agents start with an equal probability distribution, adhere to this distribution
        if self.model.params.starting_probabilities_type == model.enums.StartingProbabilities.EQUAL:
            # If we start without predetermined probabilities, do equal probabilities
            if self.model.params.starting_probabilities is None:
                self.probs = np.ones(self.model.params.num_constructions) / self.model.params.num_constructions
            # Else, adopt the given starting probabilities
            else:
                self.probs = np.array(self.model.params.starting_probabilities)
        elif self.model.params.starting_probabilities_type == model.enums.StartingProbabilities.RANDOM:
            random_numbers = self.model.nprandom.random(self.model.params.num_constructions)
            # Normalise
            self.probs = random_numbers / random_numbers.sum()

        # Remember these initial probabilities
        self.starting_probs = self.probs.copy()
        # We keep a "log" of entropy so we can compute the delta
        self.entropy = np.array([ model.entropy.compute_entropy(self.probs) ] * 2)

    def update_entropy_history(self):
        new_entropy_value = model.entropy.compute_entropy(self.probs)
        self.entropy = np.concatenate((self.entropy[1:], [ new_entropy_value ]))

    def interact_do(self):
        # First, establish whether there is a priming opportunity
        # If no chance at priming anyway, do not have a conversation
        priming_chance = self.model.nprandom.random()
        if priming_chance > self.model.params.priming_opportunity:
            # Stop decaying if maximum preference was reached
            # (and this is allowed)
            max_prob = np.max(self.probs)
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
        chosen_construction_index = self.model.nprandom.choice(construction_indices, p=self.probs)

        self.model.tracker.register_construction_chosen(chosen_construction_index)

        hearer_agent.receive_construction(chosen_construction_index)

    def receive_construction(self, construction_index: int):
        # Now the hearer has to adjust their internal distribution
        self.probs[construction_index] += self.model.params.priming_strength

        # Renormalise all probabilities
        self.probs = np.divide(self.probs, self.probs.sum())

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

        # If we do not decay to the starting probabilities, decay to a uniform distribution instead
        if not self.model.params.decay_to_starting_probabilities:
            uniform_dist = np.ones(self.model.params.num_constructions) / self.model.params.num_constructions
        # Else, the distribution to decay to is the starting probabilities
        else:
            uniform_dist = self.starting_probs.copy()

        self.probs = (1 - self.model.params.decay_strength) * self.probs + self.model.params.decay_strength * uniform_dist