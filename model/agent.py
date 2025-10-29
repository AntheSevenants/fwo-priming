import mesa
import model.types
import numpy as np


class PrimingAgent(mesa.Agent):
    """A speaker in the model"""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.init_construction_probs()

    def init_construction_probs(self):
        # Assign starting probabilities to the constructions
        self.probs = np.zeros(self.model.num_constructions)
        
        # If all agents start with an equal probability distribution, adhere to this distribution
        if self.model.starting_probabilities == model.types.StartingProbabilities.EQUAL:
            self.probs = np.ones(self.model.num_constructions) / self.model.num_constructions
        elif self.model.starting_probabilities == model.types.StartingProbabilities.RANDOM:
            random_numbers = self.model.nprandom.random(self.model.num_constructions)
            # Normalise
            self.probs = random_numbers / random_numbers.sum()

    def interact_do(self):        
        while True:
            hearer_agent = self.random.choice(self.model.agents)
            if self != hearer_agent:
                break

        self.interact(hearer_agent)

    def interact(self, hearer_agent):
        construction_indices = list(range(len(self.model.constructions)))
        chosen_construction_index = self.model.nprandom.choice(construction_indices, p=self.probs)

        self.model.tracker.register_construction_chosen(chosen_construction_index)

        hearer_agent.receive_construction(chosen_construction_index)

    def receive_construction(self, construction_index):
        # Now the hearer has to adjust their internal distribution
        self.probs[construction_index] += self.model.priming_strength

        # Renormalise all probabilities
        self.probs = np.divide(self.probs, self.probs.sum())

        # That's it!