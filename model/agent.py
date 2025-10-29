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
            random_numbers = self.model.nprandom.rand(self.model.num_constructions)
            # Normalise
            self.probs = random_numbers / random_numbers.sum()