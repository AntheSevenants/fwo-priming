import numpy as np


class Tracker:
    def __init__(self, model):
        # This will break pickling the model forever. Good :-)
        self.model = model

        # Per step!
        self.chosen_constructions = []

        # For each agent, per step
        self.construction_probabilities_per_agent = []

        # Across agents, per step
        self.construction_probabilities_across_agents = []

    def register_construction_chosen(self, construction_index):
        self.chosen_constructions.append(construction_index)

    def get_property_per_agent(self, property_name):
        agent_property_dist = []

        # Get the property for each agent
        for agent in self.model.agents:
            property_value = getattr(agent, property_name)
            agent_property_dist.append(property_value)

        # Turn into numpy array
        return np.array(agent_property_dist)
    
    def get_property_mean_across_agents(self, property_name):
        agent_property_dist = self.get_property_per_agent(property_name)

        return agent_property_dist.mean(axis=0)