import mesa
import numpy as np

from typing import Optional


class Tracker:
    """The tracker is the facilitates keeping track of the current model state.
    """
    def __init__(self, model: mesa.Model):
        """Initialise a Tracker by supplying a MESA model.

        Args:
            model (mesa.Model): A MESA model
        """

        # This will break pickling the model forever. Good :-)
        self.model = model

        # Per step!
        self.chosen_constructions = []

        # For each agent, per step
        self.construction_probabilities_per_agent = []

        # Across agents, per step
        self.construction_probabilities_across_agents = []

    def register_construction_chosen(self, construction_index: int):
        """Register in the tracker which construction was chosen by an agent.

        Args:
            construction_index (int): The index of the chosen construction
        """

        self.chosen_constructions.append(construction_index)

    def get_property_per_agent(
            self, property_name: str, for_innovator: bool | None = None, index: int | None = None
        ):
        """Retrieve a list of property values of each agent. If a property is multi-dimensional, you can ask to retrieve the value of one of the dimensions.

        Args:
            property_name (str): The name of the property that should be retrieved for each agent.
            for_innovator (bool, optional): Whether to get this property for innovators (True) or conservators (False). Returns for all agents if None. Defaults to None.
            index (int, optional): The index of the multi-dimensional value that should be retrieved, if multi-dimensional. Returns entire list if None. Defaults to None.

        Returns:
            np.Array: A numpy array containing the requested values.
        """

        agent_property_dist = []

        # Get the property for each agent
        for agent in self.model.agents:
            # Only get the correct type
            if for_innovator is not None and agent.is_innovator != for_innovator:
                continue 

            property_value = getattr(agent.atts, property_name)
            if index is not None:
                property_value = property_value[index]
            agent_property_dist.append(property_value)

        # Turn into numpy array
        return np.array(agent_property_dist)
    
    def get_property_mean_across_agents(
            self, property_name: str, for_innovator: bool | None = None, index: Optional[int] = None
        ):
        """Retrieve the mean of a requested property value across agents. If a property is multi-dimensional, you can ask to take the mean of the values of just one of the dimensions.

        Args:
            property_name (str): The name of the property that should be retrieved for each agent.
            for_innovator (bool, optional): Whether to get this property for innovators (True) or conservators (False). Returns for all agents if None. Defaults to None.
            index (int, optional): The index of the multi-dimensional value that should be retrieved, if multi-dimensional. Returns entire list if None. Defaults to None.

        Returns:
            float: A number of the mean of the value
        """

        agent_property_dist = self.get_property_per_agent(property_name, index=index, for_innovator=for_innovator)

        return agent_property_dist.mean(axis=0)
    
    
    def get_property_median_across_agents(
            self, property_name: str, for_innovator: bool | None = None, index: Optional[int] = None
        ):
        """Retrieve the median of a requested property value across agents. If a property is multi-dimensional, you can ask to take the median of the values of just one of the dimensions.

        Args:
            property_name (str): The name of the property that should be retrieved for each agent.
            for_innovator (bool, optional): Whether to get this property for innovators (True) or conservators (False). Returns for all agents if None. Defaults to None.
            index (int, optional): The index of the multi-dimensional value that should be retrieved, if multi-dimensional. Returns entire list if None. Defaults to None.

        Returns:
            float: A number of the median of the value
        """

        agent_property_dist = self.get_property_per_agent(property_name, index=index)

        return np.median(agent_property_dist, axis=0)
