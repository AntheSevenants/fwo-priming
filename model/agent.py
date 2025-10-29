import mesa


class PrimingAgent(mesa.Agent):
    """A speaker in the model"""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)