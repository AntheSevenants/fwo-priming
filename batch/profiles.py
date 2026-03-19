import model.enums

params = {
    "regular": {
        "num_agents": 10,
        "starting_probabilities": [[0.1, 0.9], [0.2, 0.8], [0.4, 0.6], [0.5, 0.5]],
        "priming_strength": [0.05, 0.2, 0.4, 0.5, 0.8],
        "base_rate_change_strength": [0.1, 0.2, 0.4, 0.5, 0.8],
        "inverse_frequency_exponent": [0, 0.5, 1, 2, 4],
        "inverse_frequency_max_multiplier": 4,
        "priming_opportunity": [0.05, 0.2, 0.4, 0.5, 0.8, 1],
        "decay_strength": [0, 0.2, 0.5, 0.8],
        "decay_to": model.enums.DecayTo.BASE_RATE,
        "affects_base_rate": model.enums.AffectsBaseRate.RECEPTION,
        "allow_decay_stop": False,
    },
    "lite": {
        "num_agents": 10,
        "starting_probabilities": [[0.1, 0.9], [0.2, 0.8], [0.4, 0.6], [0.5, 0.5]],
        "priming_strength": 0.4,
        "base_rate_change_strength": 0.05,
        "inverse_frequency_exponent": [0, 1],
        "inverse_frequency_max_multiplier": 2,
        "priming_opportunity": [ 0.05, 0.4 ],
        "decay_strength": [0, 0.2, 0.5],
        "decay_to": model.enums.DecayTo.BASE_RATE,
        "affects_base_rate": model.enums.AffectsBaseRate.RECEPTION,
        "allow_decay_stop": False,
    },
}

# Entrenchment profile, based on lite but with some params changed
params["entrenchment"] = {
    **params["lite"],
    "decay_strength": 0,
    "base_rate_change_strength": [ 0.05, 0.4 ],
    "use_activation": False,
    "inverse_frequency_exponent": 0
}