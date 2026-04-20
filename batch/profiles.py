import model.enums

params = {
    "regular": {
        "num_agents": 10,
        "memory_size": [ 1000 ],
        "starting_probabilities": [[0.2, 0.8], [0.4, 0.6]],
        "priming_strength": [0.05, 0.4, 2, 4],
        # "inverse_frequency_exponent": [0, 0.5, 1],
        "inverse_frequency_max_multiplier": 4,
        "priming_opportunity": [0.001, 0.01, 0.05, 0.4, 0.8],
        "decay_strength": [0, 0.001, 0.01, 0.2, 0.5, 0.8],
        "decay_to": model.enums.DecayTo.BASE_RATE,
        "affects_base_rate": [ model.enums.AffectsBaseRate.RECEPTION ],
        "allow_decay_stop": False,
        "activation_cap": False,
        "base_rate_update_mechanism": model.enums.BaseRateUpdateMechanism.COUNT,
        "logarithmic_perception": [True, False],
    },
    "lite": {
        "num_agents": 10,
        "starting_probabilities": [[0.1, 0.9], [0.2, 0.8], [0.4, 0.6], [0.5, 0.5]],
        "priming_strength": 0.4,
        "inverse_frequency_exponent": [0, 0.5, 1],
        "inverse_frequency_max_multiplier": 2,
        "priming_opportunity": [ 0.001, 0.01, 0.05, 0.4 ],
        "decay_strength": [0, 0.2, 0.5, 0.9],
        "decay_to": model.enums.DecayTo.BASE_RATE,
        "affects_base_rate": model.enums.AffectsBaseRate.RECEPTION,
        "allow_decay_stop": False,
        "activation_cap": False,
        "base_rate_update_mechanism": model.enums.BaseRateUpdateMechanism.COUNT,
    },
    "lite2": {
        "num_agents": 10,
        "starting_probabilities": [[0.1, 0.9]],
        "priming_strength": 0.4,
        "inverse_frequency_exponent": [0, 0.5, 1],
        "inverse_frequency_max_multiplier": 2,
        "priming_opportunity": [ 0.01, 0.05, 0.4 ],
        "decay_strength": [0.2, 0.5, 0.9],
        "decay_to": model.enums.DecayTo.BASE_RATE,
        "affects_base_rate": model.enums.AffectsBaseRate.RECEPTION,
        "allow_decay_stop": False,
        "activation_cap": False,
        "base_rate_update_mechanism": model.enums.BaseRateUpdateMechanism.LATERAL_INHIBITION,
        "linear_increase": [0.4, 0.8]
    },
}

params["lite_latin"] = {
    **params["lite"],
    "base_rate_update_mechanism": model.enums.BaseRateUpdateMechanism.LATERAL_INHIBITION
}

params["lite_latin_log"] = {
    **params["lite"],
    "base_rate_update_mechanism": model.enums.BaseRateUpdateMechanism.LATERAL_INHIBITION,
    "logarithmic_perception": True,
    "inverse_frequency_exponent": 0
}

# Entrenchment profile, based on lite but with some params changed
params["entrenchment"] = {
    **params["lite"],
    "decay_strength": 0,
    "use_activation": False,
    "inverse_frequency_exponent": 0,
}

params["entrenchment_latin"] = {
    **params["entrenchment"],
    "base_rate_update_mechanism": model.enums.BaseRateUpdateMechanism.LATERAL_INHIBITION
}