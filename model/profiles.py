import model.enums

from batch.profile import BatchProfile

params = {
    "article": BatchProfile(
        parent_params={
            "num_agents": 10,
            "innovators_share": 0,
            "conservator_innovation_share": 0.1,
            "priming_strength": 0.4,
            "inverse_frequency_exponent": [0, 1],
            "inverse_frequency_max_multiplier": 2,
            "priming_opportunity": [ 0.01, 0.05, 0.4 ],
            "decay_strength": 0.5,
            "decay_to": model.enums.DecayTo.BASE_RATE,
            "affects_base_rate": model.enums.AffectsBaseRate.RECEPTION,
            "base_rate_change_strength": [ 0.01 ],
            "allow_decay_stop": False,
            "activation_cap": False,
            "base_rate_update_mechanism": model.enums.BaseRateUpdateMechanism.DEKKER,
            "replicator_selection_sway": [ 0.01, 0.05, 0.1 ],
        },
        child_params={
            "entrenchment": {
                "inverse_frequency_exponent": 0,
                "use_activation": False,
            },
            "regular": {}
        }
    )
}
