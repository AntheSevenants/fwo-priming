class StartingProbabilities:
    EQUAL = 0
    RANDOM = 1

class DecayTo:
    UNIFORM_DIST = 0
    STARTING_DIST = 1
    BASE_RATE = 2

class AffectsBaseRate:
    NOTHING = 0b00  # 0
    RECEPTION = 0b01  # 1
    PRODUCTION = 0b10  # 2
    ALL = 0b11  # 3

    @classmethod
    def affects_reception(cls, value):
        return (value & cls.RECEPTION) != 0

    @classmethod
    def affects_production(cls, value):
        return (value & cls.PRODUCTION) != 0