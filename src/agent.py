import numpy as np


class NetworkAgent:
    def __init__(self, name, status, prob_transmission, prob_hospital, prob_death, prob_susceptible, prob_travel):
        """
        Agens have two properties, one variable and one parameter
        Variables:
        status: (i) infected, (s) susceptible, or (r) recovered
        Parameters:
        position: position on grid (row, column)
        """
        # variables
        self.sick_days = 0
        self.incubation_days = 0
        self.critical_days = 0
        self.days_recovered = 0
        self.status = status
        self.neighbourhood = None
        self.coordinates = None

        # parameters
        self.name = name
        self.prob_transmission = prob_transmission
        self.prob_hospital = prob_hospital
        self.prob_death = prob_death
        self.prob_susceptible = prob_susceptible
        self.prob_travel = prob_travel

    def __repr__(self):
        """
        :return: String representation of the trader
        """
        return self.status + ' Agent' + str(self.name)
