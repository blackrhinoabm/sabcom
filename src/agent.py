
class NetworkAgent:
    def __init__(self, name, status, prob_transmission, prob_susceptible, prob_travel):
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
        self.others_infected = 0

        # parameters
        self.name = name
        self.coordinates = None
        self.neighbourhood = None
        self.age_group = None

        # these are technically global parameters because they are not unique in the current implementation of the model
        self.prob_transmission = prob_transmission  # not unique in current implementation
        self.prob_hospital = None  # not unique in current implementation
        self.prob_death = None  # not unique in current implementation
        self.prob_susceptible = prob_susceptible  # not unique in current implementation
        self.prob_travel = prob_travel  # not unique in current implementation

    def __repr__(self):
        """
        :return: String representation of the trader
        """
        return self.status + ' Agent' + str(self.name)


class Agent:
    def __init__(self, name, status, prob_transmission, prob_susceptible, prob_travel,
                 coordinates, neighbourhood, age_group, informality,
                 probability_critical, probability_death):
        """
        Agents have two properties, one variable and one parameter
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
        self.others_infected = 0
        self.travel_neighbours = []

        # parameters
        self.name = name
        self.coordinates = coordinates
        self.neighbourhood = neighbourhood
        self.age_group = age_group
        self.informality = informality
        self.prob_hospital = probability_critical
        self.prob_death = probability_death

        # these are technically global parameters because they are not unique in the current implementation of the model
        self.prob_transmission = prob_transmission  # not unique in current implementation
        self.prob_susceptible = prob_susceptible  # not unique in current implementation
        self.prob_travel = prob_travel  # not unique in current implementation

    def __repr__(self):
        """
        :return: String representation of the trader
        """
        return self.status + ' Agent' + str(self.name)