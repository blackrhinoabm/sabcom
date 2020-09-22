
class Agent:
    def __init__(self, name, status, district, age_group,
                 informality, number_contacts, district_to_travel_to, compliance):
        """
        This method initialises an agent and its properties.

        Agents have several properties, that can either be state variables (if they change),
        or static parameters. There is a distinction between agent specific parameters
        and parameters that are the same for all agents and could thus be seen
        as global parameters.

        The following inputs are used to initialise the agent
        :param name: unique agent identifier , integer
        :param status: initial disease status, string
        :param coordinates: the geographical coordinates of where the agent lives, tuple (float, float)
        :param district: the unique code / identifier of the district, int
        :param age_group: the age group of the agent, float ([age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
              'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus')
        :param informality: a percentage indicating how 'informal' the district the agent lives in is, float (0,1)
        :param number_contacts: the amount of trips the agent will undertake on a daily basis
        """
        # state variables
        self.status = status

        # agent specific parameters
        self.name = name
        self.district = district
        self.age_group = age_group

        # agent specific variables
        self.compliance = compliance
        self.previous_compliance = compliance

        # implementation variables
        self.sick_days = 0
        self.asymptomatic_days = 0
        self.incubation_days = 0
        self.critical_days = 0
        self.exposed_days = 0
        self.days_recovered = 0
        self.others_infected = 0
        self.others_infects_total = 0
        self.travel_neighbours = []
        self.period_to_become_infected = None

        # implementation parameters
        self.num_contacts = number_contacts
        self.household_number = None

        # agent specific parameters that depend on other parameters
        self.district_to_travel_to = district_to_travel_to
        self.informality = informality

    def __repr__(self):
        """
        :return: String representation of the trader
        """
        return self.status + ' Agent' + str(self.name)
