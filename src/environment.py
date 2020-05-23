import numpy as np
from src.agent import Agent
from src.helpers import edges_to_remove_neighbourhood, what_coordinates, what_informality
import networkx as nx
import random
import copy
import pandas as pd


class Environment:
    """
    The environment class contains the agents in a network structure
    """

    def __init__(self, seed, parameters, district_data, age_distribution_per_district, distance_matrix,
                 hh_contact_matrix, other_contact_matrix, HH_size_distribution, travel_matrix):
        """
        This method initialises the environment and its properties.

        :param seed: used to initialise the random generators to ensure reproducibility, int
        :param parameters: contains all model parameters, dictionary
        :param district_data: contains empirical data on the districts, list of tuples
        :param age_distribution_per_district: contains the distribution across age categories per district, dictionary
        :param distance_matrix: contains distances between all districts, Pandas Dataframe
        """
        np.random.seed(seed)
        random.seed(seed)

        self.parameters = parameters
        self.other_contact_matrix = other_contact_matrix # store the contact matrix for use in the runner

        # sort data
        nbd_values = [x[1] for x in district_data]
        population_per_neighbourhood = [x['Population'] for x in nbd_values]

        # correct the population in neighbourhoods to be proportional to number of agents
        correction_factor = sum(population_per_neighbourhood) / parameters["number_of_agents"]
        corrected_populations = [round(x / correction_factor) for x in population_per_neighbourhood]

        # only count neighbourhoods that then have an amount of people bigger than 0
        indices_big_neighbourhoods = [i for i, x in enumerate(corrected_populations) if x > 0]
        corrected_populations_final = [x for i, x in enumerate(corrected_populations) if x > 0]

        # calculate correct density per district
        corrected_density_per_neighbourhood = [x['Density'] for i, x in enumerate(nbd_values) if
                                               i in indices_big_neighbourhoods]
        corrected_density_per_neighbourhood = [(float(i) / max(corrected_density_per_neighbourhood)) for i in
                                               corrected_density_per_neighbourhood]

        agents = []
        city_graph = nx.Graph()
        agent_name = 0
        for num_agents, idx in zip(corrected_populations_final, indices_big_neighbourhoods):
            district_list = []
            district_code = district_data[idx][0]
            coordinates = what_coordinates(district_code, district_data)
            informality = what_informality(district_code, district_data) * parameters["informality_dummy"]
            density = corrected_density_per_neighbourhood[idx]

            age_categories = np.random.choice(age_distribution_per_district[district_code].index,
                                              size=num_agents,
                                              replace=True,
                                              p=age_distribution_per_district[district_code].values)

            # add agents to neighbourhood
            for a in range(num_agents):
                district_list.append(Agent(agent_name, 's',
                                           parameters["probability_transmission"],
                                           parameters["probability_susceptible"],
                                           1.0 - parameters['percentage_contacts_recurring'],
                                           coordinates,
                                           district_code,
                                           age_categories[a],
                                           informality,
                                           parameters["probability_symptomatic"],
                                           parameters['probability_critical'][age_categories[a]],
                                           parameters['probability_to_die'][age_categories[a]],
                                           int(round(other_contact_matrix.loc[age_categories[a]].sum())) # contacts are calibrated using contact matrix
                                           ))
                agent_name += 1

            # Create the household network structure
            # 1 get household size list for this Ward and reduce list to max household size = size of ward
            max_district_household = min(len(district_list), len(HH_size_distribution.columns) - 1)
            hh_sizes = HH_size_distribution.loc[district_code][:max_district_household]
            # 2 then calculate probabilities of this being of a certain size
            hh_probability = pd.Series([float(i) / sum(hh_sizes) for i in hh_sizes])
            hh_probability.index = hh_sizes.index
            # 3 determine household sizes
            sizes = []
            while sum(sizes) < len(district_list):
                sizes.append(int(np.random.choice(hh_probability.index, size=1, p=hh_probability)[0]))
                hh_probability = hh_probability[:len(district_list) - sum(sizes)]
                # recalculate probabilities
                hh_probability = pd.Series([float(i) / sum(hh_probability) for i in hh_probability])
                hh_probability.index = hh_sizes.index[:len(district_list) - sum(sizes)]

            # To form the household...
            # (1) pick the household heads and let it form connections with other based on probabilities.
            # household heads are chosen at random without replacement
            household_heads = np.random.choice(district_list, size=len(sizes), replace=False)
            not_household_heads = [x for x in district_list if x not in household_heads]
            # let the household heads pick n other agents that are not household heads themselves
            for idx, head in enumerate(household_heads):
                if sizes[idx] > 1:
                    # pick n other agents based on probability given their age
                    p = [hh_contact_matrix[to.age_group].loc[head.age_group] for to in not_household_heads]
                    # normalize p
                    p = [float(i) / sum(p) for i in p]
                    household_members = list(np.random.choice(not_household_heads, size=sizes[idx]-1, replace=False, p=p))

                    # remove household members from not_household_heads
                    for h in household_members:
                        not_household_heads.remove(h)

                    # add head to household members:
                    household_members.append(head)
                else:
                    household_members = [head]

                # create graph for household
                HG = nx.Graph()
                HG.add_nodes_from(range(len(household_members)))

                # create edges between all household members
                edges = nx.complete_graph(len(household_members)).edges()
                HG.add_edges_from(edges)

                # add household members to the agent list
                agents.append(household_members)

                # add network to city graph
                city_graph = nx.disjoint_union(city_graph, HG)

        self.districts = [x[0] for x in district_data]
        self.district_agents = {d: a for d, a in zip(self.districts, agents)}
        self.agents = [y for x in agents for y in x]

        # Next, we create the a city wide network structure of recurring contacts
        for agent in self.agents:
            for contact in range(round(agent.num_trips * parameters['percentage_contacts_recurring'])):
                probabilities = list(travel_matrix.loc[agent.district])
                district_to_travel_to = np.random.choice(self.districts, size=1, p=probabilities)[0]
                agents_to_travel_to = self.district_agents[district_to_travel_to]

                # consider there are no viable options to travel to ... travel to multiple agents
                if agents_to_travel_to:
                    # select the agent which it is most likely to have contact with based on the travel matrix
                    p = [other_contact_matrix[a.age_group].loc[agent.age_group] for a in
                             agents_to_travel_to]
                    # normalize p
                    p = [float(i) / sum(p) for i in p]
                    location_closest_agent = np.random.choice(agents_to_travel_to, size=1, p=p)[0].name

                    # create edge to that agent and store that edge in the city graph
                    city_graph.add_edge(agent.name, location_closest_agent)

        self.network = city_graph
        self.distance_matrix = distance_matrix

        # rename agents to reflect their new position
        for idx, agent in enumerate(self.agents):
            agent.name = idx

        # Initialize the probability that a new infected agent appears in every district
        cases = [x[1]['Cases_With_Subdistricts'] for x in district_data]
        self.probabilities_new_infection_district = [float(i) / sum(cases) for i in cases]

        # add agent to the network structure
        for idx, agent in enumerate(self.agents):
            self.network.nodes[idx]['agent'] = agent

        self.infection_states = []
        self.infection_quantities = {key: [] for key in ['e', 's', 'i1', 'i2', 'c', 'r', 'd']}

    def show(self):
        """ Uses the network x draw function to draw the status of the current network"""
        nx.draw(self.network, with_labels=True, font_weight='bold')

    def store_network(self):
        """Returns a deep copy of the current network"""
        current_network = copy.deepcopy(self.network)
        return current_network

    def write_status_location(self, period, seed, base_folder='measurement/'):
        """
        Writes information about the agents and their status in the current period to a csv file

        :param period: the current time period, int
        :param seed: used to initialise the random generators to ensure reproducibility, int
        :param base_folder: the location of the folder to write the csv to, string
        :return: None
        """
        location_status_data = {'agent': [], 'lon': [], 'lat': [], 'status': [],
                                'WardID': [], 'age_group': [], 'others_infected': []}
        for agent in self.agents:
            location_status_data['agent'].append(agent.name)
            location_status_data['lon'].append(agent.coordinates[0])
            location_status_data['lat'].append(agent.coordinates[1])
            location_status_data['status'].append(agent.status)
            location_status_data['WardID'].append(agent.district)
            location_status_data['age_group'].append(agent.age_group)
            location_status_data['others_infected'].append(agent.others_infected)

        pd.DataFrame(location_status_data).to_csv(base_folder + "seed" + str(seed) + "/agent_data{0:04}.csv".format(period))

        # output links
        pd.DataFrame(self.network.edges()).to_csv(base_folder + "seed" + str(seed) + "/edge_list{0:04}.csv".format(period))


class EnvironmentMeanField:
    """
    Currently depreciated.
    The environment class contains the agents in (1) a random network structure, furthermore, agents all make
    (2) the same amount of trips
    """

    def __init__(self, seed, parameters, district_data, age_distribution_per_district, distance_matrix,
                 hh_contact_matrix, other_contact_matrix, HH_size_distribution, variation=None):
        """
        This method initialises the environment and its properties.

        :param seed: used to initialise the random generators to ensure reproducibility, int
        :param parameters: contains all model parameters, dictionary
        :param district_data: contains empirical data on the districts, list of tuples
        :param age_distribution_per_district: contains the distribution across age categories per district, dictionary
        :param distance_matrix: contains distances between all districts, Pandas Dataframe
        """
        np.random.seed(seed)
        random.seed(seed)

        self.parameters = parameters

        # sort data
        nbd_values = [x[1] for x in district_data]
        population_per_neighbourhood = [x['Population'] for x in nbd_values]

        # correct the population in neighbourhoods to be proportional to number of agents
        correction_factor = sum(population_per_neighbourhood) / parameters["number_of_agents"]
        corrected_populations = [round(x / correction_factor) for x in population_per_neighbourhood]

        # TODO this is specific to mean field model
        corrected_populations_mean_field = []
        for x in corrected_populations:
            if (x % 2) == 0:
                corrected_populations_mean_field.append(x)
            else:
                corrected_populations_mean_field.append(x + random.choice([1, -1]))

        corrected_populations = corrected_populations_mean_field

        # only count neighbourhoods that then have an amount of people bigger than 0
        indices_big_neighbourhoods = [i for i, x in enumerate(corrected_populations) if x > 0]
        corrected_populations_final = [x for i, x in enumerate(corrected_populations) if x > 0]

        # calculate correct density per district
        corrected_density_per_neighbourhood = [x['Density'] for i, x in enumerate(nbd_values) if
                                               i in indices_big_neighbourhoods]
        corrected_density_per_neighbourhood = [(float(i) / max(corrected_density_per_neighbourhood)) for i in
                                               corrected_density_per_neighbourhood]

        agents = []
        city_graph = nx.Graph()
        agent_name = 0
        for num_agents, idx in zip(corrected_populations_final, indices_big_neighbourhoods):
            district_list = []
            district_code = district_data[idx][0]
            coordinates = what_coordinates(district_code, district_data)
            informality = what_informality(district_code, district_data) * parameters["informality_dummy"]
            density = corrected_density_per_neighbourhood[idx]

            age_categories = np.random.choice(age_distribution_per_district[district_code].index,
                                              size=num_agents,
                                              replace=True,
                                              p=age_distribution_per_district[district_code].values)

            # this is specific to the meanfield version of the model
            average_travel = int(round(other_contact_matrix.sum().mean()))

            # add agents to neighbourhood
            for a in range(num_agents):
                if variation == 'poissontravel':
                    average_travel = round(np.random.poisson(4))
                district_list.append(Agent(agent_name, 's',
                                           parameters["probability_transmission"],
                                           parameters["probability_susceptible"],
                                           parameters["probability_to_travel"],
                                           coordinates,
                                           district_code,
                                           age_categories[a],
                                           informality,
                                           parameters["probability_symptomatic"],
                                           parameters['probability_critical'][age_categories[a]],
                                           parameters['probability_to_die'][age_categories[a]],
                                           average_travel
                                           ))
                agent_name += 1

            # create a household graph with equally distributed household sizes TODO this is the unique feature of this initialiser
            # 1 get household size list for this Ward and reduce list to max household size = size of ward
            hh_sizes = HH_size_distribution.loc[district_code][:len(district_list)]
            # 2 then calculate probabilities of this being of a certain size
            hh_probability = pd.Series([float(i) / sum(hh_sizes) for i in hh_sizes])
            hh_probability.index = hh_sizes.index
            # 3 determine household sizes
            sizes = []
            while sum(sizes) < len(district_list):
                sizes.append(int(np.random.choice(hh_probability.index, size=1, p=hh_probability)[0]))
                hh_probability = hh_probability[:len(district_list) - sum(sizes)]
                # recalculate probabilities
                hh_probability = pd.Series([float(i) / sum(hh_probability) for i in hh_probability])
                hh_probability.index = hh_sizes.index[:len(district_list) - sum(sizes)]

            # To form the household...
            # (1) pick the household heads and let it form connections with other based on probabilities.
            # household heads are chosen at random without replacement
            household_heads = np.random.choice(district_list, size=len(sizes), replace=False)
            not_household_heads = [x for x in district_list if x not in household_heads]
            # let the household heads pick n other agents that are not household heads themselves
            for idx, head in enumerate(household_heads):
                if sizes[idx] > 1:
                    # pick n other agents based on probability given their age
                    p = [hh_contact_matrix[to.age_group].loc[head.age_group] for to in not_household_heads]
                    # normalize p
                    p = [float(i) / sum(p) for i in p]
                    household_members = list(np.random.choice(not_household_heads, size=sizes[idx]-1, replace=False, p=p))

                    # remove household members from not_household_heads
                    for h in household_members:
                        not_household_heads.remove(h)

                    # add head to household members:
                    household_members.append(head)
                else:
                    household_members = [head]

                # create graph for household
                HG = nx.Graph()
                HG.add_nodes_from(range(len(household_members)))

                # create edges between all household members
                edges = nx.complete_graph(len(household_members)).edges()
                HG.add_edges_from(edges)

                # add household members to the agent list
                agents.append(household_members)

                # add network to city graph
                city_graph = nx.disjoint_union(city_graph, HG)
            # first create a Barabasi Albert graph for the ward and use it to calculate average degree
            # nodes = len(district_list)
            # new_edges = 2
            # BA = nx.barabasi_albert_graph(nodes, new_edges, seed=0)
            #
            # NG = nx.gnm_random_graph(nodes, len(BA.edges)) #nx.random_regular_graph(degree, nodes, seed=0) #len(NG.edges)
            #
            # if variation == 'BAsocial':
            #     NG = BA
            #
            # print('BA edges = ', len(BA.edges))
            # print('NG edges = ', len(NG.edges))
            #
            # edges = list(NG.edges)
            # # reduce the amount of edges in the district depending on its empirical density
            # for e in edges_to_remove_neighbourhood(edges, density, list(NG.nodes)):
            #     NG.remove_edge(e[0], e[1])
            #
            # # add the district agents to the agent list
            # agents.append(district_list)
            #
            # # add network to city graph
            # city_graph = nx.disjoint_union(city_graph, NG)

        self.network = city_graph
        self.distance_matrix = distance_matrix
        self.districts = [x[0] for x in district_data]

        self.district_agents = {d: a for d, a in zip(self.districts, agents)}
        self.agents = [y for x in agents for y in x]

        # Initialize the probability that a new infected agent appears in every district
        cases = [x[1]['Cases_With_Subdistricts'] for x in district_data]
        self.probabilities_new_infection_district = [float(i) / sum(cases) for i in cases]

        # add agent to the network structure
        for idx, agent in enumerate(self.agents):
            self.network.nodes[idx]['agent'] = agent

        self.infection_states = []
        self.infection_quantities = {key: [] for key in ['e', 's', 'i1', 'i2', 'c', 'r', 'd']}

    def show(self):
        """ Uses the network x draw function to draw the status of the current network"""
        nx.draw(self.network, with_labels=True, font_weight='bold')

    def store_network(self):
        """Returns a deep copy of the current network"""
        current_network = copy.deepcopy(self.network)
        return current_network

    def write_status_location(self, period, seed, base_folder='measurement/'):
        """
        Writes information about the agents and their status in the current period to a csv file

        :param period: the current time period, int
        :param seed: used to initialise the random generators to ensure reproducibility, int
        :param base_folder: the location of the folder to write the csv to, string
        :return: None
        """
        location_status_data = {'agent': [], 'lon': [], 'lat': [], 'status': [],
                                'WardID': [], 'age_group': [], 'others_infected': []}
        for agent in self.agents:
            location_status_data['agent'].append(agent.name)
            location_status_data['lon'].append(agent.coordinates[0])
            location_status_data['lat'].append(agent.coordinates[1])
            location_status_data['status'].append(agent.status)
            location_status_data['WardID'].append(agent.district)
            location_status_data['age_group'].append(agent.age_group)
            location_status_data['others_infected'].append(agent.others_infected)

        pd.DataFrame(location_status_data).to_csv(base_folder + "seed" + str(seed) + "/agent_data{0:04}.csv".format(period))

        # output links
        pd.DataFrame(self.network.edges()).to_csv(
            base_folder + "seed" + str(seed) + "/edge_list{0:04}.csv".format(period))
