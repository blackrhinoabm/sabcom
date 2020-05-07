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

    def __init__(self, seed, parameters, district_data, age_distribution_per_district, distance_matrix):
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
                                           parameters["probability_to_travel"],
                                           coordinates,
                                           district_code,
                                           age_categories[a],
                                           informality,
                                           parameters["probability_symptomatic"],
                                           parameters['probability_critical'][age_categories[a]],
                                           parameters['probability_to_die'][age_categories[a]],
                                           round(np.random.lognormal(1)) # TODO debug the amount of trips is drawn from a lognormal distribution
                                           ))
                agent_name += 1

            # create a Barabasi Albert graph for the ward
            nodes = len(district_list)
            new_edges = 2
            NG = nx.barabasi_albert_graph(nodes, new_edges, seed=0)

            edges = list(NG.edges)
            # reduce the amount of edges in the district depending on its empirical density
            for e in edges_to_remove_neighbourhood(edges, density, list(NG.nodes)):
                NG.remove_edge(e[0], e[1])

            # add the district agents to the agent list
            agents.append(district_list)

            # add network to city graph
            city_graph = nx.disjoint_union(city_graph, NG)

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
        self.infection_quantities = {key: [] for key in ['e','s', 'i1', 'i2', 'c', 'r', 'd']}

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
    The environment class contains the agents in (1) a random network structure, furthermore, agents all make
    (2) the same amount of trips
    """

    def __init__(self, seed, parameters, district_data, age_distribution_per_district, distance_matrix):
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

            average_travel = int(round(np.mean([round(np.random.lognormal(1)) for x in range(num_agents)])))

            # add agents to neighbourhood
            for a in range(num_agents):
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

            # create a random regular graph TODO this is the unique feature of this initialiser
            # first create a Barabasi Albert graph for the ward and use it to calculate average degree
            nodes = len(district_list)
            new_edges = 2
            BA = nx.barabasi_albert_graph(nodes, new_edges, seed=0)

            degree = round(len(BA.edges) / nodes) * 2 #TODO check if this works

            NG = nx.random_regular_graph(degree, nodes, seed=0) #len(NG.edges)
            print('BA edges = ', len(BA.edges))
            print('NG edges = ', len(NG.edges))

            edges = list(NG.edges)
            # reduce the amount of edges in the district depending on its empirical density
            for e in edges_to_remove_neighbourhood(edges, density, list(NG.nodes)):
                NG.remove_edge(e[0], e[1])

            # add the district agents to the agent list
            agents.append(district_list)

            # add network to city graph
            city_graph = nx.disjoint_union(city_graph, NG)

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
