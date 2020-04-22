import numpy as np
from src.agent import NetworkAgent, Agent
from src.helpers import edges_to_remove_neighbourhood, what_neighbourhood, what_coordinates, what_informality
import networkx as nx
import random
import copy
import pandas as pd


class EnvironmentNetwork:
    """
    The environment class contains the agents in a network structure
    """

    def __init__(self, seed, parameters, neighbourhood_data, age_distribution_per_ward):
        np.random.seed(seed)
        random.seed(seed)

        self.parameters = parameters

        # sort data
        nbd_values = [x[1] for x in neighbourhood_data]
        nbd_keys = [x[0] for x in neighbourhood_data]
        population_per_neighbourhood = [x['Population'] for x in nbd_values]

        # correct the population in neighbourhoods to be proportional to number of agents
        correction_factor = sum(population_per_neighbourhood) / parameters["number_of_agents"]
        corrected_populations = [int(x / correction_factor) for x in population_per_neighbourhood]

        # only count neighbourhoods that then have an amount of people bigger than 3
        indices_big_neighbourhoods = [i for i, x in enumerate(corrected_populations) if x > 0]
        corrected_populations_final = [x for i, x in enumerate(corrected_populations) if x > 0]

        # create a cave network that has max nodes equal to the biggest neighbourhood
        max_citizens = max(corrected_populations_final)
        self.network = nx.caveman_graph(len(corrected_populations_final), max_citizens)

        # find cliques and associate with neighbourhoods that are big enough
        neighbourhoods = [x for i, x in enumerate(nbd_keys) if i in indices_big_neighbourhoods]
        cliques = list(nx.find_cliques(self.network))
        neighbourhood_nodes = {ne: cl for ne, cl in zip(neighbourhoods, cliques)}

        # reduce the amount of nodes in each clique to reflect the size of the neighbourhood
        empirical_nodes_per_neighbourhood = {key: value for key, value in
                                             zip(neighbourhoods, corrected_populations_final)}
        for n in neighbourhoods:
            # remove the last nodes from list
            for nd in neighbourhood_nodes[n][empirical_nodes_per_neighbourhood[n]:]:
                self.network.remove_node(nd)

        # update cliques
        cliques = list(nx.find_cliques(self.network))

        # Next, reduce the amount of edges to reflect the density of the neighbourhood
        corrected_density_per_neighbourhood = [x['Density'] for i, x in enumerate(nbd_values) if
                                               i in indices_big_neighbourhoods]
        highest_density = parameters["highest_density_neighbourhood"]
        density_scores = [(float(i) / max(corrected_density_per_neighbourhood)) * highest_density for i in
                          corrected_density_per_neighbourhood]
        density_score_per_neighbourhood = {key: value for key, value in zip(neighbourhoods, density_scores)}
        edges = list(self.network.edges)

        for n in neighbourhoods:
            # actually remove chosen edges
            for e in edges_to_remove_neighbourhood(edges, density_score_per_neighbourhood[n], neighbourhood_nodes[n]):
                self.network.remove_edge(e[0], e[1])

        # update neighbourhood nodes to reflect some nodes have been removed
        neighbourhood_nodes = {ne: cl for ne, cl in zip(neighbourhoods, cliques)}

        # finally re-label the nodes & edges to 1, 2 ...
        mapping = {key: value for key, value in zip(self.network.nodes, range(len(self.network.nodes)))}
        self.network = nx.relabel_nodes(self.network, mapping, copy=False)

        # and once update neighbourhood nodes to reflect the new labels
        for neighb in neighbourhood_nodes:
            neighbourhood_nodes[neighb] = [mapping[x] for x in neighbourhood_nodes[neighb]]

        # TODO per neighbourhood, create a list
        age_list = []
        for neighbourhood in neighbourhoods:
            age_categories = np.random.choice(age_distribution_per_ward[neighbourhood].index,
                                              size=len(neighbourhood_nodes[neighbourhood]),
                                              replace=True,
                                              p=age_distribution_per_ward[neighbourhood].values)
            age_list.append(age_categories)
        # flatten list
        age_list = [y for x in age_list for y in x]

        # Next, create the agents TODO add realistic probability
        self.agents = [NetworkAgent(x, 's', parameters["probability_transmission"],
                                    parameters["probability_susceptible"], parameters["probability_to_travel"]
                                    ) for x in range(len(self.network.nodes))]

        # add agent to the network structure
        for idx, agent in enumerate(self.agents):
            self.network.nodes[idx]['agent'] = agent
            agent.neighbourhood = what_neighbourhood(idx, neighbourhood_nodes)
            agent.coordinates = what_coordinates(agent.neighbourhood, neighbourhood_data)
            agent.age_group = age_list[idx]
            agent.prob_hospital = parameters['probability_critical'][agent.age_group]
            agent.prob_death = parameters['probability_to_die'][agent.age_group]
            agent.informality = what_informality(agent.neighbourhood, neighbourhood_data)

        self.infection_states = []

    def show(self):
        nx.draw(self.network, with_labels=True, font_weight='bold')

    def store_network(self):
        current_network = copy.deepcopy(self.network)
        return current_network

    def write_status_location(self, period, seed, base_folder='measurement/'):
        location_status_data = {'agent': [], 'lon': [], 'lat': [], 'status': [],
                                'WardID': [], 'age_group': [], 'others_infected': []}
        for agent in self.agents:
            location_status_data['agent'].append(agent.name)
            location_status_data['lon'].append(agent.coordinates[0])
            location_status_data['lat'].append(agent.coordinates[1])
            location_status_data['status'].append(agent.status)
            location_status_data['WardID'].append(agent.neighbourhood)
            location_status_data['age_group'].append(agent.age_group)
            location_status_data['others_infected'].append(agent.others_infected)

        pd.DataFrame(location_status_data).to_csv(base_folder + "seed" + str(seed) + "/agent_data{0:04}.csv".format(period))


class Environment:
    """
    The environment class contains the agents in a network structure
    """

    def __init__(self, seed, parameters, neighbourhood_data, age_distribution_per_ward):
        np.random.seed(seed)
        random.seed(seed)

        self.parameters = parameters

        # sort data
        nbd_values = [x[1] for x in neighbourhood_data]
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
            district_code = neighbourhood_data[idx][0]
            coordinates = what_coordinates(district_code, neighbourhood_data)
            informality = what_informality(district_code, neighbourhood_data)
            density = corrected_density_per_neighbourhood[idx]

            age_categories = np.random.choice(age_distribution_per_ward[district_code].index,
                                              size=num_agents,
                                              replace=True,
                                              p=age_distribution_per_ward[district_code].values)

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
                                           parameters['probability_critical'][age_categories[a]],
                                           parameters['probability_to_die'][age_categories[a]]
                                           ))
                agent_name += 1

            # create a Barabasi Albert graph for the ward
            nodes = len(district_list)
            if nodes <= 4:
                new_edges = nodes - 1
            else:
                new_edges = 4
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

        self.agents = [y for x in agents for y in x]

        # add agent to the network structure
        for idx, agent in enumerate(self.agents):
            self.network.nodes[idx]['agent'] = agent

        self.infection_states = []

    def show(self):
        nx.draw(self.network, with_labels=True, font_weight='bold')

    def store_network(self):
        current_network = copy.deepcopy(self.network)
        return current_network

    def write_status_location(self, period, seed, base_folder='measurement/'):
        location_status_data = {'agent': [], 'lon': [], 'lat': [], 'status': [],
                                'WardID': [], 'age_group': [], 'others_infected': []}
        for agent in self.agents:
            location_status_data['agent'].append(agent.name)
            location_status_data['lon'].append(agent.coordinates[0])
            location_status_data['lat'].append(agent.coordinates[1])
            location_status_data['status'].append(agent.status)
            location_status_data['WardID'].append(agent.neighbourhood)
            location_status_data['age_group'].append(agent.age_group)
            location_status_data['others_infected'].append(agent.others_infected)

        pd.DataFrame(location_status_data).to_csv(base_folder + "seed" + str(seed) + "/agent_data{0:04}.csv".format(period))
