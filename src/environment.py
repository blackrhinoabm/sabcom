import numpy as np
from src.agent import NetworkAgent
from src.helpers import edges_to_remove_neighbourhood, what_neighbourhood, what_coordinates
import networkx as nx
import random
import copy
import pandas as pd


class EnvironmentNetwork:
    """
    The environment class contains the agents in a network structure
    """

    def __init__(self, seed, number_agents, prob_transmit, prob_hospital,
                 prob_death, prob_susceptible, prob_travel, neighbourhood_data):
        np.random.seed(seed)
        random.seed(seed)

        # sort data
        nbd_values = [x[1] for x in neighbourhood_data]
        nbd_keys = [x[0] for x in neighbourhood_data]
        population_per_neighbourhood = [x['population'] for x in nbd_values]

        # correct the population in neighbourhoods to be proportional to number of agents
        correction_factor = sum(population_per_neighbourhood) / number_agents
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
        corrected_density_per_neighbourhood = [x['population_KM'] for i, x in enumerate(nbd_values) if
                                               i in indices_big_neighbourhoods]
        highest_density = 0.4 #TODO replace this with parameter input
        density_scores = [(float(i) / max(corrected_density_per_neighbourhood)) * highest_density for i in
                          corrected_density_per_neighbourhood]
        density_score_per_neighbourhood = {key: value for key, value in zip(neighbourhoods, density_scores)}
        edges = list(self.network.edges)

        for n in neighbourhoods:
            # actually remove chosen edges
            for e in edges_to_remove_neighbourhood(edges, density_score_per_neighbourhood[n], neighbourhood_nodes[n]):
                self.network.remove_edge(e[0], e[1])

        # update neighbourhood notes to reflect some nodes have been removed
        neighbourhood_nodes = {ne: cl for ne, cl in zip(neighbourhoods, cliques)}

        # finally re-label the nodes & edges to 1, 2 ...
        mapping = {key: value for key, value in zip(self.network.nodes, range(len(self.network.nodes)))}
        self.network = nx.relabel_nodes(self.network, mapping, copy=False)

        # and once update neighbourhood nodes to reflect the new labels
        for neighb in neighbourhood_nodes:
            neighbourhood_nodes[neighb] = [mapping[x] for x in neighbourhood_nodes[neighb]]

        # Next, create the agents
        self.agents = [NetworkAgent(x, 's', prob_transmit,
                                    prob_hospital, prob_death,
                                    prob_susceptible, prob_travel) for x in range(len(self.network.nodes))]

        # add agent to the network structure
        for idx, agent in enumerate(self.agents):
            self.network.nodes[idx]['agent'] = agent
            agent.neighbourhood = what_neighbourhood(idx, neighbourhood_nodes)
            agent.coordinates = what_coordinates(agent.neighbourhood, neighbourhood_data)

        self.infection_states = []

    def show(self):
        nx.draw(self.network, with_labels=True, font_weight='bold')

    def store_network(self):
        current_network = copy.deepcopy(self.network)
        return current_network

    def write_status_location(self, period):
        location_status_data = {'agent': [], 'lon': [], 'lat': [], 'status': [], 'sp_code': []}
        for agent in self.agents:
            location_status_data['agent'].append(agent.name)
            location_status_data['lon'].append(agent.coordinates[0])
            location_status_data['lat'].append(agent.coordinates[1])
            location_status_data['status'].append(agent.status)
            location_status_data['sp_code'].append(agent.neighbourhood)

        pd.DataFrame(location_status_data).to_csv("output/2agent_data{}.csv".format(period))

