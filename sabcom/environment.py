import numpy as np
import networkx as nx
import random
import copy
import os
import pandas as pd
import scipy.stats as stats

from sabcom.agent import Agent
from sabcom.helpers import what_informality


class Environment:
    """
    The environment class contains the agents in a network structure
    """

    def __init__(self, seed, parameters, district_data, age_distribution_per_district,
                 household_contact_matrix, other_contact_matrix, household_size_distribution, travel_matrix):
        """
        This method initialises the environment and its properties.

        :param seed: used to initialise the random generators to ensure reproducibility, int
        :param parameters: contains all model parameters, dictionary
        :param district_data: contains empirical data on the districts, list of tuples
        :param age_distribution_per_district: contains the distribution across age categories per district, dictionary
        :param household_contact_matrix: contains number and age groups for household contacts, Pandas DataFrame
        :param other_contact_matrix: contains number and age groups for all other contacts, Pandas DataFrame
        :param household_size_distribution: contains distribution of household size for all districts, Pandas DataFrame
        :param travel_matrix: contains number and age groups for all other contacts, Pandas DataFrame
        """
        np.random.seed(seed)
        random.seed(seed)
        random.seed(seed)

        self.parameters = parameters
        self.other_contact_matrix = other_contact_matrix

        # 1 create modelled districts
        # retrieve population data
        nbd_values = [x[1] for x in district_data]
        population_per_neighbourhood = [x['Population'] for x in nbd_values]

        # 1.1 correct the population in districts to be proportional to number of agents
        correction_factor = sum(population_per_neighbourhood) / parameters["number_of_agents"]
        corrected_populations = [int(round(x / correction_factor)) for x in population_per_neighbourhood]

        # 1.2 only count districts that then have an amount of people bigger than 0
        indices_big_neighbourhoods = [i for i, x in enumerate(corrected_populations) if x > 0]
        corrected_populations_final = [x for i, x in enumerate(corrected_populations) if x > 0]

        # 1.3 create a shock generator for the initialisation of agents initial compliance
        lower, upper = -(parameters['stringency_index'][0] / 100), (1 - (parameters['stringency_index'][0] / 100))
        mu, sigma = 0.0, parameters['private_shock_stdev']
        shocks = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma,
                                     size=sum(corrected_populations_final))

        # 1.4 fill up the districts with agents
        self.districts = [x[0] for x in district_data]
        self.district_agents = {d: [] for d in self.districts}
        agents = []
        city_graph = nx.Graph()
        agent_name = 0
        all_travel_districts = {district_data[idx][0]: [] for idx in indices_big_neighbourhoods}

        # for every district
        for num_agents, idx in zip(corrected_populations_final, indices_big_neighbourhoods):
            # 1.5.1 determine district code, informality, and age categories
            district_list = []
            district_code = district_data[idx][0]
            informality = what_informality(district_code, district_data) * parameters["informality_dummy"]

            age_categories = np.random.choice(age_distribution_per_district[district_code].index,
                                              size=int(num_agents),
                                              replace=True,
                                              p=age_distribution_per_district[district_code].values)

            # 1.5.2 determine districts to travel to
            available_districts = list(all_travel_districts.keys())
            probabilities = list(travel_matrix[[str(x) for x in available_districts]].loc[district_code])

            # 1.5.3 add agents to district
            for a in range(num_agents):
                init_private_signal = parameters['stringency_index'][0] / 100 + shocks[agent_name]
                district_to_travel_to = np.random.choice(available_districts, size=1, p=probabilities)[0]
                agent = Agent(agent_name, 's',
                              district_code,
                              age_categories[a],
                              informality,
                              int(round(other_contact_matrix.loc[age_categories[a]].sum())),
                              district_to_travel_to,
                              init_private_signal
                              )
                self.district_agents[district_code].append(agent)
                district_list.append(agent)
                all_travel_districts[district_to_travel_to].append(agent)
                agent_name += 1

            # 2 Create the household network structure
            # 2.1 get household size list for this Ward and reduce list to max household size = size of ward
            max_district_household = min(len(district_list), len(household_size_distribution.columns) - 1)
            hh_sizes = household_size_distribution.loc[district_code][:max_district_household]
            # 2.2 then calculate probabilities of this being of a certain size
            hh_probability = pd.Series([float(i) / sum(hh_sizes) for i in hh_sizes])
            hh_probability.index = hh_sizes.index
            # 2.3 determine household sizes
            sizes = []
            while sum(sizes) < len(district_list):
                sizes.append(int(np.random.choice(hh_probability.index, size=1, p=hh_probability)[0]))
                hh_probability = hh_probability[:len(district_list) - sum(sizes)]
                # recalculate probabilities
                hh_probability = pd.Series([float(i) / sum(hh_probability) for i in hh_probability])
                try:
                    hh_probability.index = hh_sizes.index[:len(district_list) - sum(sizes)]
                except:
                    print('Error occured')
                    print('lenght of district list = {}'.format(len(district_list)))
                    print('sum(sizes) = {}'.format(sum(sizes)))
                    print('hh_sizes.index[:len(district_list) - sum(sizes)]is '.format(hh_sizes.index[:len(district_list) - sum(sizes)]))
                    print('hh_probability.index = {}'.format(hh_probability.index))
                    break

            # 2.4 Distribute agents over households
            # 2.4.1 pick the household heads and let it form connections with other based on probabilities.
            # household heads are chosen at random without replacement
            household_heads = np.random.choice(district_list, size=len(sizes), replace=False)
            not_household_heads = [x for x in district_list if x not in household_heads]
            # 2.4.2 let the household heads pick n other agents that are not household heads themselves
            for i, head in enumerate(household_heads):
                head.household_number = i
                if sizes[i] > 1:
                    # pick n other agents based on probability given their age
                    p = [household_contact_matrix[to.age_group].loc[head.age_group] for to in not_household_heads]
                    # normalize p
                    p = [float(i) / sum(p) for i in p]
                    household_members = list(np.random.choice(not_household_heads, size=sizes[i]-1, replace=False, p=p))

                    # remove household members from not_household_heads
                    for h in household_members:
                        h.household_number = i
                        not_household_heads.remove(h)

                    # add head to household members:
                    household_members.append(head)
                else:
                    household_members = [head]

                # 2.4.3 create graph for household
                household_graph = nx.Graph()
                household_graph.add_nodes_from(range(len(household_members)))

                # create edges between all household members
                edges = nx.complete_graph(len(household_members)).edges()
                household_graph.add_edges_from(edges, label='household')

                # add household members to the agent list
                agents.append(household_members)

                # 2.4.4 add network to city graph
                city_graph = nx.disjoint_union(city_graph, household_graph)

        self.agents = [y for x in agents for y in x]

        # 3 Next, we create the a city wide network structure of recurring contacts based on the travel matrix
        for agent in self.agents:
            agents_to_travel_to = all_travel_districts[agent.district_to_travel_to]
            agents_to_travel_to.remove(agent)  # remove the agent itself

            if agents_to_travel_to:
                # select the agents which it is most likely to have contact with based on the travel matrix
                p = [other_contact_matrix[a.age_group].loc[agent.age_group] for a in agents_to_travel_to]
                # normalize p
                p = [float(i) / sum(p) for i in p]

                location_closest_agents = np.random.choice(agents_to_travel_to,
                                                           size=min(agent.num_contacts, len(agents_to_travel_to)),
                                                           replace=False,
                                                           p=p)

                for ca in location_closest_agents:
                    city_graph.add_edge(agent.name, ca.name, label='other')

        self.network = city_graph

        # rename agents to reflect their new position
        for idx, agent in enumerate(self.agents):
            agent.name = idx

        # 5 add agent to the network structure
        for idx, agent in enumerate(self.agents):
            self.network.nodes[idx]['agent'] = agent

        self.infection_states = []
        self.infection_quantities = {key: [] for key in ['e', 's', 'i1', 'i2', 'c', 'r', 'd', 'compliance', 'contacts']}

        # 6 add stringency index from parameters to reflect how strict regulations are enforced
        self.stringency_index = parameters['stringency_index']
        if len(parameters['stringency_index']) < parameters['time']:
            self.stringency_index += [parameters['stringency_index'][-1] for x in range(len(
                parameters['stringency_index']), parameters['time'])]

    def store_network(self):
        """Returns a deep copy of the current network"""
        current_network = copy.deepcopy(self.network)
        return current_network

    def write_status_location(self, period, seed, base_folder='output_data'):
        """
        Writes information about the agents and their status in the current period to a csv file

        :param period: the current time period, int
        :param seed: used to initialise the random generators to ensure reproducibility, int
        :param base_folder: the location of the folder to write the csv to, string
        :return: None
        """
        location_status_data = {'agent': [], 'status': [], 'WardID': [], 'age_group': [],
                                'others_infected': [], 'compliance': []}
        for agent in self.agents:
            location_status_data['agent'].append(agent.name)
            location_status_data['status'].append(agent.status)
            location_status_data['WardID'].append(agent.district)
            location_status_data['age_group'].append(agent.age_group)
            location_status_data['others_infected'].append(agent.others_infected)
            location_status_data['compliance'].append(agent.compliance)

        pd.DataFrame(location_status_data).to_csv(os.path.join(base_folder, "seed{}".format(seed)) + "/agent_data{0:04}.csv".format(
            period))

        # output links
        if period == 0:
            pd.DataFrame(self.network.edges()).to_csv(base_folder + "seed" + str(seed) + "/edge_list{0:04}.csv".format(
                period))
