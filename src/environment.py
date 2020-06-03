import numpy as np
from src.agent import Agent
from src.helpers import what_coordinates, what_informality
import networkx as nx
import random
import copy
import pandas as pd


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
        :param other_contact_matrix: contains number and age groups for all other contacts, Pandas DataFrame
        """
        np.random.seed(seed)
        random.seed(seed)

        self.parameters = parameters
        self.other_contact_matrix = other_contact_matrix

        # 1 initialise city districts
        # 1.1 retrieve population data
        nbd_values = [x[1] for x in district_data]
        population_per_neighbourhood = [x['Population'] for x in nbd_values]

        # 1.2 correct the population in districts to be proportional to number of agents
        correction_factor = sum(population_per_neighbourhood) / parameters["number_of_agents"]
        corrected_populations = [round(x / correction_factor) for x in population_per_neighbourhood]

        # 1.3 only count districts that then have an amount of people bigger than 0
        indices_big_neighbourhoods = [i for i, x in enumerate(corrected_populations) if x > 0]
        corrected_populations_final = [x for i, x in enumerate(corrected_populations) if x > 0]

        # 2 fill up the districts with agents
        agents = []
        city_graph = nx.Graph()
        agent_name = 0
        all_travel_districts = {district_data[idx][0]: [] for idx in indices_big_neighbourhoods}
        for num_agents, idx in zip(corrected_populations_final, indices_big_neighbourhoods):
            # 2.1 determine district code, informality, and age categories
            district_list = []
            district_code = district_data[idx][0]
            coordinates = what_coordinates(district_code, district_data)
            informality = what_informality(district_code, district_data) * parameters["informality_dummy"]

            age_categories = np.random.choice(age_distribution_per_district[district_code].index,
                                              size=num_agents,
                                              replace=True,
                                              p=age_distribution_per_district[district_code].values)

            # 2.2 determine district to travel to
            available_districts = list(all_travel_districts.keys())
            probabilities = list(travel_matrix[[str(x) for x in available_districts]].loc[district_code])
            district_to_travel_to = np.random.choice(available_districts, size=1, p=probabilities)[0]

            # 2.3 add agents to neighbourhood
            for a in range(num_agents):
                agent = Agent(agent_name, 's',
                              parameters["probability_transmission"],
                              parameters["probability_susceptible"],
                              coordinates,
                              district_code,
                              age_categories[a],
                              informality,
                              parameters["probability_symptomatic"],
                              parameters['probability_critical'][age_categories[a]],
                              parameters['probability_to_die'][age_categories[a]],
                              int(round(other_contact_matrix.loc[age_categories[a]].sum())),
                              district_to_travel_to
                              )
                district_list.append(agent)
                all_travel_districts[district_to_travel_to].append(agent)
                agent_name += 1

            # 3 Create the household network structure
            # 3.1 get household size list for this Ward and reduce list to max household size = size of ward
            max_district_household = min(len(district_list), len(household_size_distribution.columns) - 1)
            hh_sizes = household_size_distribution.loc[district_code][:max_district_household]
            # 3.2 then calculate probabilities of this being of a certain size
            hh_probability = pd.Series([float(i) / sum(hh_sizes) for i in hh_sizes])
            hh_probability.index = hh_sizes.index
            # 3.3 determine household sizes
            sizes = []
            while sum(sizes) < len(district_list):
                sizes.append(int(np.random.choice(hh_probability.index, size=1, p=hh_probability)[0]))
                hh_probability = hh_probability[:len(district_list) - sum(sizes)]
                # recalculate probabilities
                hh_probability = pd.Series([float(i) / sum(hh_probability) for i in hh_probability])
                hh_probability.index = hh_sizes.index[:len(district_list) - sum(sizes)]

            # 3.4 Distribute agents over households
            # 3.4.1 pick the household heads and let it form connections with other based on probabilities.
            # household heads are chosen at random without replacement
            household_heads = np.random.choice(district_list, size=len(sizes), replace=False)
            not_household_heads = [x for x in district_list if x not in household_heads]
            # 3.4.2 let the household heads pick n other agents that are not household heads themselves
            for i, head in enumerate(household_heads):
                if sizes[i] > 1:
                    # pick n other agents based on probability given their age
                    p = [household_contact_matrix[to.age_group].loc[head.age_group] for to in not_household_heads]
                    # normalize p
                    p = [float(i) / sum(p) for i in p]
                    household_members = list(np.random.choice(not_household_heads, size=sizes[i]-1, replace=False, p=p))

                    # remove household members from not_household_heads
                    for h in household_members:
                        not_household_heads.remove(h)

                    # add head to household members:
                    household_members.append(head)
                else:
                    household_members = [head]

                # 3.4.3 create graph for household
                household_graph = nx.Graph()
                household_graph.add_nodes_from(range(len(household_members)))

                # create edges between all household members
                edges = nx.complete_graph(len(household_members)).edges()
                household_graph.add_edges_from(edges)

                # add household members to the agent list
                agents.append(household_members)

                # 3.4.4 add network to city graph
                city_graph = nx.disjoint_union(city_graph, household_graph)

        self.districts = [x[0] for x in district_data]
        self.district_agents = {d: a for d, a in zip(self.districts, agents)}
        self.agents = [y for x in agents for y in x]

        # 4 Next, we create the a city wide network structure of recurring contacts
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
                    city_graph.add_edge(agent.name, ca.name)

        self.network = city_graph

        # rename agents to reflect their new position
        for idx, agent in enumerate(self.agents):
            agent.name = idx

        # 5 Initialize the probability that a new infected agent appears in every district
        cases = [x[1]['Cases_With_Subdistricts'] for x in district_data]
        self.probabilities_new_infection_district = [float(i) / sum(cases) for i in cases]

        # add agent to the network structure
        for idx, agent in enumerate(self.agents):
            self.network.nodes[idx]['agent'] = agent

        self.infection_states = []
        self.infection_quantities = {key: [] for key in ['e', 's', 'i1', 'i2', 'c', 'r', 'd']}

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

        pd.DataFrame(location_status_data).to_csv(base_folder + "seed" + str(seed) + "/agent_data{0:04}.csv".format(
            period))

        # output links
        pd.DataFrame(self.network.edges()).to_csv(base_folder + "seed" + str(seed) + "/edge_list{0:04}.csv".format(
            period))
