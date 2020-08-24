import random
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
import scipy.stats as stats


def edge_in_cliq(edge, nodes_in_cliq):
    if edge[0] in nodes_in_cliq:
        return True
    else:
        return False


def edges_to_remove_neighbourhood(all_edges, neighbourhood_density, nbh_nodes):
    neighbourhood_edges = [e for e in all_edges if edge_in_cliq(e, nbh_nodes)]
    sample_size = int(len(neighbourhood_edges) * (1-neighbourhood_density))
    # sample random edges
    chosen_edges = random.sample(neighbourhood_edges, sample_size)
    return chosen_edges


def what_neighbourhood(index, neighbourhood_nodes):
    for n in neighbourhood_nodes:
        if index in neighbourhood_nodes[n]:
            return n

    raise ValueError('Neighbourhood not found.')


def what_coordinates(neighbourhood_name, dataset):
    for x in range(len(dataset)):
        if neighbourhood_name in dataset[x]:
            return dataset[x][1]['lon'], dataset[x][1]['lat'],

    raise ValueError("Corresponding coordinates not found")


def what_informality(neighbourhood_name, dataset):
    for x in range(len(dataset)):
        if neighbourhood_name in dataset[x]:
            try:
                return dataset[x][1]['Informal_residential']
            except:
                return None

    raise ValueError("Corresponding informality not found")


def confidence_interval(data, av):
    sample_stdev = np.std(data)
    sigma = sample_stdev/math.sqrt(len(data))
    return stats.t.interval(alpha=0.95, df=24, loc=av, scale=sigma)


def generate_district_data(number_of_agents, path, max_districts=None):
    """
    Transforms input data on informal residential, initial infections, and population and transforms it to
    a list of organised data for the simulation.

    :param number_of_agents: number of agents in the simulation, integer
    :param max_districts: (optional) maximum amount of districts simulated, integer
    :return: data set containing district data for simulation, list
    """
    informal_residential = pd.read_csv('{}/f_informality.csv'.format(path))#.iloc[:-1]
    inital_infections = pd.read_csv('{}/f_initial_cases.csv'.format(path), index_col=1)
    inital_infections = inital_infections.sort_index()
    population = pd.read_csv('{}/f_population.csv'.format(path))

    # normalise district informality
    x = informal_residential[['Informal_residential']].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    informal_residential['Informal_residential'] = pd.DataFrame(x_scaled)
    population['Informal_residential'] = informal_residential['Informal_residential']

    # determine smallest district based on number of agents
    smallest_size = population['Population'].sum() / number_of_agents

    # generate data set for model input
    districts_data = []
    for i in range(len(population)):
        if population['Population'].iloc[i] > smallest_size:
            districts_data.append(
                [int(population['WardID'].iloc[i]), {'Population': population['Population'].iloc[i],
                                                     #'lon': population['lon'].iloc[i],
                                                     #'lat': population['lat'].iloc[i],
                                                     'Informal_residential': population['Informal_residential'].iloc[i],
                                                     'Cases_With_Subdistricts':
                                                         inital_infections.loc[population['WardID'].iloc[i]][
                                                             'Cases'],
                                                     },
                 ])

    if max_districts is None:
        max_districts = len(districts_data)  # this can be manually shortened to study dynamics in some districts

    return districts_data[:max_districts]
