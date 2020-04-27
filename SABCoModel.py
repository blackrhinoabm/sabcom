import pandas as pd
import networkx as nx
import json
import os

from src.environment import Environment
from src.runner import runner
import time


start = time.time()
data_folder = 'measurement/baseline/'

# load parameters
with open('parameters/baseline/parameters.json') as json_file:
    parameters = json.load(json_file)

# load neighbourhood data
with open('parameters/baseline/district_data.json') as json_file:
    neighbourhood_data = json.load(json_file)

# load travel matrix
travel_matrix = pd.read_csv('input_data/Travel_Probability_Matrix.csv', index_col=0)

# load age data
age_distribution = pd.read_csv('input_data/age_dist.csv', sep=';', index_col=0)
age_distribution_per_ward = dict(age_distribution.transpose())

# load distance_matrix
distance_matrix = pd.read_csv('parameters/baseline/distance_matrix.csv', index_col=0)

# Monte Carlo simulations
for seed in range(parameters['monte_carlo_runs']):
    # make new folder for seed, if it does not exist
    if not os.path.exists('{}seed{}'.format(data_folder, seed)):
        os.makedirs('{}seed{}'.format(data_folder, seed))

    # initialization
    environment = Environment(seed, parameters, neighbourhood_data, age_distribution_per_ward, distance_matrix)

    # running the simulation
    runner(environment, seed, data_output=parameters["data_output"], data_folder=data_folder, travel_matrix=travel_matrix)

    # save network
    if not parameters["high_performance"]:
        for idx, network in enumerate(environment.infection_states):
            for i, node in enumerate(network.nodes):
                network.nodes[i]['agent'] = network.nodes[i]['agent'].status

            idx_string = '{0:04}'.format(idx)
            nx.write_graphml_lxml(network, "{}seed{}/network_time{}.graphml".format(data_folder, seed, idx_string))

end = time.time()
print(end - start)
