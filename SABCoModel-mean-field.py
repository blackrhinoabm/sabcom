import pandas as pd
import networkx as nx
import json
import os

from src.environment import EnvironmentMeanField
from src.runner import runner_mean_field, runner
import time


start = time.time()
data_folder = 'measurement/baseline/'

# load parameters
with open('parameters/parameters.json') as json_file:
    parameters = json.load(json_file)

# Change parameters depending on experiment
age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
              'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

parameters['data_output'] = 'csv_light'

# load neighbourhood data
with open('parameters/district_data.json') as json_file:
    neighbourhood_data = json.load(json_file)

# load travel matrix
travel_matrix = pd.read_csv('input_data/Travel_Probability_Matrix.csv', index_col=0)

# load age data
age_distribution = pd.read_csv('input_data/age_dist.csv', sep=';', index_col=0)
age_distribution_per_ward = dict(age_distribution.transpose())

# load distance_matrix
distance_matrix = pd.read_csv('parameters/distance_matrix.csv', index_col=0)

# Monte Carlo simulations
for seed in range(parameters['monte_carlo_runs']):
    # make new folder for seed, if it does not exist
    if not os.path.exists('{}seed{}'.format(data_folder, seed)):
        os.makedirs('{}seed{}'.format(data_folder, seed))

    # initialization
    environment = EnvironmentMeanField(seed, parameters, neighbourhood_data, age_distribution_per_ward, distance_matrix)

    # running the simulation
    runner_mean_field(environment, seed, data_output=parameters["data_output"], data_folder=data_folder)

    # save network
    if parameters["data_output"] == 'network':
        for idx, network in enumerate(environment.infection_states):
            for i, node in enumerate(network.nodes):
                network.nodes[i]['agent'] = network.nodes[i]['agent'].status

            idx_string = '{0:04}'.format(idx)
            nx.write_graphml_lxml(network, "{}seed{}/network_time{}.graphml".format(data_folder, seed, idx_string))
    elif parameters["data_output"] == 'csv_light':
        pd.DataFrame(environment.infection_quantities).to_csv('{}seed{}/quantities_state_time.csv'.format(data_folder,
                                                                                                          seed))

end = time.time()
print(end - start)
