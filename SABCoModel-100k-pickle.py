import pandas as pd
import networkx as nx
import json
import os

import pickle

from src.environment import Environment
from src.runner import runner
from src.helpers import what_informality
import time

# set scenario using the following parameter:
scenarios = ['baseline', 'lockdown', 'ineffective_lockdown']
scenario = scenarios[1]  # set scenario here

start = time.time()
data_folder = 'measurement/baseline_100k/'

# 1 load general the parameters
with open('parameters/parameters.json') as json_file:
    parameters = json.load(json_file)

# 1.1 optionally set monte carlo runs quickly here
#parameters['monte_carlo_runs'] = 5

# 1.2 set scenario specific parameters
if scenario == 'baseline':
    parameters["lockdown_days"] = [None for x in range(len(parameters['lockdown_days']))]
elif scenario == 'lockdown':
    parameters['informality_dummy'] = 0.0
    parameters["lockdown_days"] = [x for x in range(len(parameters['lockdown_days']))]
elif scenario == 'ineffective_lockdown':
    parameters['informality_dummy'] = 1.0
    parameters["lockdown_days"] = [x for x in range(len(parameters['lockdown_days']))]


# Change parameters depending on experiment
age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
              'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

parameters['data_output'] = 'csv_light'
parameters['probability_transmission'] = 0.3

# 2 load district data
# 2.1 general neighbourhood data
with open('parameters/district_data_100k.json') as json_file:
    neighbourhood_data = json.load(json_file)

# 2.2 age data
age_distribution = pd.read_csv('input_data/age_dist.csv', sep=';', index_col=0)
age_distribution_per_ward = dict(age_distribution.transpose())

# 2.3 household size distribution
HH_size_distribution = pd.read_excel('input_data/HH_Size_Distribution.xlsx', index_col=0)

# 3 load travel matrix
travel_matrix = pd.read_csv('input_data/Travel_Probability_Matrix.csv', index_col=0)

# 4 load contact matrices
# 4.1 load household contact matrix
hh_contact_matrix = pd.read_excel('input_data/ContactMatrices_10year.xlsx', sheet_name="Home", index_col=0)
# add a col & row for 80 plus. Rename columns to mathc our age categories
hh_contact_matrix['80plus'] = hh_contact_matrix['70_80']
row = hh_contact_matrix.xs('70_80')
row.name = '80plus'
hh_contact_matrix = hh_contact_matrix.append(row)
hh_contact_matrix.columns = age_groups
hh_contact_matrix.index = age_groups

# 4.2 load other contact matrix
other_contact_matrix = pd.read_excel('input_data/ContactMatrices_10year.xlsx', sheet_name="OutsideOfHome", index_col=0)
other_contact_matrix['80plus'] = other_contact_matrix['70_80']
row = other_contact_matrix.xs('70_80')
row.name = '80plus'
other_contact_matrix = other_contact_matrix.append(row)
other_contact_matrix.columns = age_groups
other_contact_matrix.index = age_groups


# Monte Carlo simulations
for seed in range(parameters['monte_carlo_runs']):
    # make new folder for seed, if it does not exist
    if not os.path.exists('{}seed{}'.format(data_folder, seed)):
        os.makedirs('{}seed{}'.format(data_folder, seed))

    # initialization from pickle
    data = open('seed_22_forjoeri.pkl', "rb")
    list_of_objects = pickle.load(data)
    environment = list_of_objects[0]

    # environment = Environment(seed, parameters, neighbourhood_data, age_distribution_per_ward,
    #                           hh_contact_matrix, other_contact_matrix, HH_size_distribution, travel_matrix)

    environment.parameters['time'] = parameters['time']

    # correct time
    environment.parameters['time'] = parameters['time']

    # correct informality
    if parameters["informality_dummy"] == 1.0:
        for agent in environment.agents:
            agent.informality = what_informality(agent.district, neighbourhood_data) * parameters["informality_dummy"]

    environment.parameters['probability_transmission'] = parameters['probability_transmission']  # !!!!###!!!###
    environment.parameters['number_of_agents'] = 100000  # !!!!###!!!###
    environment.parameters['data_output'] = parameters['data_output']  # !!!!###!!!###
    environment.parameters["total_initial_infections"] = parameters["total_initial_infections"]
    environment.parameters['informality_dummy'] = parameters["informality_dummy"]
    environment.parameters["lockdown_days"] = parameters["lockdown_days"]
    environment.parameters["health_system_capacity"] = parameters["health_system_capacity"]


    # running the simulation
    environment = runner(environment, seed, data_output=parameters["data_output"], data_folder=data_folder,
                         calculate_r_naught=False)

    # save network
    if parameters["data_output"] == 'network':
        for idx, network in enumerate(environment.infection_states):
            for i, node in enumerate(network.nodes):
                network.nodes[i]['agent'] = network.nodes[i]['agent'].status

            idx_string = '{0:04}'.format(idx)
            nx.write_graphml(network, "{}seed{}/network_time{}.graphml".format(data_folder, seed, idx_string))
    elif parameters["data_output"] == 'csv_light':
        pd.DataFrame(environment.infection_quantities).to_csv('{}seed{}/quantities_state_time.csv'.format(data_folder,
                                                                                                          seed))

end = time.time()
print(end - start)
