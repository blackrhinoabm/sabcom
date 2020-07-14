import pandas as pd
import networkx as nx
import json, os, re

import pickle,time

from src.runner import runner
from src.helpers import what_informality

start = time.time()

# set scenario using the following parameter:
scenarios = ['no_intervention',"formal_lockdown","ineffective_lockdown"]
scenario=scenarios[0] 

data_folder = 'measurement/'+scenario+'/'
initial_infections = pd.read_csv('input_data/Cases_With_Subdistricts.csv', index_col=0)

# Monte Carlo simulations
monte_carlo_runs=20
for seed in range(monte_carlo_runs):

    environment_folder='measurement/env_pickls/'
    for subdir, dirs, files in os.walk('./'+environment_folder):
        for file in sorted(files):
            x = re.findall(r'\d+',file)
            if x[0]==str(seed):
                filename = file
                seed = x[0]
                data = open(environment_folder+filename, "rb")
                list_of_objects = pickle.load(data)
                list_of_objects=[list_of_objects[0], list_of_objects[1]]
    environment = list_of_objects[0]
    seed=int(seed)

    # 1.0 set time and output format
    environment.parameters['time'] = 350
    environment.parameters['data_output'] = 'csv_light'

    # 1.1 general neighbourhood data
    with open('parameters/district_data_100k.json') as json_file:
        neighbourhood_data = json.load(json_file)

    # 1.2 set scenario specific parameters
    if scenario == 'no_intervention':
        environment.parameters['likelihood_awareness'] = [0.0 for x in environment.parameters['likelihood_awareness']]
        environment.parameters['visiting_recurring_contacts_multiplier'] = [1.0 for x in environment.parameters['visiting_recurring_contacts_multiplier']]
        environment.parameters['gathering_max_contacts'] = [float('inf') for x in environment.parameters['gathering_max_contacts']]
        environment.parameters['physical_distancing_multiplier'] = [1.0 for x in environment.parameters['physical_distancing_multiplier']]
        environment.parameters['informality_dummy'] = 0.0
        for agent in environment.agents:
            agent.informality = what_informality(agent.district, neighbourhood_data) * environment.parameters["informality_dummy"]

    elif scenario == 'lockdown':
        environment.parameters['informality_dummy'] = 0.0
        for agent in environment.agents:
            agent.informality = what_informality(agent.district, neighbourhood_data) * environment.parameters["informality_dummy"]

    elif scenario == 'ineffective_lockdown':
        environment.parameters['informality_dummy'] = 1.0
        for agent in environment.agents:
            agent.informality = what_informality(agent.district, neighbourhood_data) * environment.parameters["informality_dummy"]

    # make new folder for seed, if it does not exist
    if not os.path.exists('{}seed{}'.format(data_folder, seed)):
        os.makedirs('{}seed{}'.format(data_folder, seed))

    environment = runner(environment, initial_infections, seed, data_output=environment.parameters["data_output"], data_folder=data_folder,calculate_r_naught=False)

    #save csv light and network
    if environment.parameters["data_output"] == 'network':
        for idx, network in enumerate(environment.infection_states):
            for i, node in enumerate(network.nodes):
                network.nodes[i]['agent'] = network.nodes[i]['agent'].status
            idx_string = '{0:04}'.format(idx)
            nx.write_graphml(network, "{}seed{}/network_time{}.graphml".format(data_folder, seed, idx_string))
    elif environment.parameters["data_output"] == 'csv_light':
        pd.DataFrame(environment.infection_quantities).to_csv('{}seed{}/quantities_state_time.csv'.format(data_folder,
                                                                                                          seed))
end = time.time()
hours_total, rem_total = divmod(end-start, 3600)
minutes_total, seconds_total = divmod(rem_total, 60)
print("TOTAL RUNTIME","{:0>2}:{:0>2}:{:05.2f}".format(int(hours_total),int(minutes_total),seconds_total))
