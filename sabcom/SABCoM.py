import pandas as pd
import networkx as nx
import os
import re
import pickle
import time

from runner import runner
from helpers import generate_district_data
from helpers import what_informality

# time how long it takes to run the script
start = time.time()

# define scenarios and cities:
scenarios = ['no_intervention', "formal_lockdown", "ineffective_lockdown"]
cities = ['cape_town']
data_output_modes = ['csv_light', 'csv', 'network']

# set these parameters before running the scenario simulations
SIMULATION_TIME = 350
MONTE_CARLO_RUNS = 2
SCENARIO_NUMBERS = [0, 1, 2]  # include in this list all scenario indices that you want to run
CITY = cities[0]
DATA_OUTPUT_NUMBER = 0

for sc in SCENARIO_NUMBERS:
    # set correct data files
    scenario = scenarios[sc]
    data_folder = 'output_data/{}/'.format(scenario)
    initial_infections = pd.read_csv('input_data/Cases_With_Subdistricts.csv', index_col=0)
    environment_folder = 'initialisations/cape_town/'
    initialisations = [name for name in os.listdir(environment_folder) if '.pkl' in name]
    n_initialisations = len(initialisations)

    # Monte Carlo simulations
    for run in range(min(n_initialisations, MONTE_CARLO_RUNS)):  # simulate min environment files / monte carlo runs
        file = initialisations[run]
        seed = int(re.findall(r'\d+', file)[0])
        data = open(environment_folder + file, "rb")
        environment = pickle.load(data)[0]

        # update time and output format in the environment
        max_time = environment.parameters['time']  # you cannot simulate longer than initialised
        environment.parameters['time'] = min(SIMULATION_TIME, max_time)
        environment.parameters['data_output'] = data_output_modes[DATA_OUTPUT_NUMBER]

        # transform input data to general district data for simulations
        district_data = generate_district_data(environment.parameters['number_of_agents'])

        # set scenario specific parameters
        if scenario == 'no_intervention':
            environment.parameters['likelihood_awareness'] = [
                0.0 for x in environment.parameters['likelihood_awareness']]
            environment.parameters['visiting_recurring_contacts_multiplier'] = [
                1.0 for x in environment.parameters['visiting_recurring_contacts_multiplier']]
            environment.parameters['gathering_max_contacts'] = [
                float('inf') for x in environment.parameters['gathering_max_contacts']]
            environment.parameters['physical_distancing_multiplier'] = [
                1.0 for x in environment.parameters['physical_distancing_multiplier']]
            environment.parameters['informality_dummy'] = 0.0
        elif scenario == 'lockdown':
            environment.parameters['informality_dummy'] = 0.0
        elif scenario == 'ineffective_lockdown':
            environment.parameters['informality_dummy'] = 1.0

        for agent in environment.agents:
            agent.informality = what_informality(agent.district, district_data
                                                 ) * environment.parameters["informality_dummy"]

        # make new folder for seed, if it does not exist
        if not os.path.exists('{}seed{}'.format(data_folder, seed)):
            os.makedirs('{}seed{}'.format(data_folder, seed))

        # simulate the model
        environment = runner(environment, initial_infections, seed, data_output=environment.parameters["data_output"],
                             data_folder=data_folder, calculate_r_naught=False)

        # save csv light or network data
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
print("TOTAL RUNTIME", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours_total), int(minutes_total), seconds_total))
