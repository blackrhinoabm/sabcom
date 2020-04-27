import pandas as pd
import networkx as nx
import json
import os

from src.environment import Environment
from src.runner import runner
import time


start = time.time()

age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
              'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

parameters = {
    # general simulation parameters
    "time": 90,
    "number_of_agents": 25,
    "monte_carlo_runs": 1,
    "data_output": False,
    # specific simulation parameters
    "incubation_days": 5, # average number of days agents are infected but do not have symptoms SOURCE Zhang et al. 2020
    "symptom_days": 10,# average number of days agents have mild symptoms
    "critical_days": 8, # average number of days agents are in critical condition
    "health_system_capacity": 0.0021, # relative (in terms of population) capacity of the hospitals
    "no_hospital_multiplier": 1.79, # the increase in probability if a critical agent cannot go to the hospital SOURCE: Zhou et al. 2020
    "travel_sample_size": 0.02, # amount of agents that an agent might choose to travel to
    "foreign_infection_days": [x for x in range(0, 19)], # days at which 1 agent will be infected every day from abroad
    # agent parameters
    "probability_transmission": 0.30, # should be estimated to replicate realistic R0 number.
    "probability_to_travel": 0.25, # should be estimated to replicate travel data
    "probability_critical": {key:value for key, value in zip(age_groups, [0.001, 0.003, 0.012, 0.032, 0.049, 0.102, 0.166, 0.244, 0.273])}, # probability that an agent enters a critical stage of the disease SOURCE: Verity et al.
    "probability_to_die": {key:value for key, value in zip(age_groups, [0.005, 0.021, 0.053, 0.126, 0.221, 0.303, 0.565, 0.653, 0.765])}, # probability to die per age group in critical stage SOURCE: Verity et al.
    "probability_susceptible": 0.000, # probability that the agent will again be susceptible after having recovered
    # experiment parameter
    "lockdown_days" : [0 for x in range(11, 46)], # in the baseline this is 0, 5 march was the first reported case, 27 march was the start of the lockdown 35 days
    "lockdown_travel_multiplier": 1.0 - ((0.85 + 0.62) / 2), # need estimate for this based on apple travel data reduction of 85% google work of -62% for Western Cape
    "lockdown_infection_multiplier": 1.0,#0.27, # Jarvis et al. 2020
    "informality_dummy": 0.0, # setting this parameter at 0 will mean the lockdown is equally effective anywhere, alternative = 1
    "at_risk_groups": age_groups # list all age groups for baseline
}

districts_data = [[1, {'Population': parameters['number_of_agents'],
                                                 'Density': 1.0,
                                                 'lon': 1.0,
                                                 'lat': 1.0,
                                                 'Informal_residential': 0.0,
                                                 'Cases_With_Subdistricts': 1.0,
                                                }]]

distribution = [0.112314, 0.118867, 0.145951, 0.145413, 0.151773, 0.139329, 0.099140, 0.058729, 0.028484]
age_distribution_per_district = {1: pd.Series(distribution, index=age_groups)}

distance_matrix = {districts_data[0][0]: [0.0]}
distance_matrix = pd.DataFrame(distance_matrix).transpose()
distance_matrix.columns = [districts_data[0][0]]

data_folder = 'measurement/simple/'

# Monte Carlo simulations
for seed in range(parameters['monte_carlo_runs']):
    # make new folder for seed, if it does not exist
    if not os.path.exists('{}seed{}'.format(data_folder, seed)):
        os.makedirs('{}seed{}'.format(data_folder, seed))

    parameters['lockdown_days'] = [x for x in range(0, 46)]
    # initialization
    environment = Environment(0, parameters, districts_data, age_distribution_per_district, distance_matrix)

    # running the simulation
    runner(environment, 0, data_output='csv', data_folder=data_folder)

    # save network
    if parameters["high_performance"] == 'network':
        for idx, network in enumerate(environment.infection_states):
            for i, node in enumerate(network.nodes):
                network.nodes[i]['agent'] = network.nodes[i]['agent'].status

            idx_string = '{0:04}'.format(idx)
            nx.write_graphml_lxml(network, "{}seed{}/network_time{}.graphml".format(data_folder, seed, idx_string))

end = time.time()
print(end - start)