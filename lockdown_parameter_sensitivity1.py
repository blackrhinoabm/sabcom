import pandas as pd
import networkx as nx
import numpy as np
from src.helpers import confidence_interval
import json
import os

from src.environment import Environment, EnvironmentMeanField
from src.runner import runner, runner_mean_field
import time


start = time.time()
data_folder = 'measurement/'

# load parameters
with open('parameters/parameters.json') as json_file:
    parameters = json.load(json_file)

# Change parameters depending on experiment
age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
              'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

parameters['data_output'] = 'network'

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

# set experiment parameters to neutral:
parameters["lockdown_days"] = [x for x in range(0, parameters['time'])]
# (1) physical distancing measures such as increased hygiëne & face mask adoption
parameters["physical_distancing_multiplier"] = 1.0
# (2) reducing travel e.g. by reducing it for work, school or all
parameters["travel_restrictions_multiplier"] = {key:value for key, value in zip(age_groups,
                                                                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])}
# (3) reducing close contacts
parameters["visiting_close_contacts_multiplier"] = 1.0 # depending on how strict the lockdown is at keeping you put.
# (4) Testing and general awareness
parameters['likelihood_awareness'] = 0.0 # this will be increased through track & trace and coviid
parameters['self_isolation_multiplier'] = 1.0 # determines the percentage of connections cut thanks to self-isoluation can go up with coviid
parameters['aware_status'] = ['i2'] # i1 can be added if there is large scale testing
# (5) limiting mass contact e.g. forbidding large events
parameters["gathering_max_contacts"] = 5000

# set policy parameters here
experiment_values = [x / 10 for x in range(0, 11, 1)]
parameters["monte_carlo_runs"] = 2
# set this parameter to True to run the meanfield version of the model instead
mean_field = False

# run the experiment
pd_experiment = {}
for experiment in experiment_values:
    print('experiment: ', experiment)
    parameters["physical_distancing_multiplier"] = experiment
    # print('experiment with value: ', parameters["physical_distancing_multiplier"])
    baseline_summary_stats = []
    # Monte Carlo simulations
    for seed in range(parameters['monte_carlo_runs']):
        # make new folder for seed, if it does not exist
        if not os.path.exists('{}seed{}'.format(data_folder, seed)):
            os.makedirs('{}seed{}'.format(data_folder, seed))

        # initialization
        if mean_field:
            data_name = 'meanfield'
            environment = EnvironmentMeanField(seed, parameters, neighbourhood_data, age_distribution_per_ward, distance_matrix)
        else:
            data_name = 'SABCoM'
            environment = Environment(seed, parameters, neighbourhood_data, age_distribution_per_ward, distance_matrix)

        # running the simulation
        if mean_field:
            runner_mean_field(environment, seed, data_output=parameters["data_output"], data_folder=data_folder,
                              travel_matrix=travel_matrix, verbose=False)
        else:
            runner(environment, seed, data_output=parameters["data_output"], data_folder=data_folder,
                   travel_matrix=travel_matrix, verbose=False)

        # saving the network data
        for idx, network in enumerate(environment.infection_states):
            for i, node in enumerate(network.nodes):
                network.nodes[i]['agent'] = network.nodes[i]['agent'].status

        susceptible_ot = []
        infected_1_ot = []
        infected_2_ot = []
        critical_ot = []
        dead_ot = []
        recovered_ot = []
        exposed_ot = []

        for t in range(parameters['time']):
            network = environment.infection_states[t]
            susceptible = 0
            infected_1 = 0
            infected_2 = 0
            critical = 0
            dead = 0
            recovered = 0
            exposed = 0
            for idx, node in enumerate(network):
                if network.nodes[idx]['agent'] == 's':
                    susceptible += 1
                elif network.nodes[idx]['agent'] == 'e':
                    exposed += 1
                elif network.nodes[idx]['agent'] == 'i1':
                    infected_1 += 1
                elif network.nodes[idx]['agent'] == 'i2':
                    infected_2 += 1
                elif network.nodes[idx]['agent'] == 'c':
                    critical += 1
                elif network.nodes[idx]['agent'] == 'd':
                    dead += 1
                elif network.nodes[idx]['agent'] == 'r':
                    recovered += 1
                else:
                    print('no status?')

            susceptible_ot.append((susceptible / float(len(network))))
            infected_1_ot.append((infected_1 / float(len(network))))
            infected_2_ot.append((infected_2 / float(len(network))))
            critical_ot.append((critical / float(len(network))))
            dead_ot.append((dead / float(len(network))))
            recovered_ot.append((recovered / float(len(network))))
            exposed_ot.append((exposed / float(len(network))))

        # save output data
        baseline_summary_stats.append(
            {'total dead': dead_ot[-1], 'peak critical': max(critical_ot), 'total recovered': recovered_ot[-1]})
    # add outcome to dictionary
    total_dead_baseline = [baseline_summary_stats[x]['total dead'] for x in range(len(baseline_summary_stats))]
    peak_critical_baseline = [baseline_summary_stats[x]['peak critical'] for x in range(len(baseline_summary_stats))]
    total_infected_baseline = [baseline_summary_stats[x]['total dead'] + baseline_summary_stats[x]['total recovered']
                               for x in range(len(baseline_summary_stats))]

    simulation_summary = pd.DataFrame({
        'total dead': [np.mean(total_dead_baseline),
                       confidence_interval(total_dead_baseline, np.mean(total_dead_baseline))[0],
                       confidence_interval(total_dead_baseline, np.mean(total_dead_baseline))[1]],
        'peak critical': [np.mean(peak_critical_baseline),
                          confidence_interval(peak_critical_baseline, np.mean(peak_critical_baseline))[0],
                          confidence_interval(peak_critical_baseline, np.mean(peak_critical_baseline))[1]],
        'total infected': [np.mean(total_infected_baseline),
                           confidence_interval(total_infected_baseline, np.mean(total_infected_baseline))[0],
                           confidence_interval(total_infected_baseline, np.mean(total_infected_baseline))[1]]
    }).transpose()
    simulation_summary.columns = ['average', 'lower', 'upper']

    pd_experiment[experiment] = simulation_summary

av_total_infected = [pd_experiment[x]['average'].loc['total infected'] for x in experiment_values]
up_total_infected = [pd_experiment[x]['upper'].loc['total infected'] for x in experiment_values]
lo_total_infected = [pd_experiment[x]['lower'].loc['total infected'] for x in experiment_values]

av_peak_critical = [pd_experiment[x]['average'].loc['peak critical'] for x in experiment_values]
up_peak_critical = [pd_experiment[x]['upper'].loc['peak critical'] for x in experiment_values]
lo_peak_critical = [pd_experiment[x]['lower'].loc['peak critical'] for x in experiment_values]

av_total_dead = [pd_experiment[x]['average'].loc['total dead'] for x in experiment_values]
up_total_dead = [pd_experiment[x]['upper'].loc['total dead'] for x in experiment_values]
lo_total_dead = [pd_experiment[x]['lower'].loc['total dead'] for x in experiment_values]

v = [av_total_infected, up_total_infected, lo_total_infected, av_peak_critical, up_peak_critical, lo_peak_critical,
     av_total_dead, up_total_dead, lo_total_dead]
k = ['av_total_infected', 'up_total_infected', 'lo_total_infected', 'av_peak_critical', 'up_peak_critical',
     'lo_peak_critical', 'av_total_dead', 'up_total_dead', 'lo_total_dead']

model_output = pd.DataFrame({key: value for key, value in zip(k, v)})
model_output.index = experiment_values
model_output.to_csv('{}experiment1_{}.csv'.format(data_folder, data_name))
