import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import time
import ipywidgets as wg
from ipywidgets import interact
from IPython.display import display
import networkx as nx
from src.environment import Environment
from src.runner import runner

import os

age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
              'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

TIME = 60

parameters = {
    # general simulation parameters
    "time": TIME,
    "number_of_agents": 50,
    "monte_carlo_runs": 1,

    # Cape Town specific parameters
    "total_initial_infections": [x for x in range(0, 19)],  # total agents infected in CT
    "health_system_capacity": 0.0021,  # relative (in terms of population) capacity of the hospitals

    # COVID-19 parameters
    "exposed_days": 4,  # average number of days without symptoms and being able to infect others
    "asymptom_days": 10,  # average number of days agents are infected but do not have symptoms
    "symptom_days": 10,  # average number of days agents have mild symptoms
    "critical_days": 8,  # average number of days agents are in critical condition
    "probability_symptomatic": 0.6165,  # determines whether an agent will become asymptomatic or asymptomatic spreader
    "no_hospital_multiplier": 1.79,
    # the increase in probability if a critical agent cannot go to the hospital SOURCE: Zhou et al. 2020
    "probability_transmission": 0.05,  # should be estimated to replicate realistic R0 number.

    "probability_critical": {key: value for key, value in
                             zip(age_groups, [0.001, 0.003, 0.012, 0.032, 0.049, 0.102, 0.166, 0.244, 0.273])},
    # probability that an agent enters a critical stage of the disease SOURCE: Verity et al.
    "probability_to_die": {key: value for key, value in
                           zip(age_groups, [0.005, 0.021, 0.053, 0.126, 0.221, 0.303, 0.565, 0.653, 0.765])},
    # probability to die per age group in critical stage SOURCE: Verity et al.

    # Policy parameters
    "lockdown_days": [None for x in range(0, TIME)],
    # in the baseline this is 0, 5 march was the first reported case, 27 march was the start of the lockdown 35 days
    "informality_dummy": 1.0,
    # setting this parameter at 0 will mean the lockdown is equally effective anywhere, alternative = 1

    # Specific policy parameters
    # (1) physical distancing measures such as increased hygiëne & face mask adoption
    "physical_distancing_multiplier": 0.27,  # Jarvis et al. 2020,
    # (2) reducing travel e.g. by reducing it for work, school or all
    "visiting_recurring_contacts_multiplier": 0.6,  # depending on how strict the lockdown is at keeping you put.
    # (3) Testing and general awareness
    'likelihood_awareness': 0.6,  # this will be increased through track & trace and coviid
    'self_isolation_multiplier': 0.4,
    # determines the percentage of connections cut thanks to self-isoluation can go up with coviid
    'aware_status': ['i2'],  # i1 can be added if there is large scale testing
    # (4) limiting mass contact e.g. forbidding large events
    "gathering_max_contacts": 6,

    # Technical parameters
    'init_infected_agent': 0,  # to calculate R0
    "data_output": 'network',  # 'csv' or 'network', or 'False'

    # Depreciated paramters (can be used later)
    "probability_susceptible": 0.000,  # probability that the agent will again be susceptible after having recovered
}

districts_data = [[1, {'Population': round(parameters['number_of_agents'] / 2),
                                                 'Density': 1.0,
                                                 'lon': 1.0,
                                                 'lat': 1.0,
                                                 'Informal_residential': 0.0,
                                                 'Cases_With_Subdistricts': 0.5,
                                                }],
                 [2, {'Population': round(parameters['number_of_agents'] / 2),
                                                 'Density': 1.0,
                                                 'lon': 2.0,
                                                 'lat': 2.0,
                                                 'Informal_residential': 0.0,
                                                 'Cases_With_Subdistricts': 0.5,
                                                }]]

a_distribution =  [0.112314, 0.118867, 0.145951, 0.145413, 0.151773, 0.139329, 0.099140, 0.058729, 0.028484]

age_distribution_per_district = {1: pd.Series(a_distribution, index=age_groups), 2: pd.Series(a_distribution, index=age_groups)}

travel_matrix = {districts_data[0][0]: [0.5, 0.5], districts_data[1][0]: [0.5, 0.5]}
travel_matrix = pd.DataFrame(travel_matrix).transpose()
travel_matrix.columns = [str(districts_data[0][0]), str(districts_data[1][0])]

age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
              'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

hh_contact_matrix = pd.read_excel('input_data/ContactMatrices_10year.xlsx', sheet_name="Home", index_col=0)
hh_contact_matrix['80plus'] = hh_contact_matrix['70_80']
row = hh_contact_matrix.xs('70_80')
row.name = '80plus'
hh_contact_matrix = hh_contact_matrix.append(row)
hh_contact_matrix.columns = age_groups
hh_contact_matrix.index = age_groups

other_contact_matrix = pd.read_excel('input_data/ContactMatrices_10year.xlsx', sheet_name="OutsideOfHome", index_col=0)
other_contact_matrix['80plus'] = other_contact_matrix['70_80']
row = other_contact_matrix.xs('70_80')
row.name = '80plus'
other_contact_matrix = other_contact_matrix.append(row)

other_contact_matrix.columns = age_groups
other_contact_matrix.index = age_groups

HH_size_distribution = pd.read_excel('input_data/HH_Size_Distribution.xlsx', index_col=0)
HH_size_distribution = HH_size_distribution.iloc[0:2]
HH_size_distribution.index = [1, 2]

data_folder = 'measurement/simple/'

if not os.path.exists('{}seed{}'.format(data_folder, 0)):
    os.makedirs('{}seed{}'.format(data_folder, 0))


environment = Environment(0, parameters, districts_data, age_distribution_per_district,
                          hh_contact_matrix, other_contact_matrix, HH_size_distribution, travel_matrix)
environment = runner(environment, 0, data_output='network', data_folder=data_folder)

