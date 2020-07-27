import pandas as pd
import json
import pickle
import itertools
import sys
import os
import time

from environment import Environment
from helpers import generate_district_data

# This script works best run on a high performance cluster computer (HPC), where it wil initialise multiple seeds
# in that case run in the console:
#
# python initialisation.py 'slurm-cluster'
#
# it will initialise only a single seed when run on a normal computer because of long run time.

start = time.time()

cities = ['cape_town']

# set these parameters before initialising a seed
SIMULATION_TIME = 350
MONTE_CARLO_RUNS = 20
CITY = cities[0]
SEED = 22

# set number of seeds
seeds = list(range(MONTE_CARLO_RUNS))
simulations = ['50_seeds_initialisation']


# The following lines detect whether you are working on HPC with the SLURM scheduler TODO update this
try:
    arg = sys.argv[1]
except:
    arg = None

if arg == 'slurm-cluster':
    parameter_set = list(itertools.product(simulations, seeds))  # This gives a list with parameter combinations
    pos = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    tupl = parameter_set[pos]
    sim = tupl[0]  # assign simulation
    seed = tupl[1]  # assign seed

    # set folder names for storage:
    data_folder = 'initialisations/' + sim + '/'
    data_folder_environment = data_folder + 'env_pickls/'
else:
    data_folder_environment = 'sabcom/initialisations/' + CITY
    seed = SEED

# 1 load general the parameters
with open('parameters/parameters_{}.json'.format(CITY)) as json_file:
    parameters = json.load(json_file)

# Change parameters depending on experiment
parameters['data_output'] = 'csv_light'

age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
              'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

# 2 load district data
# transform input data to general district data for simulations
district_data = generate_district_data(int(parameters['number_of_agents']))

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

# make new folder if it does not exist
if not os.path.exists('{}'.format(data_folder_environment)):
    os.makedirs('{}'.format(data_folder_environment))

# initialisation
environment = Environment(seed, parameters, district_data, age_distribution_per_ward,
                          hh_contact_matrix, other_contact_matrix, HH_size_distribution, travel_matrix)

# save environment objects as pickls
file_name = data_folder_environment + "seed_" + str(seed) + '.pkl'
save_objects = open(file_name, 'wb')
pickle.dump([environment, seed], save_objects)
save_objects.close()

end = time.time()
hours_total, rem_total = divmod(end-start, 3600)
minutes_total, seconds_total = divmod(rem_total, 60)
print("TOTAL RUNTIME", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours_total), int(minutes_total), seconds_total))
