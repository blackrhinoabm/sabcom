
import pandas as pd
import networkx as nx
import json, pickle, itertools
from src.environment import Environment 
from src.runner import runner 
import sys, os, time
 
start = time.time()

# set number of seeds
seeds = list(range(50))  
simulations=['50seeds_initialisation']

#when parallel in SLURM/HPC
if sys.argv[1]=='slurm-cluster':
    parameter_set = list(itertools.product(simulations, seeds))  #This gives a list with parameter combinations
    pos = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    tupl= parameter_set[pos] 
    sim=tupl[0] #assign simulation
    seed=tupl[1] #assign seed

    # set folder names for storage:
    data_folder = 'measurement/'+sim +'/'
    data_folder_environment = data_folder+'env_pickls/'
else:
    data_folder = 'measurement/'+simulations[0] +'/'
    data_folder_environment = data_folder+'env_pickls/'   
    seed=22 #assigns random seed if not run parallel on cluster

# 1 load general the parameters
with open('parameters/parameters.json') as json_file:
    parameters = json.load(json_file)

# Change parameters depending on experiment
parameters['data_output'] = 'csv_light'

age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
              'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

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

# make new folder for seed, if it does not exist
if not os.path.exists('{}seed{}'.format(data_folder, seed)):
    os.makedirs('{}seed{}'.format(data_folder, seed))
if not os.path.exists('{}'.format(data_folder_environment)):
    os.makedirs('{}'.format(data_folder_environment))

# initialisation
environment = Environment(seed, parameters, neighbourhood_data, age_distribution_per_ward,
                          hh_contact_matrix, other_contact_matrix, HH_size_distribution, travel_matrix)

# save environment objects as pickls
file_name = data_folder_environment + "seed_" + str(seed) + '.pkl'
save_objects = open(file_name, 'wb')
pickle.dump([environment, seed], save_objects)
save_objects.close()

# running the simulation
environment = runner(environment, seed, data_output=parameters["data_output"], data_folder=data_folder,calculate_r_naught=False)

if parameters["data_output"] == 'network':
    for idx, network in enumerate(environment.infection_states):
        for i, node in enumerate(network.nodes):
            network.nodes[i]['agent'] = network.nodes[i]['agent'].status
        idx_string = '{0:04}'.format(idx)
        nx.write_graphml(network, "{}seed{}/network_time{}.graphml".format(data_folder, seed, idx_string))
elif parameters["data_output"] == 'csv_light':
    pd.DataFrame(environment.infection_quantities).to_csv('{}seed{}/quantities_state_time.csv'.format(data_folder,seed))


end = time.time()
hours_total, rem_total = divmod(end-start, 3600)
minutes_total, seconds_total = divmod(rem_total, 60)
print("TOTAL RUNTIME","{:0>2}:{:0>2}:{:05.2f}".format(int(hours_total),int(minutes_total),seconds_total))
