import pandas as pd
import json
import os

from src.environment import EnvironmentNetwork
from src.runner import Runner


# load parameters
with open('parameters_R0.json') as json_file:
    parameters = json.load(json_file)

# load neighbourhood data
with open('neighbourhood_data_R0.json') as json_file:
    neighbourhood_data = json.load(json_file)

# make new folder for R0
if not os.path.exists('measurement/R0'):
    os.makedirs('measurement/R0')

# load age data
age_distribution = pd.read_csv('age_dist.csv', sep=';', index_col=0)
age_distribution_per_ward = dict(age_distribution.transpose())

# find out initial amount of agents
environment = EnvironmentNetwork(1, parameters, neighbourhood_data, age_distribution_per_ward)
number_agents = len(environment.agents)

# store R0s in list
RZeros = []

# Simulation for every agent
for idx_patient_zero in range(number_agents):
    print(idx_patient_zero)
    # initialization
    environment = EnvironmentNetwork(1, parameters, neighbourhood_data, age_distribution_per_ward)

    # running the simulation
    runner = Runner()

    RZeros.append(runner.calculate_R0(environment, seed=1, idx_patient_zero=idx_patient_zero))

pd.DataFrame(RZeros).to_csv("measurement/R0/RZeros.csv")

