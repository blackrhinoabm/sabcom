from src.environment import EnvironmentNetwork
from src.runner import Runner
import networkx as nx
import json

import sys
sys.path.append('/scratch/kzltin001/sabcom/')
from SABCoModel import *

# load parameters
with open('parameters.json') as json_file:
    parameters = json.load(json_file)

# load neighbourhood data
with open('neighbourhood_data.json') as json_file:
    neighbourhood_data = json.load(json_file)


# Monte Carlo simulation
pos = int(os.getenv('SLURM_ARRAY_TASK_ID')) 
seed=pos
# initialization
environment = EnvironmentNetwork(seed, parameters, neighbourhood_data)
# running the simulation
runner = Runner()
runner.do_run(environment, seed)

# save network
if not parameters["high_performance"]:
    for idx, network in enumerate(environment.infection_states):
        for i, node in enumerate(network.nodes):
            network.nodes[i]['agent'] = network.nodes[i]['agent'].status

        nx.write_graphml_lxml(network, "measurement/"+str(pos)+"/network_time{}.graphml".format(idx))
