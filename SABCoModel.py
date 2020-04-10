from src.environment import EnvironmentNetwork
from src.runner import Runner
import networkx as nx
import json


# load parameters
with open('parameters.json') as json_file:
    parameters = json.load(json_file)

# load neighbourhood data
with open('neighbourhood_data.json') as json_file:
    neighbourhood_data = json.load(json_file)

# Monte Carlo simulation
for seed in range(parameters['monte_carlo_runs']):
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

            idx_string = '{0:04}'.format(idx)
            nx.write_graphml_lxml(network, "measurement/{}_network_time{}.graphml".format(seed, idx_string))
