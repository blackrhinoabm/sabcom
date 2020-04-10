from multiprocessing import Pool
from src.environment import EnvironmentNetwork
from src.runner import Runner
import networkx as nx
import json
import numpy as np

np.seterr(all='ignore')

# load parameters
with open('parameters.json') as json_file:
    parameters = json.load(json_file)

# load neighbourhood data
with open('neighbourhood_data.json') as json_file:
    neighbourhood_data = json.load(json_file)

CORES = parameters["monte_carlo_runs"]

def pool_handler():
    p = Pool(CORES) # argument is how many process happening in parallel
    list_of_seeds = [x for x in range(parameters["monte_carlo_runs"])]

    output = constrNM(model_performance, init_parameters, LB, UB, maxiter=25, full_output=True)

    with open('estimated_params.json', 'w') as f:
        json.dump(list(output['xopt']), f)

    print('All outputs are: ', output)


if __name__ == '__main__':
    pool_handler()
    print("The simulations took", time.time() - start_time, "to run")
