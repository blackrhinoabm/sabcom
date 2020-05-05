import random
import numpy as np
import math
import scipy.stats as stats


def edge_in_cliq(edge, nodes_in_cliq):
    if edge[0] in nodes_in_cliq:
        return True
    else:
        return False


def edges_to_remove_neighbourhood(all_edges, neighbourhood_density, nbh_nodes):
    neighbourhood_edges = [e for e in all_edges if edge_in_cliq(e, nbh_nodes)]
    sample_size = int(len(neighbourhood_edges) * (1-neighbourhood_density))
    # sample random edges
    chosen_edges = random.sample(neighbourhood_edges, sample_size)
    return chosen_edges


def what_neighbourhood(index, neighbourhood_nodes):
    for n in neighbourhood_nodes:
        if index in neighbourhood_nodes[n]:
            return n

    raise ValueError('Neighbourhood not found.')


def what_coordinates(neighbourhood_name, dataset):
    for x in range(len(dataset)):
        if neighbourhood_name in dataset[x]:
            return dataset[x][1]['lon'], dataset[x][1]['lat'],

    raise ValueError("Corresponding coordinates not found")


def what_informality(neighbourhood_name, dataset):
    for x in range(len(dataset)):
        if neighbourhood_name in dataset[x]:
            try:
                return dataset[x][1]['Informal_residential']
            except:
                return None

    raise ValueError("Corresponding informality not found")


def confidence_interval(data, av):
    sample_stdev = np.std(data)
    sigma = sample_stdev/math.sqrt(len(data))
    return stats.t.interval(alpha=0.95, df=24, loc=av, scale=sigma)
