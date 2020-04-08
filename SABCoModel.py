from src.environment import EnvironmentNetwork
from src.runner import Runner
import networkx as nx

TIME = 100
NUM_AGENTS = 1000
SEED = 1
# agent parameters
TRANSMISSION_RATE = 0.04
PROBABILITY_HOSPITAL = 0.2
PROBABILITY_TO_DIE = 0.03
PROBABILITY_SUSCEPTIBLE = 0.001
PROBABILITY_TRAVEL = 0.25

# simulation parameters
INCUBATION_DAYS = 7
SYMPTOM_DAYS = 8
CRITICAL_DAYS = 10
HEALTH_SYSTEM_CAPACITY = 0.2  # relative
NO_HOSPITAL_MULTIPLIER = 1.5
TRAVEL_SAMPLE_SIZE = 0.1

NEIGHBOURHOOD_DATA = [[199016089.0, {'population': 1214.0, 'population_KM': 2.0460201586268703,
                                     'lon': 18.6453829505698, 'lat': -33.9139569040631}],
                      [199041008.0, {'population': 11.0, 'population_KM': 0.00232805753448935,
                                     'lon': 18.392843611306898, 'lat': -33.9304751653803}],
                      [199017021.0, {'population': 7889.0, 'population_KM': 1.4014101214518002,
                                     'lon': 18.6501824979933, 'lat': -33.832976845384195}]]

# parameter that makes the model less memory intensive
HIGH_PERFORMANCE = False

# initialization
environment = EnvironmentNetwork(seed=SEED, number_agents=NUM_AGENTS, prob_transmit=TRANSMISSION_RATE,
                                 prob_hospital=PROBABILITY_HOSPITAL, prob_death=PROBABILITY_TO_DIE,
                                 prob_susceptible=PROBABILITY_SUSCEPTIBLE, prob_travel=PROBABILITY_TRAVEL,
                                 neighbourhood_data=NEIGHBOURHOOD_DATA)

# running the simulation
runner = Runner()
runner.do_run(SEED, TIME, days_incubation=INCUBATION_DAYS, days_with_symptoms=SYMPTOM_DAYS,
                     days_critical=CRITICAL_DAYS, relative_hospital_capacity=HEALTH_SYSTEM_CAPACITY,
                     hospital_overburdened_multiplier=NO_HOSPITAL_MULTIPLIER, travel_sample_size=TRAVEL_SAMPLE_SIZE,
                     high_performance=HIGH_PERFORMANCE, verbose=False)


# save network
if not HIGH_PERFORMANCE:
    for idx, network in enumerate(environment.infection_states):
        for i, node in enumerate(network.nodes):
            network.nodes[i]['agent'] = network.nodes[i]['agent'].status

        nx.write_graphml_lxml(network, "output/network_time{}.graphml".format(idx))