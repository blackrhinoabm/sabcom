import numpy as np
import random
import geopy.distance


class Runner:
    def __init__(self):
        self.identifier = 1

    def do_run(self, environment, seed, time, days_incubation=7, days_with_symptoms=8, days_critical=10,
                 relative_hospital_capacity=0.2, hospital_overburdened_multiplier=1.5, travel_sample_size=0.1,
                 verbose=False, high_performance=False):
        # set monte carlo seed
        np.random.seed(seed)
        random.seed(seed)

        # infect five random agents TODO make param as percentage of population
        for p in range(5):
            environment.agents[np.random.randint(0, len(environment.agents))].status = 'i1'
        health_overburdened_multiplier = 1

        # some agents are dead and will never come back
        dead = []

        for t in range(time):
            # some agents are the infected
            sick_with_symptoms = []
            sick_without_symptoms = []

            # some agents are in the hospital
            critical = []
            # some agents are recovered
            recovered = []

            # create empty list of travel edges
            travel_edges = []

            for idx, agent in enumerate(environment.agents):
                # an agent might travel
                if np.random.random() < agent.prob_travel:
                    # they sample all agents
                    agents_to_travel_to = random.sample(environment.agents, int(travel_sample_size * len(environment.agents)))
                    # and include travel time to each of these
                    agents_to_travel_to = {a2.name :geopy.distance.geodesic(agent.coordinates, a2.coordinates
                                                                           ).km for a2 in agents_to_travel_to if geopy.distance.geodesic(agent.coordinates, a2.coordinates
                                                                                                                                         ).km > 0.0}
                    # consider there are no viable options to travel to
                    if agents_to_travel_to:
                        # select the agent with shortest travel time
                        location_closest_agent = min(agents_to_travel_to, key=agents_to_travel_to.get)

                        # create edge to that agent
                        edge = (idx, location_closest_agent) # own network location to other agent network location

                        # and store that edge
                        travel_edges.append(edge)

                # next assign the sickness status to the agents
                if agent.status == 'i1':
                    sick_without_symptoms.append(agent)
                    agent.incubation_days += 1

                    # some agents get symptoms
                    if agent.incubation_days > days_incubation:
                        agent.status = 'i2'
                        sick_without_symptoms.remove(agent)

                if agent.status == 'i2':
                    sick_with_symptoms.append(agent)
                    agent.sick_days += 1
                    # some agents recover
                    if agent.sick_days > days_with_symptoms:
                        if np.random.random() < agent.prob_hospital:
                            agent.status = 'c'
                            sick_with_symptoms.remove(agent)
                        else:
                            agent.status = 'r'
                            sick_with_symptoms.remove(agent)

                if agent.status == 'c':
                    critical.append(agent)
                    agent.critical_days += 1
                    # some agents in critical status will die, the rest will recover
                    if agent.critical_days > days_critical:
                        if np.random.random() < (agent.prob_death * health_overburdened_multiplier):
                            agent.status = 'd'
                            critical.remove(agent)
                            dead.append(agent)
                        else:
                            agent.status = 'r'
                            critical.remove(agent)

                if agent.status == 'r':
                    recovered.append(agent)
                    agent.days_recovered += 1
                    if np.random.random() < (agent.prob_susceptible * agent.days_recovered):
                        recovered.remove(agent)
                        agent.status = 's'

            # if the health system is overburdened the multiplier for the death rate is higher than otherwise
            if len(critical) / len(environment.agents) > relative_hospital_capacity:
                health_overburdened_multiplier = hospital_overburdened_multiplier
            else:
                health_overburdened_multiplier = 1.0

            # create travel edges
            environment.network.add_edges_from(travel_edges)

            for agent in sick_without_symptoms + sick_with_symptoms:
                # find indices from neighbour agents
                neighbours_from_graph = [x for x in environment.network.neighbors(agent.name)]
                # find the corresponding agents
                neighbours_to_infect = [environment.agents[idx] for idx in neighbours_from_graph]
                agent.infect(neighbours_to_infect)

            if high_performance:
                print(t)
                # self.infection_states.append({'s1':len(sick_without_symptoms), 's2': len(sick_with_symptoms),
                #             'c': len(critical), 'd': len(dead), 'r': len(recovered)})
            else:
                environment.infection_states.append(environment.store_network())
                environment.write_status_location(t)

            # delete travel edges
            environment.network.remove_edges_from(travel_edges)

            if verbose:
                print('time = ', t)
                print(environment.network.nodes)