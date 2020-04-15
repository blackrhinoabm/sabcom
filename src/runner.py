import numpy as np
import random
import geopy.distance


class Runner:
    def __init__(self):
        self.identifier = 1

    def do_run(self, environment, seed, verbose=False, high_performance=False):
        # set monte carlo seed
        np.random.seed(seed)
        random.seed(seed)

        # define sets for all agent types
        dead = []
        recovered = []
        critical = []
        sick_with_symptoms = []
        sick_without_symptoms = []
        susceptible = [agent for agent in environment.agents]

        # infect init agents
        init_agents_infected = int(
            environment.parameters["share_inital_agents_infected"] * environment.parameters["number_of_agents"])
        for p in range(init_agents_infected):
            agent_index = np.random.randint(0, len(environment.agents))
            environment.agents[agent_index].status = 'i1'
            sick_without_symptoms.append(environment.agents[agent_index])
        health_overburdened_multiplier = 1

        for t in range(environment.parameters["time"]):
            # create empty list of travel edges
            travel_edges = []

            for agent in susceptible + sick_without_symptoms + sick_with_symptoms + critical:
                # an agent might travel if it is not in critical state
                if np.random.random() < agent.prob_travel and agent.status != 'c':
                    # they sample all agents
                    agents_to_travel_to = random.sample(
                        environment.agents, int(environment.parameters["travel_sample_size"] * len(environment.agents)))
                    # and include travel time to each of these
                    agents_to_travel_to = {a2.name: geopy.distance.geodesic(
                        agent.coordinates, a2.coordinates).km for a2 in agents_to_travel_to if geopy.distance.geodesic(
                        agent.coordinates, a2.coordinates).km > 0.0}
                    # consider there are no viable options to travel to
                    if agents_to_travel_to:
                        # select the agent with shortest travel time
                        location_closest_agent = min(agents_to_travel_to, key=agents_to_travel_to.get)

                        # create edge to that agent
                        edge = (agent.name, location_closest_agent)  # own network location to other network location

                        # and store that edge
                        travel_edges.append(edge)

                # next assign the sickness status to the agents
                if agent.status == 'i1':
                    agent.incubation_days += 1
                    # some agents get symptoms
                    if agent.incubation_days > environment.parameters["incubation_days"]:
                        agent.status = 'i2'
                        sick_without_symptoms.remove(agent)
                        sick_with_symptoms.append(agent)

                elif agent.status == 'i2':
                    agent.sick_days += 1
                    # some agents recover
                    if agent.sick_days > environment.parameters["symptom_days"]:
                        if np.random.random() < agent.prob_hospital:
                            agent.status = 'c'
                            sick_with_symptoms.remove(agent)
                            critical.append(agent)
                        else:
                            agent.status = 'r'
                            sick_with_symptoms.remove(agent)
                            recovered.append(agent)

                elif agent.status == 'c':
                    agent.critical_days += 1
                    # some agents in critical status will die, the rest will recover
                    if agent.critical_days > environment.parameters["critical_days"]:
                        if np.random.random() < (agent.prob_death * health_overburdened_multiplier):
                            agent.status = 'd'
                            critical.remove(agent)
                            dead.append(agent)
                        else:
                            agent.status = 'r'
                            critical.remove(agent)
                            recovered.append(agent)

                elif agent.status == 'r':
                    agent.days_recovered += 1
                    if np.random.random() < (agent.prob_susceptible * agent.days_recovered):
                        recovered.remove(agent)
                        agent.status = 's'
                        susceptible.append(agent)

            # if the health system is overburdened the multiplier for the death rate is higher than otherwise
            if len(critical) / len(environment.agents) > environment.parameters["health_system_capacity"]:
                health_overburdened_multiplier = environment.parameters["no_hospital_multiplier"]
            else:
                health_overburdened_multiplier = 1.0

            # create travel edges
            environment.network.add_edges_from(travel_edges)

            for agent in sick_without_symptoms + sick_with_symptoms:
                # set the number of other agents infected this period 0
                agent.others_infected = 0
                # find indices from neighbour agents
                neighbours_from_graph = [x for x in environment.network.neighbors(agent.name)]
                # find the corresponding agents
                neighbours_to_infect = [environment.agents[idx] for idx in neighbours_from_graph]
                # let these agents be infected (with random probability
                for neighbour in neighbours_to_infect:
                    if neighbour.status == 's' and np.random.random() < agent.prob_transmission:
                        neighbour.status = 'i1'
                        susceptible.remove(neighbour)
                        sick_without_symptoms.append(neighbour)
                        agent.others_infected += 1

            if high_performance:
                print(t)
                # self.infection_states.append({'s1':len(sick_without_symptoms), 's2': len(sick_with_symptoms),
                #             'c': len(critical), 'd': len(dead), 'r': len(recovered)})
            else:
                environment.infection_states.append(environment.store_network())
                environment.write_status_location(t, seed)

            # delete travel edges
            environment.network.remove_edges_from(travel_edges)

            if verbose:
                print('time = ', t)
                print(environment.network.nodes)

    def calculate_R0(self, environment, seed, idx_patient_zero):
        # set monte carlo seed
        np.random.seed(seed)
        random.seed(seed)

        # define sets for all agent types
        dead = []
        recovered = []
        critical = []
        sick_with_symptoms = []
        sick_without_symptoms = []
        susceptible = [agent for agent in environment.agents]

        # infect patient zero
        environment.agents[idx_patient_zero].status = 'i1'
        sick_without_symptoms.append(environment.agents[idx_patient_zero])
        health_overburdened_multiplier = 1

        for t in range(environment.parameters["time"]):
            # create empty list of travel edges
            travel_edges = []

            for agent in susceptible + sick_without_symptoms + sick_with_symptoms + critical:
                # an agent might travel if it is not in critical state
                if np.random.random() < agent.prob_travel and agent.status != 'c':
                    # they sample all agents
                    agents_to_travel_to = random.sample(
                        environment.agents, int(environment.parameters["travel_sample_size"] * len(environment.agents)))
                    # and include travel time to each of these
                    agents_to_travel_to = {a2.name: geopy.distance.geodesic(
                        agent.coordinates, a2.coordinates).km for a2 in agents_to_travel_to if geopy.distance.geodesic(
                        agent.coordinates, a2.coordinates).km > 0.0}
                    # consider there are no viable options to travel to
                    if agents_to_travel_to:
                        # select the agent with shortest travel time
                        location_closest_agent = min(agents_to_travel_to, key=agents_to_travel_to.get)

                        # create edge to that agent
                        edge = (agent.name, location_closest_agent)  # own network location to other network location

                        # and store that edge
                        travel_edges.append(edge)

                # next assign the sickness status to the agents
                if agent.status == 'i1':
                    agent.incubation_days += 1
                    # some agents get symptoms
                    if agent.incubation_days > environment.parameters["incubation_days"]:
                        agent.status = 'i2'
                        sick_without_symptoms.remove(agent)
                        sick_with_symptoms.append(agent)

                elif agent.status == 'i2':
                    agent.sick_days += 1
                    # some agents recover
                    if agent.sick_days > environment.parameters["symptom_days"]:
                        if np.random.random() < agent.prob_hospital:
                            agent.status = 'c'
                            sick_with_symptoms.remove(agent)
                            critical.append(agent)
                        else:
                            if agent == environment.agents[idx_patient_zero]:
                                print('patient zero recovered or dead')
                                return agent.others_infected

                            agent.status = 'r'
                            sick_with_symptoms.remove(agent)
                            recovered.append(agent)

                elif agent.status == 'c':
                    agent.critical_days += 1
                    # some agents in critical status will die, the rest will recover
                    if agent.critical_days > environment.parameters["critical_days"]:
                        if agent == environment.agents[idx_patient_zero]:
                            print('patient zero recovered or dead')
                            return agent.others_infected
                        if np.random.random() < (agent.prob_death * health_overburdened_multiplier):
                            agent.status = 'd'
                            critical.remove(agent)
                            dead.append(agent)
                        else:
                            agent.status = 'r'
                            critical.remove(agent)
                            recovered.append(agent)

                elif agent.status == 'r':
                    agent.days_recovered += 1
                    if np.random.random() < (agent.prob_susceptible * agent.days_recovered):
                        recovered.remove(agent)
                        agent.status = 's'
                        susceptible.append(agent)

            # if the health system is overburdened the multiplier for the death rate is higher than otherwise
            if len(critical) / len(environment.agents) > environment.parameters["health_system_capacity"]:
                health_overburdened_multiplier = environment.parameters["no_hospital_multiplier"]
            else:
                health_overburdened_multiplier = 1.0

            # create travel edges
            environment.network.add_edges_from(travel_edges)

            for agent in sick_without_symptoms + sick_with_symptoms:
                # find indices from neighbour agents
                neighbours_from_graph = [x for x in environment.network.neighbors(agent.name)]
                # find the corresponding agents
                neighbours_to_infect = [environment.agents[idx] for idx in neighbours_from_graph]
                # let these agents be infected (with random probability
                for neighbour in neighbours_to_infect:
                    if neighbour.status == 's' and np.random.random() < agent.prob_transmission:
                        neighbour.status = 'i1'
                        susceptible.remove(neighbour)
                        sick_without_symptoms.append(neighbour)
                        agent.others_infected += 1

            # delete travel edges
            environment.network.remove_edges_from(travel_edges)
