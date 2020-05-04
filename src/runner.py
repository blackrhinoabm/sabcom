import numpy as np
import random
import geopy.distance


def runner(environment, seed, data_folder='measurement/',
           verbose=False, data_output=False, travel_matrix=None):
    """
    This function is used to run / simulate the model.

    :param environment: contains the parameters and agents, Environment object
    :param seed: used to initialise the random generators to ensure reproducibility, int
    :param data_folder: specifying the folder that will be used to write the data to, string
    :param verbose: specify whether or not the model will print out overview, Boolean
    :param data_output:  can be 'csv', 'network', or False (for no output)
    :param travel_matrix: contains rows per district and the probability of travelling to another int he columns,
    pandas Dataframe
    :return: environment object containing the updated agents, Environment object
    """
    # set monte carlo seed
    np.random.seed(seed)
    random.seed(seed)

    # define sets for all agent types
    dead = []
    recovered = []
    critical = []
    sick_with_symptoms = []
    sick_without_symptoms = []
    exposed = []
    susceptible = [agent for agent in environment.agents]
    health_overburdened_multiplier = 1

    for t in range(environment.parameters["time"]):
        # for the first days of the simulation there will be one new agent infected
        if t in environment.parameters['foreign_infection_days']:
            # select district with probability
            chosen_district = np.random.choice(environment.districts, 1,
                                               environment.probabilities_new_infection_district)[0]
            # select random agent in that ward
            if t == 0:
                for x in range(1):
                    chosen_agent = random.choice(environment.district_agents[chosen_district])
                    chosen_agent.status = 'i1'
                    sick_without_symptoms.append(chosen_agent)
            else:
                chosen_agent = random.choice(environment.district_agents[chosen_district])
                chosen_agent.status = 'i1'
                sick_without_symptoms.append(chosen_agent)

        # the lockdown influence the travel multiplier and infection multiplier (probability to infect) TODO add influence superspreaders?
        if t in environment.parameters["lockdown_days"]:
            # During lockdown days the probability that others are infected and that there is travel will be reduced
            lockdown_infection_multiplier = environment.parameters["lockdown_infection_multiplier"]
            lockdown_travel_multiplier = environment.parameters["lockdown_travel_multiplier"]
            lockdown_travel_max = environment.parameters['lockdown_travel_max_contacts']
        else:
            lockdown_infection_multiplier = 1.0
            lockdown_travel_multiplier = 1.0
            lockdown_travel_max = float('inf')

        # create empty list of travel edges
        travel_edges = []

        for agent in exposed + susceptible + sick_without_symptoms + sick_with_symptoms + critical:
            # determine if the agent is at risk
            if agent.age_group not in environment.parameters["at_risk_groups"]:
                not_at_risk_dummy = 1.0
            else:
                not_at_risk_dummy = 0.0

            informality_term = (1 - lockdown_travel_multiplier) * agent.informality
            at_risk_term = 1 - lockdown_travel_multiplier - informality_term
            # an agent might travel (multiple times) if it is not in critical state agent.num_travel
            if np.random.random() < (agent.prob_travel * (
                    lockdown_travel_multiplier + informality_term + (at_risk_term * not_at_risk_dummy))) and \
                    agent.status != 'c':
                for trip in range(min(lockdown_travel_max, agent.num_trips)):  # TODO debug... can this be targeted directly? Perhaps in if lockdown... reduce num_trips
                    # they sample all agents
                    agents_to_travel_to = random.sample(
                        environment.agents, int(environment.parameters["travel_sample_size"] * len(environment.agents)))

                    if travel_matrix is None:
                        # and include travel time to each of these
                        agents_to_travel_to = {a2.name: environment.distance_matrix[str(agent.district)].loc[a2.district] for a2 in
                                               agents_to_travel_to if
                                               environment.distance_matrix[str(agent.district)].loc[a2.district] > 0.0}
                    else:
                        probabilities = list(travel_matrix.loc[agent.district])

                        district_to_travel_to = np.random.choice(environment.districts, size=1, p=probabilities)[0]
                        agents_to_travel_to = environment.district_agents[district_to_travel_to]

                    # consider there are no viable options to travel to ... travel to multiple agents
                    if agents_to_travel_to:
                        if travel_matrix is None:
                            # select the agent with shortest travel time
                            location_closest_agent = min(agents_to_travel_to, key=agents_to_travel_to.get)
                        else:
                            location_closest_agent = random.choice(agents_to_travel_to).name

                        # create edge to that agent
                        edge = (agent.name, location_closest_agent)  # own network location to other network location

                        # and store that edge
                        travel_edges.append(edge)

            if agent.status == 'e':
                agent.exposed_days += 1
                # some agents will become infectious but do not show agents while others will show symptoms
                if agent.exposed_days > environment.parameters["exposed_days"]:
                    if np.random.random() < agent.prob_symptomatic:
                        agent.status = 'i2'
                        exposed.remove(agent)
                        sick_with_symptoms.append(agent)
                    else:
                        agent.status = 'i1'
                        exposed.remove(agent)
                        sick_without_symptoms.append(agent)

            if agent.status == 'i1':
                agent.asymptom_days += 1
                # these agents all recover after some time
                if agent.asymptom_days > environment.parameters["asymptom_days"]:
                    agent.status = 'r'
                    sick_without_symptoms.remove(agent)
                    recovered.append(agent)

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
                if neighbour.age_group not in environment.parameters["at_risk_groups"]:
                    not_at_risk_dummy = 1.0
                else:
                    not_at_risk_dummy = 0.0

                informality_term = (1 - lockdown_infection_multiplier) * agent.informality
                at_risk_term = 1 - lockdown_infection_multiplier - informality_term

                if neighbour.status == 's' and np.random.random() < (
                        agent.prob_transmission * (
                        lockdown_infection_multiplier + informality_term + (at_risk_term * not_at_risk_dummy))):
                    neighbour.status = 'e'
                    susceptible.remove(neighbour)
                    exposed.append(neighbour)
                    agent.others_infected += 1

        if data_output == 'network':
            environment.infection_states.append(environment.store_network())
        elif data_output == 'csv':
            environment.write_status_location(t, seed, data_folder)
        elif data_output == 'csv_light':
            # save only the total quantity of agents per category
            for key, quantity in zip(['e', 's', 'i1', 'i2', 'c', 'r', 'd'], [exposed, susceptible,
                                                                        sick_without_symptoms, sick_with_symptoms,
                                                                        critical, recovered, dead]):
                environment.infection_quantities[key].append(len(quantity))

            # delete travel edges
        environment.network.remove_edges_from(travel_edges)

        if verbose:
            print('time = ', t)
            print(environment.network.nodes)

    return environment


def runner_mean_field(environment, seed, data_folder='measurement/',
           verbose=False, data_output=False, travel_matrix=None):
    """
    This function is used to run / simulate the model mean field version where agents are in a random network,
    all travel equally much, and do not travel to the closest agent / most visited district, but at random.

    :param environment: contains the parameters and agents, Environment object
    :param seed: used to initialise the random generators to ensure reproducibility, int
    :param data_folder: specifying the folder that will be used to write the data to, string
    :param verbose: specify whether or not the model will print out overview, Boolean
    :param data_output:  can be 'csv', 'network', or False (for no output)
    :param travel_matrix: contains rows per district and the probability of travelling to another int he columns,
    pandas Dataframe
    :return: environment object containing the updated agents, Environment object
    """
    # set monte carlo seed
    np.random.seed(seed)
    random.seed(seed)

    # define sets for all agent types
    dead = []
    recovered = []
    critical = []
    sick_with_symptoms = []
    sick_without_symptoms = []
    exposed = []
    susceptible = [agent for agent in environment.agents]
    health_overburdened_multiplier = 1

    for t in range(environment.parameters["time"]):
        # for the first days of the simulation there will be one new agent infected
        if t in environment.parameters['foreign_infection_days']:
            # select district with probability
            chosen_district = np.random.choice(environment.districts, 1,
                                               environment.probabilities_new_infection_district)[0]
            # select random agent in that ward
            if t == 0:
                for x in range(1):
                    chosen_agent = random.choice(environment.district_agents[chosen_district])
                    chosen_agent.status = 'i1'
                    sick_without_symptoms.append(chosen_agent)
            else:
                chosen_agent = random.choice(environment.district_agents[chosen_district])
                chosen_agent.status = 'i1'
                sick_without_symptoms.append(chosen_agent)

        # the lockdown influence the travel multiplier and infection multiplier (probability to infect) TODO add influence superspreaders?
        if t in environment.parameters["lockdown_days"]:
            # During lockdown days the probability that others are infected and that there is travel will be reduced
            lockdown_infection_multiplier = environment.parameters["lockdown_infection_multiplier"]
            lockdown_travel_multiplier = environment.parameters["lockdown_travel_multiplier"]
            lockdown_travel_max = environment.parameters['lockdown_travel_max_contacts']
        else:
            lockdown_infection_multiplier = 1.0
            lockdown_travel_multiplier = 1.0
            lockdown_travel_max = float('inf')

        # create empty list of travel edges
        travel_edges = []

        for agent in exposed + susceptible + sick_without_symptoms + sick_with_symptoms + critical:
            # determine if the agent is at risk
            if agent.age_group not in environment.parameters["at_risk_groups"]:
                not_at_risk_dummy = 1.0
            else:
                not_at_risk_dummy = 0.0

            informality_term = (1 - lockdown_travel_multiplier) * agent.informality
            at_risk_term = 1 - lockdown_travel_multiplier - informality_term
            # an agent might travel (multiple times) if it is not in critical state agent.num_travel
            if np.random.random() < (agent.prob_travel * (
                    lockdown_travel_multiplier + informality_term + (at_risk_term * not_at_risk_dummy))) and \
                    agent.status != 'c':
                for trip in range(min(lockdown_travel_max, agent.num_trips)):  # TODO can this be targeted directly? Perhaps in if lockdown... reduce num_trips
                    # they sample all agents
                    agents_to_travel_to = random.sample(
                        environment.agents, int(environment.parameters["travel_sample_size"] * len(environment.agents)))

                    # consider there are no viable options to travel to ... travel to multiple agents
                    if agents_to_travel_to:
                        # select a random  agent to travel to TODO this is the unique feature of the no geography model
                        location_random_agent = random.choice(agents_to_travel_to).name

                        # create edge to that agent
                        edge = (agent.name, location_random_agent)

                        # and store that edge
                        travel_edges.append(edge)

            if agent.status == 'e':
                agent.exposed_days += 1
                # some agents will become infectious but do not show agents while others will show symptoms
                if agent.exposed_days > environment.parameters["exposed_days"]:
                    if np.random.random() < agent.prob_symptomatic:
                        agent.status = 'i2'
                        exposed.remove(agent)
                        sick_with_symptoms.append(agent)
                    else:
                        agent.status = 'i1'
                        exposed.remove(agent)
                        sick_without_symptoms.append(agent)

            if agent.status == 'i1':
                agent.asymptom_days += 1
                # these agents all recover after some time
                if agent.asymptom_days > environment.parameters["asymptom_days"]:
                    agent.status = 'r'
                    sick_without_symptoms.remove(agent)
                    recovered.append(agent)

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
                if neighbour.age_group not in environment.parameters["at_risk_groups"]:
                    not_at_risk_dummy = 1.0
                else:
                    not_at_risk_dummy = 0.0

                informality_term = (1 - lockdown_infection_multiplier) * agent.informality
                at_risk_term = 1 - lockdown_infection_multiplier - informality_term

                if neighbour.status == 's' and np.random.random() < (
                        agent.prob_transmission * (
                        lockdown_infection_multiplier + informality_term + (at_risk_term * not_at_risk_dummy))):
                    neighbour.status = 'e'
                    susceptible.remove(neighbour)
                    exposed.append(neighbour)
                    agent.others_infected += 1

        if data_output == 'network':
            environment.infection_states.append(environment.store_network())
        elif data_output == 'csv':
            environment.write_status_location(t, seed, data_folder)
        elif data_output == 'csv_light':
            # save only the total quantity of agents per category
            for key, quantity in zip(['e', 's', 'i1', 'i2', 'c', 'r', 'd'], [exposed, susceptible,
                                                                        sick_without_symptoms, sick_with_symptoms,
                                                                        critical, recovered, dead]):
                environment.infection_quantities[key].append(len(quantity))

            # delete travel edges
        environment.network.remove_edges_from(travel_edges)

        if verbose:
            print('time = ', t)
            print(environment.network.nodes)

    return environment


def runner_calculate_r_naught(environment, seed, idx_patient_zero): #TODO update
    """Depreciated!"""
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