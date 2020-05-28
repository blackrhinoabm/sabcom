import numpy as np
import random


def runner(environment, seed, data_folder='measurement/',
           data_output=False, travel_matrix=None, calculate_r_naught=False):
    """
    This function is used to run / simulate the model.

    :param environment: contains the parameters and agents, Environment object
    :param seed: used to initialise the random generators to ensure reproducibility, int
    :param data_folder:  string of the folder where data output files should be created
    :param data_output:  can be 'csv', 'network', or False (for no output)
    :param travel_matrix: contains rows per district and the probability of travelling to another int he columns,
    pandas Dataframe
    :param calculate_r_naught: set to True to calculate the R0 that the model produces given a single infected agent
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

    # here a fixed initial agent can be infected once to calculate R0
    if calculate_r_naught:
        initial_infected = []
        chosen_agent = environment.agents[environment.parameters['init_infected_agent']]
        chosen_agent.status = 'e'
        # this list can be used to calculate R0
        initial_infected.append(chosen_agent)
        exposed.append(chosen_agent)
        susceptible.remove(chosen_agent)
    # else infect a set of agents and
    else:
        initial_infected = []
        # select districts with probability
        chosen_districts = list(np.random.choice(environment.districts,
                                                 len(environment.parameters['foreign_infection_days']), #  TODO change name of parameter
                                                 environment.probabilities_new_infection_district))
        # count how often a district is in that list

        chosen_districts = {distr: min(len(environment.district_agents[distr]),
                                       chosen_districts.count(distr)) for distr in chosen_districts}

        for district in chosen_districts:
            # infect x random agents
            chosen_agents = np.random.choice(environment.district_agents[district], chosen_districts[district],
                                             replace=False)
            for chosen_agent in chosen_agents:
                chosen_agent.status = 'e'
                exposed.append(chosen_agent)
                susceptible.remove(chosen_agent)

    for t in range(environment.parameters["time"]):
        if t in environment.parameters["lockdown_days"]:
            # During lockdown days the probability that others are infected and that there is travel will be reduced
            physical_distancing_multiplier = environment.parameters["physical_distancing_multiplier"]
            gathering_max_contacts = environment.parameters['gathering_max_contacts']
            general_isolation_multiplier = environment.parameters['self_isolation_multiplier']

            # Furthermore, a set of close contact edges may be removed
            original_edges = environment.network.edges
            k = int(round(len(original_edges) * (environment.parameters["visiting_recurring_contacts_multiplier"])))
            to_be_removed_edges = random.sample(original_edges, k)
            # remove some of the original edges
            environment.network.remove_edges_from(to_be_removed_edges)
        else:
            general_isolation_multiplier = 1.0
            physical_distancing_multiplier = 1.0
            gathering_max_contacts = float('inf')
            to_be_removed_edges = []

        # update infection status of all agents
        for agent in exposed + susceptible + sick_without_symptoms + sick_with_symptoms + critical:
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
                agent.asymptomatic_days += 1
                # these agents all recover after some time
                if agent.asymptomatic_days > environment.parameters["asymptom_days"]:
                    # calculate R0 here
                    if calculate_r_naught and agent in initial_infected:
                        print(t, ' patient zero recovered or dead with R0 = ', agent.others_infects_total)
                        return agent.others_infects_total

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
                        # calculate R0 here
                        if calculate_r_naught and agent in initial_infected:
                            print(t, ' patient zero recovered or dead with R0 = ', agent.others_infects_total)
                            return agent.others_infects_total
                        agent.status = 'r'
                        sick_with_symptoms.remove(agent)
                        recovered.append(agent)

            elif agent.status == 'c':
                agent.critical_days += 1
                # some agents in critical status will die, the rest will recover
                if agent.critical_days > environment.parameters["critical_days"]:
                    # calculate R0 here
                    if calculate_r_naught and agent in initial_infected:
                        print(t, ' patient zero recovered or dead with R0 = ', agent.others_infects_total)
                        return agent.others_infects_total

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

        # sick agents will infect other agents
        for agent in sick_without_symptoms + sick_with_symptoms:
            if agent.status in environment.parameters['aware_status'] and \
                    np.random.random() < environment.parameters['likelihood_awareness']:
                self_isolation_multiplier = general_isolation_multiplier
            else:
                self_isolation_multiplier = 1.0
            # set the number of other agents infected this period 0
            agent.others_infected = 0
            # find indices from neighbour agents
            neighbours_from_graph = [x for x in environment.network.neighbors(agent.name)]

            # depending on lockdown policies, the amount of contacts an agent can visit is limited by the self
            # isolation multiplier and gathering max contacts
            k = int(round(len(neighbours_from_graph) * self_isolation_multiplier)) #TODO debug
            if k > gathering_max_contacts:
                neighbours_from_graph = random.sample(neighbours_from_graph, gathering_max_contacts)
            else:
                neighbours_from_graph = random.sample(neighbours_from_graph, k)

            # find the corresponding agents
            neighbours_to_infect = [environment.agents[idx] for idx in neighbours_from_graph]
            # let these agents be infected (with random probability
            for neighbour in neighbours_to_infect:
                informality_term = (1 - physical_distancing_multiplier) * agent.informality

                if neighbour.status == 's' and np.random.random() < (
                        environment.parameters['probability_transmission'] * (
                        physical_distancing_multiplier + informality_term)):
                    neighbour.status = 'e'
                    susceptible.remove(neighbour)
                    exposed.append(neighbour)
                    agent.others_infected += 1
                    agent.others_infects_total += 1

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

        # add social network edges that were removed in lockdown
        environment.network.add_edges_from(to_be_removed_edges)

    return environment
