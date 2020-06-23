import numpy as np
import random


def runner(environment, seed, data_folder='measurement/',
           data_output=False, calculate_r_naught=False):
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

    # PHASE 5.2 of the initialisation happens in the runner to allow for cluster computing of calc R_naught
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
                                                 len(environment.parameters['total_initial_infections']),
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
                chosen_agent.exposed_days = np.random.randint(0, environment.parameters['exposed_days'])
                exposed.append(chosen_agent)
                susceptible.remove(chosen_agent)

    for t in range(environment.parameters["time"]):
        # PHASE 1 HEALTH SYSTEM check if it is not overburdened
        if len(critical) / len(environment.agents) > environment.parameters["health_system_capacity"]:
            health_overburdened_multiplier = environment.parameters["no_hospital_multiplier"]
        else:
            health_overburdened_multiplier = 1.0

        # PHASE 2 STATUS UPDATE update infection status of all agents
        for agent in exposed + sick_without_symptoms + sick_with_symptoms + critical:  # + recovered if SEIRS model
            if agent.status == 'e':
                agent.exposed_days += 1
                # some agents will become infectious but do not show agents while others will show symptoms
                if agent.exposed_days > environment.parameters["exposed_days"]:
                    if np.random.random() < environment.parameters["probability_symptomatic"]:  # agent.prob_symptomatic:
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
                    if np.random.random() < environment.parameters["probability_critical"][agent.age_group]:   #agent.prob_hospital:
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

                    if np.random.random() < (environment.parameters["probability_to_die"][
                                agent.age_group] * health_overburdened_multiplier):
                        agent.status = 'd'
                        critical.remove(agent)
                        dead.append(agent)
                    else:
                        agent.status = 'r'
                        critical.remove(agent)
                        recovered.append(agent)

            elif agent.status == 'r':
                agent.days_recovered += 1
                if np.random.random() < (environment.parameters["probability_susceptible"] * agent.days_recovered):
                    recovered.remove(agent)
                    agent.status = 's'
                    susceptible.append(agent)

        # PHASE 3 LOCKDOWN
        if t in environment.parameters["lockdown_days"]:
            # During lockdown days the probability that others are infected and that there is travel will be reduced
            physical_distancing_multiplier = environment.parameters["physical_distancing_multiplier"]
            gathering_max_contacts = environment.parameters['gathering_max_contacts']
            likelihood_awareness = environment.parameters['likelihood_awareness']
            visiting_r_contacts_multiplier = environment.parameters["visiting_recurring_contacts_multiplier"]
        else:
            likelihood_awareness = 0.0
            physical_distancing_multiplier = 1.0
            gathering_max_contacts = float('inf')
            visiting_r_contacts_multiplier = 1.0

        # PHASE 4 INFECTIONS
        for agent in sick_without_symptoms + sick_with_symptoms:
            agent.others_infected = 0

            # find indices from neighbour agents
            #neighbours_from_graph = [x for x in environment.network.neighbors(agent.name)]
            household_neighbours = [x for x in environment.network.neighbors(agent.name) if
                                    environment.agents[x].household_number == agent.household_number and
                                    environment.agents[x].district == agent.district]
            other_neighbours = [x for x in environment.network.neighbors(agent.name) if
                                environment.agents[x].household_number != agent.household_number or
                                environment.agents[x].district != agent.district]

            # depending on lockdown policies, the amount of non-household contacts an agent can visit is reduced
            informality_term_contacts = (1 - visiting_r_contacts_multiplier) * agent.informality

            planned_contacts = int(round(len(other_neighbours
                                             ) * (visiting_r_contacts_multiplier + informality_term_contacts)))

            # by gathering max contacts
            if t in environment.parameters["lockdown_days"]:
                individual_max_contacts = int(round(gathering_max_contacts * (1 + agent.informality)))
            else:
                individual_max_contacts = gathering_max_contacts

            if planned_contacts > individual_max_contacts:
                other_neighbours = random.sample(other_neighbours, individual_max_contacts)
            else:
                other_neighbours = random.sample(other_neighbours, planned_contacts)

            # Next, combine household neighbours with other neighbours
            neighbours_from_graph = household_neighbours + other_neighbours

            # find the corresponding agents and add them to a list to infect
            # if the agent is aware it will limit its contact to only household contacts
            if agent.status in environment.parameters['aware_status'] and \
                    np.random.random() < likelihood_awareness * (1 - agent.informality):
                neighbours_to_infect = [environment.agents[idx] for idx in household_neighbours]
                # for idx in neighbours_from_graph:
                #     if environment.agents[idx].household_number == agent.household_number and \
                #             environment.agents[idx].district == agent.district:
                #         neighbours_to_infect.append(environment.agents[idx])
            # otherwise the agent will interact with all neighbours from graph
            else:
                neighbours_to_infect = [environment.agents[idx] for idx in neighbours_from_graph]

            # let these agents be infected (with random probability
            for neighbour in neighbours_to_infect:
                # informality term phys dist is 1 if it comes to infecting household members: TODO add to model description
                if neighbour.household_number == agent.household_number and neighbour.district == agent.district:
                    informality_term_phys_dis = (1 - physical_distancing_multiplier)
                else:
                    informality_term_phys_dis = (1 - physical_distancing_multiplier) * agent.informality

                if neighbour.status == 's' and np.random.random() < (
                        environment.parameters['probability_transmission'] * (
                        physical_distancing_multiplier + informality_term_phys_dis)):
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

    print(len(dead))
    return environment
