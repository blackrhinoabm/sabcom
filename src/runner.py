import numpy as np
import random


def runner(environment, initial_infections, seed, data_folder='measurement/',
           data_output=False, calculate_r_naught=False):
    """
    This function is used to run / simulate the model.

    :param environment: contains the parameters and agents, Environment object
    :param initial_infections: contains the Wards and corresponding initial infections, Pandas DataFrame
    :param seed: used to initialise the random generators to ensure reproducibility, int
    :param data_folder:  string of the folder where data output files should be created
    :param data_output:  can be 'csv', 'network', or False (for no output)
    :param calculate_r_naught: set to True to calculate the R0 that the model produces given a single infected agent
    :return: environment object containing the updated agents, Environment object
    """
    # set monte carlo seed
    np.random.seed(seed)
    random.seed(seed)

    # create sets for all agent types
    dead = []
    recovered = []
    critical = []
    sick_with_symptoms = []
    sick_without_symptoms = []
    exposed = []
    susceptible = [agent for agent in environment.agents]

    # 4 Initialisation of infections
    # here either a fixed initial agent can be infected once to calculate R0
    if calculate_r_naught:
        initial_infected = []
        chosen_agent = environment.agents[environment.parameters['init_infected_agent']]
        chosen_agent.status = 'e'
        initial_infected.append(chosen_agent)
        exposed.append(chosen_agent)
        susceptible.remove(chosen_agent)
    # otherwise infect a set of agents based on the locations of observed infections
    else:
        initial_infections = initial_infections.sort_index()
        cases = [x for x in initial_infections['Cases_03292020']]
        probabilities_new_infection_district = [float(i) / sum(cases) for i in cases]

        initial_infected = []
        environment.newly_detected_cases[0] += round(len(environment.parameters['total_initial_infections']) *
                                                     environment.parameters["perc_infections_detects"])
        # select districts with probability
        chosen_districts = list(np.random.choice(environment.districts,
                                                 len(environment.parameters['total_initial_infections']),
                                                 p=probabilities_new_infection_district))
        # count how often a district is in that list
        chosen_districts = {distr: min(len(environment.district_agents[distr]),
                                       chosen_districts.count(distr)) for distr in chosen_districts}

        for district in chosen_districts:
            # infect appropriate number of random agents
            chosen_agents = np.random.choice(environment.district_agents[district], chosen_districts[district],
                                             replace=False)
            for chosen_agent in chosen_agents:
                chosen_agent.status = 'e'
                # give exposed days a random value to avoid an unrealistic wave of initial infections
                chosen_agent.exposed_days = np.random.randint(0, environment.parameters['exposed_days'])
                exposed.append(chosen_agent)
                susceptible.remove(chosen_agent)

    for t in range(environment.parameters["time"]):
        # Check if the health system is not overburdened
        if len(critical) / len(environment.agents) > environment.parameters["health_system_capacity"]:
            health_overburdened_multiplier = environment.parameters["no_hospital_multiplier"]
        else:
            health_overburdened_multiplier = 1.0

        # 5 update infection status of all agents
        for agent in exposed + sick_without_symptoms + sick_with_symptoms + critical:  # + recovered if SEIRS model
            if agent.status == 'e':
                agent.exposed_days += 1
                # some agents will become infectious but do not show agents while others will show symptoms
                if agent.exposed_days > environment.parameters["exposed_days"]:
                    if np.random.random() < environment.parameters["probability_symptomatic"]:
                        agent.status = 'i2'
                        exposed.remove(agent)
                        sick_with_symptoms.append(agent)
                    else:
                        agent.status = 'i1'
                        exposed.remove(agent)
                        sick_without_symptoms.append(agent)

            if agent.status == 'i1':
                agent.asymptomatic_days += 1
                # asymptomatic agents all recover after some time
                if agent.asymptomatic_days > environment.parameters["asymptom_days"]:
                    # calculate R0 here if the first agent recovers
                    if calculate_r_naught and agent in initial_infected:
                        print(t, ' patient zero recovered or dead with R0 = ', agent.others_infects_total)
                        return agent.others_infects_total

                    agent.status = 'r'
                    sick_without_symptoms.remove(agent)
                    recovered.append(agent)

            elif agent.status == 'i2':
                agent.sick_days += 1
                # some symptomatic agents recover
                if agent.sick_days > environment.parameters["symptom_days"]:
                    if np.random.random() < environment.parameters["probability_critical"][agent.age_group]:
                        agent.status = 'c'
                        sick_with_symptoms.remove(agent)
                        critical.append(agent)
                    else:
                        # calculate R0 here if the first agent recovers
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
                    # calculate R0 here if the first agent recovers or dies
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

        # 6 New infections
        for agent in sick_without_symptoms + sick_with_symptoms:
            agent.others_infected = 0

            # find indices from neighbour agents
            household_neighbours = [x for x in environment.network.neighbors(agent.name) if
                                    environment.agents[x].household_number == agent.household_number and
                                    environment.agents[x].district == agent.district]
            other_neighbours = [x for x in environment.network.neighbors(agent.name) if
                                environment.agents[x].household_number != agent.household_number or
                                environment.agents[x].district != agent.district]

            # depending on lockdown policies, the amount of non-household contacts an agent can visit is reduced
            visiting_r_contacts_multiplier = environment.parameters["visiting_recurring_contacts_multiplier"][t]
            informality_term_contacts = (1 - visiting_r_contacts_multiplier) * agent.informality

            planned_contacts = int(round(len(other_neighbours
                                             ) * (visiting_r_contacts_multiplier + informality_term_contacts)))

            # by gathering max contacts
            gathering_max_contacts = environment.parameters['gathering_max_contacts'][t]
            if gathering_max_contacts != float('inf'):
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
            likelihood_awareness = environment.parameters['likelihood_awareness'][t]
            if agent.status in environment.parameters['aware_status'] and \
                    np.random.random() < likelihood_awareness * (1 - agent.informality):
                neighbours_to_infect = [environment.agents[idx] for idx in household_neighbours]
            # otherwise the agent will interact with all neighbours from graph
            else:
                neighbours_to_infect = [environment.agents[idx] for idx in neighbours_from_graph]

            # let these agents be infected (with random probability
            physical_distancing_multiplier = environment.parameters["physical_distancing_multiplier"][t]
            for neighbour in neighbours_to_infect:
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

                    # add to detected agents with probability
                    if np.random.random() < environment.parameters["perc_infections_detects"]:
                        environment.newly_detected_cases[t] += 1

        if data_output == 'network':
            environment.infection_states.append(environment.store_network())
        elif data_output == 'csv':
            environment.write_status_location(t, seed, data_folder)
        elif data_output == 'csv_light':
            # save only the total quantity of agents per category
            for key, quantity in zip(['e', 's', 'i1', 'i2', 'c', 'r', 'd', 'detected'], [exposed,
                                                                                         susceptible,
                                                                                         sick_without_symptoms,
                                                                                         sick_with_symptoms,
                                                                                         critical, recovered, dead,
                                                                                         [x for x in range(environment.newly_detected_cases[t])]]):
                environment.infection_quantities[key].append(len(quantity))

    return environment
