import numpy as np
import scipy.stats as stats
import random


def runner(environment, initial_infections, seed, data_folder='output_data/',
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
    compliance = []

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
        cases = [x for x in initial_infections['Cases']]
        probabilities_new_infection_district = [float(i) / sum(cases) for i in cases]

        initial_infected = []
        # select districts with probability
        chosen_districts = list(np.random.choice(environment.districts,
                                                 environment.parameters['total_initial_infections'],
                                                 p=probabilities_new_infection_district))
        # count how often a district is in that list
        chosen_districts = {distr: min(len(environment.district_agents[distr]),
                                       chosen_districts.count(distr)) for distr in chosen_districts}

        for district in chosen_districts:
            # infect appropriate number of random agents
            chosen_agents = np.random.choice(environment.district_agents[district], chosen_districts[district],
                                             replace=False)
            for chosen_agent in chosen_agents:
                chosen_agent.status = 'i2'
                # give i2 days a random value to avoid an unrealistic wave of initial critical cases and deaths
                chosen_agent.sick_days = np.random.randint(0, environment.parameters['symptom_days'])
                sick_with_symptoms.append(chosen_agent)
                susceptible.remove(chosen_agent)

    for t in range(environment.parameters["time"]):
        print(t)
        # Check if the health system is not overburdened
        if len(critical) / len(environment.agents) > environment.parameters["health_system_capacity"]:
            health_overburdened_multiplier = environment.parameters["no_hospital_multiplier"]
        else:
            health_overburdened_multiplier = 1.0

        # create truncnorm generator to generate shocks for this period based on current stringency index
        lower, upper = -(environment.stringency_index[t] / 100), (1 - (environment.stringency_index[t] / 100))
        mu, sigma = 0.0, environment.parameters['private_shock_stdev']
        truncnorm_shock_generator = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        # 5 update status loop
        for agent in susceptible + exposed + sick_without_symptoms + sick_with_symptoms + critical + recovered:
            # construct deGroot signals
            # save compliance to previous compliance
            agent.previous_compliance = agent.compliance

            # find indices from neighbour agents
            neighbours_to_learn_from = [environment.agents[x] for x in environment.network.neighbors(agent.name)]

            private_signal = environment.stringency_index[t] / 100 + truncnorm_shock_generator.rvs(1)[0]
            if neighbours_to_learn_from: # sometimes an agent has no neighbours
                neighbour_signal = np.mean([x.previous_compliance for x in neighbours_to_learn_from])
            else:
                neighbour_signal = private_signal

            agent.compliance = (1 - agent.informality) * \
                               (environment.parameters['weight_private_signal'] * private_signal +
                                (1 - environment.parameters['weight_private_signal']) * neighbour_signal)

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
                # check if the agent is aware that is is infected here and set compliance to 1.0 if so
                likelihood_awareness = environment.parameters['likelihood_awareness']# * (
                            # environment.stringency_index[t] / 100) * (
                            #                        1 - agent.informality)
                if np.random.random() < likelihood_awareness:
                    agent.compliance = 1.0

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
                agent.compliance = 1.0  # TODO debug! make sure that this remains so when dead
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

            compliance.append(agent.compliance)

        # 6 New infections loop
        for agent in sick_without_symptoms + sick_with_symptoms:
            agent.others_infected = 0

            # find indices from neighbour agents
            household_neighbours = [x for x in environment.network.neighbors(agent.name) if
                                    environment.agents[x].household_number == agent.household_number and
                                    environment.agents[x].district == agent.district]
            other_neighbours = [x for x in environment.network.neighbors(agent.name) if
                                environment.agents[x].household_number != agent.household_number or
                                environment.agents[x].district != agent.district]

            # depending on compliance, the amount of non-household contacts an agent can visit is reduced
            visiting_r_contacts_multiplier = environment.parameters["visiting_recurring_contacts_multiplier"][t] # TODO debug!
            compliance_term_contacts = (1 - visiting_r_contacts_multiplier) * (1 - agent.compliance)

            # step 1 planned contacts is shaped by
            if other_neighbours:
                planned_contacts = int(round(len(other_neighbours
                                                 ) * (visiting_r_contacts_multiplier + compliance_term_contacts))) # * (visiting_r_contacts_multiplier +
            else:
                planned_contacts = 0

            # step 2 by gathering max contacts
            gathering_max_contacts = environment.parameters['gathering_max_contacts']
            if gathering_max_contacts != float('inf'):
                gathering_max_contacts = round(gathering_max_contacts * (1 + (1 - (environment.stringency_index[t] / 100))))
                individual_max_contacts = int(round(gathering_max_contacts * (1 + (1 - agent.compliance))))
            else:
                individual_max_contacts = gathering_max_contacts

            if planned_contacts > individual_max_contacts:
                other_neighbours = random.sample(other_neighbours, individual_max_contacts)
            else:
                other_neighbours = random.sample(other_neighbours, planned_contacts)

            # step 3 combine household neighbours with other neighbours
            neighbours_from_graph = household_neighbours + other_neighbours
            # step 4 find the corresponding agents and add them to a list to infect
            if agent.status in ['i1', 'i2']:
                if agent.compliance == 1.0: #TODO debug, this means an agent with 1.0 compliance will self-isolate
                    neighbours_to_infect = [environment.agents[idx] for idx in household_neighbours]
                else:
                    neighbours_to_infect = [environment.agents[idx] for idx in neighbours_from_graph]
                # step 4 let these agents be infected (with random probability
                physical_distancing_multiplier = environment.parameters["physical_distancing_multiplier"] #1 - ((1 - environment.parameters["physical_distancing_multiplier"]) * agent.compliance)
                for neighbour in neighbours_to_infect:
                    if neighbour.household_number == agent.household_number and neighbour.district == agent.district:
                        compliance_term_phys_dis = 0.0#(1 - physical_distancing_multiplier)
                        compliance_term_phys_dis_neighbour = 0.0
                    else:
                        compliance_term_phys_dis = (1 - physical_distancing_multiplier) * (1 - agent.compliance)
                        # TODO debug NEW! takes into account the compliance of two neighbours
                        compliance_term_phys_dis_neighbour = (1 - physical_distancing_multiplier) * (1 - neighbour.compliance)

                    if neighbour.status == 's' and np.random.random() < (
                            environment.parameters['probability_transmission'] * (
                            physical_distancing_multiplier + compliance_term_phys_dis) * (
                            physical_distancing_multiplier + compliance_term_phys_dis_neighbour)):
                        neighbour.status = 'e'
                        susceptible.remove(neighbour)
                        exposed.append(neighbour)
                        agent.others_infected += 1
                        agent.others_infects_total += 1

        # NEW infections!!! TODO debug!
        if t == environment.parameters['time_4_new_infections']:
            if environment.parameters['new_infections_scenario'] == 'initial':
                cases = [x for x in initial_infections['Cases']]
                probabilities_second_infection_district = [float(i) / sum(cases) for i in cases]
                # select districts with probability
                chosen_districts = list(np.random.choice(environment.districts,
                                                         environment.parameters['total_initial_infections'],
                                                         p=probabilities_second_infection_district))
                # count how often a district is in that list
                chosen_districts = {distr: min(len(environment.district_agents[distr]),
                                               chosen_districts.count(distr)) for distr in chosen_districts}

            elif environment.parameters['new_infections_scenario'] == 'random':
                cases = [1 for x in initial_infections['Cases']] #TODO debug this should lead to a uniform distribution
                probabilities_second_infection_district = [float(i) / sum(cases) for i in cases]
                # select districts with probability
                chosen_districts = list(np.random.choice(environment.districts,
                                                         environment.parameters['total_initial_infections'],
                                                         p=probabilities_second_infection_district))
                # count how often a district is in that list
                chosen_districts = {distr: min(len(environment.district_agents[distr]),
                                               chosen_districts.count(distr)) for distr in chosen_districts}
            else:
                chosen_districts = []  # TODO debug ..

            for district in chosen_districts:
                # infect appropriate number of random agents
                chosen_agents = np.random.choice(environment.district_agents[district], chosen_districts[district],
                                                 replace=False)
                for chosen_agent in chosen_agents:
                    if chosen_agent.status == 's':
                        chosen_agent.status = 'i2'
                        # give i2 days a random value to avoid an unrealistic wave of initial critical cases and deaths
                        chosen_agent.sick_days = np.random.randint(0, environment.parameters['symptom_days'])
                        sick_with_symptoms.append(chosen_agent)
                        susceptible.remove(chosen_agent)

        if data_output == 'network':
            environment.infection_states.append(environment.store_network())
        elif data_output == 'csv':
            environment.write_status_location(t, seed, data_folder)
        elif data_output == 'csv-light':
            # save only the total quantity of agents per category
            for key, quantity in zip(['e', 's', 'i1', 'i2',
                                      'c', 'r', 'd'],
                                     [exposed, susceptible, sick_without_symptoms, sick_with_symptoms,
                                      critical, recovered, dead]):
                environment.infection_quantities[key].append(len(quantity))
            environment.infection_quantities['compliance'].append(np.mean(compliance))

    return environment
