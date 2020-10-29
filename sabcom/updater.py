import random
import numpy as np
import scipy.stats as stats


def updater(environment, initial_infections, seed, data_folder='output_data/',
            data_output=False):
    """
    This function is used to run / simulate the model.

    :param environment: contains the parameters and agents, Environment object
    :param initial_infections: contains the Wards and corresponding initial infections, Pandas DataFrame
    :param seed: used to initialise the random generators to ensure reproducibility, int
    :param data_folder:  string of the folder where data output files should be created
    :param data_output:  can be 'csv', 'network', or False (for no output)
    :return: environment object containing the updated agents, Environment object
    """
    # 1 set monte carlo seed
    np.random.seed(seed)
    random.seed(seed)

    # 2 create sets for all agent types
    dead = []
    recovered = []
    critical = []
    sick_with_symptoms = []
    sick_without_symptoms = []
    exposed = []
    susceptible = [agent for agent in environment.agents]

    # 3 Initialisation of infections
    initial_infections = initial_infections.sort_index()
    cases = [x for x in initial_infections['Cases']]
    probabilities_new_infection_district = [float(i) / sum(cases) for i in cases]

    # 3.1 select districts with probability
    chosen_districts = list(np.random.choice(environment.districts,
                                             environment.parameters['total_initial_infections'],
                                             p=probabilities_new_infection_district))
    # 3.2 count how often a district is in that list
    chosen_districts = {distr: min(len(environment.district_agents[distr]),
                                   chosen_districts.count(distr)) for distr in chosen_districts}

    for district in chosen_districts:
        # 3.3 infect appropriate number of random agents
        chosen_agents = np.random.choice(environment.district_agents[district], chosen_districts[district],
                                         replace=False)
        categories = ['e', 'i1', 'i2']
        # 3.4 and give them a random status exposed, asymptomatic, or symptomatic with a random number of days
        # already passed being in that state
        for chosen_agent in chosen_agents:
            new_status = random.choice(categories)
            chosen_agent.status = new_status
            if new_status == 'e':
                chosen_agent.incubation_days = np.random.randint(0, environment.parameters['exposed_days'])
                exposed.append(chosen_agent)
            elif new_status == 'i1':
                chosen_agent.asymptomatic_days = np.random.randint(0, environment.parameters['asymptom_days'])
                sick_without_symptoms.append(chosen_agent)
            elif new_status == 'i2':
                chosen_agent.sick_days = np.random.randint(0, environment.parameters['symptom_days'])
                sick_with_symptoms.append(chosen_agent)

            susceptible.remove(chosen_agent)

    # 4 day loop
    for t in range(environment.parameters["time"]):
        # keep track of average contacts and compliance
        contacts = []
        compliance = []

        # 4.1 check if the health system is not overburdened
        if len(critical) / len(environment.agents) > environment.parameters["health_system_capacity"]:
            health_overburdened_multiplier = environment.parameters["no_hospital_multiplier"]
        else:
            health_overburdened_multiplier = 1.0

        # 4.2 create a generator to generate shocks for private signal for this period based on current stringency index
        lower, upper = -(environment.stringency_index[t] / 100), (1 - (environment.stringency_index[t] / 100))
        mu, sigma = 0.0, environment.parameters['private_shock_stdev']
        shocks = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma,
                                     size=len(susceptible + exposed + sick_without_symptoms + sick_with_symptoms + critical + recovered))

        # 4.3 update status loop for all agents, except dead agents
        visiting_r_contacts_multiplier = environment.parameters["visiting_recurring_contacts_multiplier"]
        for i, agent in enumerate(susceptible + exposed + sick_without_symptoms + sick_with_symptoms + critical + recovered):
            # 4.3.1 save compliance to previous compliance and record private signal
            agent.previous_compliance = agent.compliance
            private_signal = environment.stringency_index[t] / 100 + shocks[i]

            # 4.3.2 loop over neighbours for learning and record effective contacts
            household_neighbours = []
            other_neighbours = []
            likelihood_to_meet_neighbours = []
            total_compliance = []
            for nb in environment.network.neighbors(agent.name):
                total_compliance.append(environment.agents[nb].previous_compliance)

                # make a distinction between household and non-household neighbours
                if environment.agents[nb].household_number == agent.household_number and environment.agents[nb].district == agent.district:
                    household_neighbours.append(nb)
                else:
                    other_neighbours.append(nb)
                    # record likelihood to meet neighbours
                    neighbour_compliance_term = (1 - visiting_r_contacts_multiplier) * (
                            1 - environment.agents[nb].compliance)
                    agent_compliance_term = (1 - visiting_r_contacts_multiplier) * (1 - agent.compliance)
                    likelihood_to_meet = (visiting_r_contacts_multiplier + agent_compliance_term) * (
                            visiting_r_contacts_multiplier + neighbour_compliance_term)
                    likelihood_to_meet_neighbours.append(likelihood_to_meet)
            if total_compliance:
                neighbour_signal = np.mean(total_compliance)
            else:
                neighbour_signal = private_signal

            if other_neighbours:
                average_neighbours_met = np.mean(likelihood_to_meet_neighbours)
            else:
                average_neighbours_met = 1.0

            total_contacts = len(household_neighbours) + (len(other_neighbours) * average_neighbours_met)
            contacts.append(total_contacts)

            agent.compliance = (1 - agent.informality) * \
                               (environment.parameters['weight_private_signal'] * private_signal +
                                (1 - environment.parameters['weight_private_signal']) * neighbour_signal)

            # 4.3.3 update the disease status of the agent and possibly infect others
            if agent.status == 's' and agent.period_to_become_infected == t:
                agent.status = 'e'
                susceptible.remove(agent)
                exposed.append(agent)

            elif agent.status == 'e':
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

            # any agent with status i1, or i2 might first infect other agents and then will update her status
            elif agent.status in ['i1', 'i2']:
                # check if the agent is aware that is is infected here and set compliance to 1.0 if so
                if agent.status == 'i2':
                    agent.compliance = 1.0

                # Infect others / TAG SUSCEPTIBLE AGENTS FOR INFECTION
                agent.others_infected = 0
                for nb in environment.network.neighbors(agent.name):
                    if environment.agents[nb].status == 's':
                        if environment.agents[nb].name in other_neighbours:
                            # for other neighbours determine the
                            neighbour_compliance_term = (1 - visiting_r_contacts_multiplier) * (
                                        1 - environment.agents[nb].compliance)
                            agent_compliance_term = (1 - visiting_r_contacts_multiplier) * (1 - agent.compliance)
                            likelihood_to_meet = (visiting_r_contacts_multiplier + agent_compliance_term) * (visiting_r_contacts_multiplier + neighbour_compliance_term)

                        else:
                            likelihood_to_meet = 1.0

                        if np.random.random() < environment.parameters['probability_transmission'] and np.random.random() < likelihood_to_meet:
                            environment.agents[nb].period_to_become_infected = t + 1
                            agent.others_infected += 1
                            agent.others_infects_total += 1

                # update current status based on category
                if agent.status == 'i1':
                    agent.asymptomatic_days += 1
                    # asymptomatic agents all recover after some time
                    if agent.asymptomatic_days > environment.parameters["asymptom_days"]:
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
                            agent.status = 'r'
                            sick_with_symptoms.remove(agent)
                            recovered.append(agent)

            elif agent.status == 'c':
                agent.compliance = 1.0
                agent.critical_days += 1
                # some agents in critical status will die, the rest will recover
                if agent.critical_days > environment.parameters["critical_days"]:
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

            # record compliance
            compliance.append(agent.compliance)

        # 5 allow for the possibility of new infections
        if t == environment.parameters['time_4_new_infections']:
            if environment.parameters['new_infections_scenario'] == 'initial':
                cases = [x for x in initial_infections['Cases']]
                probabilities_second_infection_district = [float(i) / sum(cases) for i in cases]
                # select districts with probability
                chosen_districts = list(np.random.choice(environment.districts,
                                                         environment.parameters['second_infection_n'],
                                                         p=probabilities_second_infection_district))
                # count how often a district is in that list
                chosen_districts = {distr: min(len(environment.district_agents[distr]),
                                               chosen_districts.count(distr)) for distr in chosen_districts}

            elif environment.parameters['new_infections_scenario'] == 'random':
                cases = [1 for x in initial_infections['Cases']]
                probabilities_second_infection_district = [float(i) / sum(cases) for i in cases]
                # select districts with probability
                chosen_districts = list(np.random.choice(environment.districts,
                                                         environment.parameters['second_infection_n'],
                                                         p=probabilities_second_infection_district))
                # count how often a district is in that list
                chosen_districts = {distr: min(len(environment.district_agents[distr]),
                                               chosen_districts.count(distr)) for distr in chosen_districts}
            else:
                chosen_districts = []

            for district in chosen_districts:
                # infect appropriate number of random agents
                chosen_agents = np.random.choice(environment.district_agents[district], chosen_districts[district],
                                                 replace=False)
                categories = ['e', 'i1', 'i2']
                for chosen_agent in chosen_agents:
                    if chosen_agent.status == 's':
                        new_status = random.choice(categories)
                        chosen_agent.status = new_status
                        if new_status == 'e':
                            chosen_agent.incubation_days = np.random.randint(0, environment.parameters['exposed_days'])
                            exposed.append(chosen_agent)
                        elif new_status == 'i1':
                            chosen_agent.asymptomatic_days = np.random.randint(0,
                                                                               environment.parameters['asymptom_days'])
                            sick_without_symptoms.append(chosen_agent)
                        elif new_status == 'i2':
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
            # sometimes there are no contacts
            if contacts:
                environment.infection_quantities['contacts'].append(np.mean(contacts))
            else:
                environment.infection_quantities['contacts'].append(np.nan)

    return environment
