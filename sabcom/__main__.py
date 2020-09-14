import sys
import click
import pickle
import os
import time
import logging
import json
import pandas as pd
import networkx as nx
import numpy as np
from scipy.integrate import odeint

from sabcom.runner import runner
from sabcom.differential_equation_model import differential_equations_model
from sabcom.environment import Environment
from sabcom.helpers import generate_district_data, what_informality


@click.group()
@click.version_option("1.0.0")
def main():
    """
    An open source, easy-to-use-and-adapt, spatial network, multi-agent, model
    that can be used to simulate the effects of different lockdown policy measures
    on the spread of the Covid-19 virus in several (South African) cities.
    """
    pass


# @main.command()
# @click.option('--input_folder_path', '-i', type=click.Path(exists=True), required=True,
#               help="This should contain all necessary input files, specifically an initialisation folder")
# @click.option('--output_folder_path', '-o', type=click.Path(exists=True), required=True,
#               help="All simulation output will be deposited here")
# @click.option('--seed', '-s', type=int, required=True,
#               help="Integer seed number that is used for Monte Carlo simulations")
# @click.option('--data_output_mode', '-d', default='csv-light', show_default=True,
#               type=click.Choice(['csv-light', 'csv', 'network'],  case_sensitive=False,))
# @click.option('--scenario', '-sc', default='no-intervention', show_default=True,
#               type=click.Choice(['no-intervention', 'lockdown', 'ineffective-lockdown'],  case_sensitive=False,))
# @click.option('--days', '-day', default=None, type=int, required=False,
#               help="change the number of simulation days here with the caveat that simulation time can only be shortened compared to what was initialised.")
# @click.option('--probability_transmission', '-pt', default=None, type=float, required=False,
#               help="change the probability of transmission between two agents.")
# @click.option('--visiting_recurring_contacts_multiplier', '-cont', default=None, type=float, required=False,
#               help="change the percentage of contacts agent may have.")
# @click.option('--likelihood_awareness', '-la', default=None, type=float, required=False,
#               help="change the likelihood that an agent is aware it is infected.")
# @click.option('--gathering_max_contacts', '-maxc', default=None, type=int, required=False,
#               help="change maximum number of contacts and agent is allowed to have.")
# @click.option('--initial_infections', '-ini', default=None, type=int, required=False,
#               help="number of initial infections")
# @click.option('--sensitivity_config_file_path', '-scf', type=click.Path(exists=True), required=False,
#               help="Config file that contains parameter combinations for sensitivity analysis on HPC")
def simulate(**kwargs):
    """Simulate the model"""
    start = time.time()

    # format arguments
    seed = kwargs.get('seed')
    output_folder_path = kwargs.get('output_folder_path')

    # logging initialisation
    logging.basicConfig(filename=os.path.join(output_folder_path,
                                              'simulation_seed{}.log'.format(seed)), filemode='w', level=logging.DEBUG)

    input_folder_path = kwargs.get('input_folder_path')
    inititialisation_path = os.path.join(input_folder_path, 'initialisations')
    seed_path = os.path.join(inititialisation_path, 'seed_{}.pkl'.format(seed))
    logging.info('Start of simulation seed{} with arguments -i ={}, -o={}'.format(seed,
                                                                                  input_folder_path,
                                                                                  output_folder_path))

    if not os.path.exists(seed_path):
        click.echo(seed_path + ' not found', err=True)
        click.echo('Error: specify a valid seed')
        return

    data = open(seed_path, "rb")
    list_of_objects = pickle.load(data)
    environment = list_of_objects[0]

    # update optional parameters
    if kwargs.get('days'):
        environment.parameters['time'] = kwargs.get('days')
        # add line to expand stringency index
        click.echo('Time has been set to {}'.format(environment.parameters['time']))
        logging.debug('Time has been set to {}'.format(environment.parameters['time']))
        # ensure that stringency is never shorter than time if time length is increased
        if len(environment.stringency_index) < environment.parameters['time']:
            environment.stringency_index += [environment.stringency_index[-1] for x in range(
                len(environment.stringency_index), environment.parameters['time'])]
        logging.debug('The stringency index has been lenghtened by {}'.format(
            environment.parameters['time'] - len(environment.stringency_index)))

    if kwargs.get('probability_transmission'):
        environment.parameters['probability_transmission'] = kwargs.get('probability_transmission')
        click.echo('Transmission probability has been set to {}'.format(environment.parameters['probability_transmission']))
        logging.debug('Transmission probability has been set to {}'.format(environment.parameters['probability_transmission']))

    if kwargs.get('visiting_recurring_contacts_multiplier'):
        environment.parameters['visiting_recurring_contacts_multiplier'] = [kwargs.get('visiting_recurring_contacts_multiplier') for x in range(environment.parameters['time'])]
        click.echo('Recurring contacts has been set to {}'.format(environment.parameters['visiting_recurring_contacts_multiplier'][0]))
        logging.debug(
            'Recurring contacts has been set to {}'.format(environment.parameters['visiting_recurring_contacts_multiplier'][0]))

    if kwargs.get('likelihood_awareness'):
        environment.parameters['likelihood_awareness'] = kwargs.get('likelihood_awareness')
        click.echo('Likelihood awareness has been set to {}'.format(environment.parameters['likelihood_awareness']))
        logging.debug(
            'Likelihood awareness has been set to {}'.format(environment.parameters['likelihood_awareness']))

    if kwargs.get('gathering_max_contacts'):
        environment.parameters['gathering_max_contacts'] = kwargs.get('gathering_max_contacts')
        click.echo('Max contacts has been set to {}'.format(environment.parameters['gathering_max_contacts']))
        logging.debug(
            'Max contacts has been set to {}'.format(environment.parameters['gathering_max_contacts']))

    if kwargs.get('initial_infections'):
        environment.parameters['initial_infections'] = [x for x in range(round(int(kwargs.get('initial_infections'))))]
        click.echo('Initial infections have been set to {}'.format(len(environment.parameters['initial_infections'])))
        logging.debug('Initial infections have been set to {}'.format(len(environment.parameters['initial_infections'])))

    if kwargs.get('sensitivity_config_file_path'):
        # open file
        config_path = kwargs.get('sensitivity_config_file_path')
        if not os.path.exists(config_path):
            click.echo(config_path + ' not found', err=True)
            click.echo('Error: specify a valid path to the sensitivity config file')
            return
        else:
            with open(config_path) as json_file:
                config_file = json.load(json_file)

                for param in config_file:
                    environment.parameters[param] = config_file[param]

    # transform input data to general district data for simulations
    district_data = generate_district_data(environment.parameters['number_of_agents'], path=input_folder_path)

    # set scenario specific parameters
    scenario = kwargs.get('scenario', 'no-intervention')
    print('scenario is ', scenario)
    if scenario == 'no-intervention':
        environment.parameters['likelihood_awareness'] = 0.0
        environment.parameters['visiting_recurring_contacts_multiplier'] = [
            1.0 for x in environment.parameters['visiting_recurring_contacts_multiplier']]
        environment.parameters['gathering_max_contacts'] = float('inf')
        environment.parameters['physical_distancing_multiplier'] = 1.0
        environment.parameters['informality_dummy'] = 0.0
    elif scenario == 'lockdown':
        environment.parameters['informality_dummy'] = 0.0
    elif scenario == 'ineffective-lockdown':
        environment.parameters['informality_dummy'] = 1.0

    # log parameters used after scenario called
    for param in environment.parameters:
        logging.debug('Parameter {} has the value {}'.format(param, environment.parameters[param]))

    for agent in environment.agents:
        agent.informality = what_informality(agent.district, district_data
                                             ) * environment.parameters["informality_dummy"]

    initial_infections = pd.read_csv(os.path.join(input_folder_path, 'f_initial_cases.csv'), index_col=0)

    # save csv light or network data
    data_output_mode = kwargs.get('data_output_mode', 'csv-light')  # default output mode is csv_light
    environment.parameters["data_output"] = data_output_mode

    environment = runner(environment=environment, initial_infections=initial_infections, seed=int(seed),
                         data_folder=output_folder_path,
                         data_output=data_output_mode)

    if data_output_mode == 'network':
        for idx, network in enumerate(environment.infection_states):
            for i, node in enumerate(network.nodes):
                network.nodes[i]['agent'] = network.nodes[i]['agent'].status
            idx_string = '{0:04}'.format(idx)
            nx.write_graphml(network, os.path.join(output_folder_path, "seed{}network_time{}.graphml".format(seed,
                                                                                                              idx_string)))
    elif data_output_mode == 'csv-light':
        pd.DataFrame(environment.infection_quantities).to_csv(os.path.join(output_folder_path,
                                                                           'seed{}quantities_state_time.csv'.format(seed)))

    end = time.time()
    hours_total, rem_total = divmod(end - start, 3600)
    minutes_total, seconds_total = divmod(rem_total, 60)
    click.echo("TOTAL RUNTIME {:0>2}:{:0>2}:{:05.2f}".format(int(hours_total), int(minutes_total), seconds_total))
    click.echo('Simulation done, check out the output data here: {}'.format(output_folder_path))


# @main.command()
# @click.option('--input_folder_path', '-i', type=click.Path(exists=True), required=True,
#               help="Folder containing parameters file, input data and an empty initialisations folder")
# @click.option('--seed', '-s', type=int, required=True,
#               help="Integer seed number that is used for Monte Carlo simulations")
# @click.option('--output_folder_path', '-o', type=click.Path(exists=True), required=True,
#               help="All simulation output will be deposited here")
# @@click.option('--output_folder_path', '-o', type=click.Path(exists=True), required=True)
def initialise(**kwargs):  # input output seed
    """Initialise the model in specified directory"""
    start = time.time()
    seed = kwargs.get('seed')

    input_folder_path = kwargs.get('input_folder_path')

    # format optional arguments
    parameters_path = os.path.join(input_folder_path, 'parameters.json')
    initialisations_folder_path = os.path.join(input_folder_path, 'initialisations')

    # logging initialisation
    logging.basicConfig(filename=os.path.join(initialisations_folder_path,
                                              'initialise_seed{}.log'.format(seed)), filemode='w', level=logging.DEBUG)
    logging.info('Start of initialisation seed{} with arguments -i ={}'.format(seed, input_folder_path))

    if not os.path.exists(initialisations_folder_path):
        click.echo(initialisations_folder_path + ' not found', err=True)
        click.echo('No initialisation folder to place initialisation pickle')
        return

    if not os.path.exists(parameters_path):
        click.echo(parameters_path + ' not found', err=True)
        click.echo('No parameter file found')
        return

    with open(parameters_path) as json_file:
        parameters = json.load(json_file)
        for param in parameters:
            logging.debug('Parameter {} is {}'.format(param, parameters[param]))

    # Change parameters depending on experiment
    #data_output_mode = kwargs.get('data_output_mode', 'csv-light')  # TODO is this still needed?

    #parameters['data_output'] = data_output_mode

    age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
                  'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

    # 2 load district data
    # transform input data to general district data for simulations
    district_data = generate_district_data(int(parameters['number_of_agents']), path=input_folder_path)

    # 2.2 age data
    age_distribution = pd.read_csv(os.path.join(input_folder_path, 'f_age_distribution.csv'), sep=',', index_col=0)
    age_distribution_per_ward = dict(age_distribution.transpose())

    # 2.3 household size distribution
    HH_size_distribution = pd.read_csv(os.path.join(input_folder_path, 'f_household_size_distribution.csv'), index_col=0)

    # 3 load travel matrix
    travel_matrix = pd.read_csv(os.path.join(input_folder_path, 'f_travel.csv'), index_col=0)

    # 4 load contact matrices
    # 4.1 load household contact matrix
    hh_contact_matrix = pd.read_csv(os.path.join(input_folder_path, 'f_household_contacts.csv'), index_col=0)
    # add a column & row for 80 plus. Rename columns to match our age categories
    hh_contact_matrix['80plus'] = hh_contact_matrix['70_80']
    row = hh_contact_matrix.xs('70_80')
    row.name = '80plus'
    hh_contact_matrix = hh_contact_matrix.append(row)
    hh_contact_matrix.columns = age_groups
    hh_contact_matrix.index = age_groups

    # 4.2 load other contact matrix
    other_contact_matrix = pd.read_csv(os.path.join(input_folder_path, 'f_nonhousehold_contacts.csv'), index_col=0)
    other_contact_matrix['80plus'] = other_contact_matrix['70_80']
    row = other_contact_matrix.xs('70_80')
    row.name = '80plus'
    other_contact_matrix = other_contact_matrix.append(row)
    other_contact_matrix.columns = age_groups
    other_contact_matrix.index = age_groups

    # make new folder if it does not exist
    if not os.path.exists('{}'.format(initialisations_folder_path)):
        os.makedirs('{}'.format(initialisations_folder_path))

    # initialisation
    environment = Environment(int(seed), parameters, district_data, age_distribution_per_ward,
                              hh_contact_matrix, other_contact_matrix, HH_size_distribution, travel_matrix)

    # save environment objects as pickls
    file_name = os.path.join(initialisations_folder_path, "seed_{}.pkl".format(str(seed)))
    save_objects = open(file_name, 'wb')
    pickle.dump([environment, seed], save_objects)
    save_objects.close()

    end = time.time()
    hours_total, rem_total = divmod(end - start, 3600)
    minutes_total, seconds_total = divmod(rem_total, 60)
    click.echo("TOTAL RUNTIME {:0>2}:{:0>2}:{:05.2f}".format(int(hours_total), int(minutes_total), seconds_total))
    click.echo('Initialisation done, check out the output data here: {}'.format(initialisations_folder_path))


@main.command()
@click.option('--input_folder_path', '-i', type=click.Path(exists=True), required=True,
              help="This should contain all necessary input files, specifically a parameter file")
@click.option('--output_folder_path', '-o', type=click.Path(exists=True), required=True,
              help="All simulation output will be deposited here")
@click.option('--r_zero', '-rz', type=float, required=True,
              help="The reproductive rate of the virus in a fully susceptible population")
def demodel(**kwargs):
    input_folder_path = kwargs.get('input_folder_path')
    output_folder_path = kwargs.get('output_folder_path')

    # logging initialisation
    logging.basicConfig(filename=os.path.join(output_folder_path,
                                              'de_model.log'), filemode='w', level=logging.DEBUG)
    logging.info('Start of DE simulation')

    parameters_path = os.path.join(input_folder_path, 'parameters.json')

    if not os.path.exists(parameters_path):
        click.echo(parameters_path + ' not found', err=True)
        click.echo('No parameter file found')
        return

    with open(parameters_path) as json_file:
        parameters = json.load(json_file)
        for param in parameters:
            logging.debug('Parameter {} is {}'.format(param, parameters[param]))

    # arguments = city
    initial_infected = len(parameters['total_initial_infections'])
    T = parameters['time']  # total number of period simulated:

    # Set Covid-19 Parameters:
    # basic reproduction number
    r_zero = kwargs.get('r_zero') #initial_recovered
    exposed_days = float(parameters["exposed_days"])
    asymptomatic_days = float(parameters["asymptom_days"])
    symptomatic_days = float(parameters["symptom_days"])
    critical_days = float(parameters["critical_days"])

    # compartment exit rates
    exit_rate_exposed = 1.0 / exposed_days
    exit_rate_asymptomatic = 1.0 / asymptomatic_days
    exit_rate_symptomatic = 1.0 / symptomatic_days
    exit_rate_critical = 1.0 / critical_days

    probability_symptomatic = parameters["probability_symptomatic"]
    # Probability to become critically ill if symptomatic (source: Verity et al.2020)
    probability_critical = np.array([x for x in parameters["probability_critical"].values()])
    # Probability to die if critically ill (source: Silal et al.2020)
    probability_to_die = np.array([x for x in parameters["probability_critical"].values()])

    # Total population:
    district_population = pd.read_csv(os.path.join(input_folder_path, 'f_population.csv'), index_col=0)
    district_population = district_population.values
    population = district_population.sum()  # sum over wards to obtain city population

    # Set city specific parameters
    hospital_capacity = int(round(parameters["health_system_capacity"] * population))

    # Population by age group (N_age(i) is the population of age group i)
    ward_age_distribution = pd.read_csv(os.path.join(input_folder_path, 'f_age_distribution.csv'),
                                        index_col=0)  # the datafile contains ward level fractions in each age group
    N_age = ward_age_distribution * district_population  # convert to number of people in age group per ward
    N_age = N_age.sum()  # sum over wards
    N_age = N_age.values  # store city level population sizes of each age group

    # Load raw contact matrices
    household_contacts = pd.read_csv(os.path.join(input_folder_path, 'f_household_contacts.csv'), index_col=0)
    other_contacts = pd.read_csv(os.path.join(input_folder_path, 'f_nonhousehold_contacts.csv'), index_col=0)
    contact_matrix = household_contacts + other_contacts
    contact_matrix = contact_matrix.values

    # Replicate last row and column to change the 8 category contact matrix to a 9 category matrix
    contact_matrix = np.vstack((contact_matrix, contact_matrix[7, 0:8]))
    C_last_column = contact_matrix[0:9, 7]
    C_last_column.shape = (9, 1)
    contact_matrix = np.hstack((contact_matrix, C_last_column))

    # Apply reciprocity correction (see Towers and Feng (2012))
    # C_corrected(j,k) = (C(j,k)*N(j) + C(k,j)*N(k))/(2*N(j))
    for j in range(contact_matrix.shape[0]):
        for k in range(contact_matrix.shape[0]):
            contact_matrix[j, k] = (contact_matrix[j, k] * N_age[j] + contact_matrix[k, j] * N_age[k]) / (2 * N_age[j])

    # Scale contact matrix by population size
    # - each column is normalized by the population of that age group: X(i,j)=C(i,j)/N_age(j)
    N_age_row_vector = np.array(N_age)
    N_age_row_vector.shape = (1, 9)
    contact_probability_matrix = np.divide(contact_matrix,
                                           N_age_row_vector)  # X(i,j)=C(i,j)/N_age(j) - entries now measure fraction of each age group contacted on average per day

    # Compute infection_rate from R0, exit_rate_asymptomatic, e_S and dominant eigenvalue of matrix X(i,j)*N_age(i)
    N_age_column_vector = np.array(N_age)
    N_age_column_vector.shape = (9, 1)
    eigen_values, eigen_vectors = np.linalg.eig(np.multiply(contact_probability_matrix, N_age_column_vector))
    dom_eig_val = max(eigen_values)

    infection_rate = (((1 - probability_symptomatic) * exit_rate_asymptomatic + probability_symptomatic * exit_rate_symptomatic) * r_zero) / dom_eig_val
    click.echo('infection rate (beta) is {}'.format(round(infection_rate, 4)))

    # Set initial conditions
    # spread initial infections (exposed individuals) across age groups equally
    initial_exposed = (initial_infected / 9) * np.ones(9)
    # compute remaining initial populations in susceptible compartments
    initial_susceptible = N_age - initial_exposed
    # initiallise other compartments at zero
    initial_asymptomatic = np.zeros(9)
    initial_symptomatic = np.zeros(9)
    initial_critical = np.zeros(9)
    initial_recovered = np.zeros(9)
    initial_dead = np.zeros(9)

    # Solve model over time from initial conditions, using ODE solver from scipy:
    time_points = np.linspace(1, T, T)  # Grid of time points (in days)
    initial_compartments = np.concatenate((initial_susceptible, initial_exposed, initial_asymptomatic,
                                           initial_symptomatic, initial_critical, initial_recovered, initial_dead),
                                          axis=0)

    # Integrate the differential equations over the time grid, t.
    integrals = odeint(differential_equations_model, initial_compartments, time_points, args=(
        infection_rate, contact_probability_matrix, exit_rate_exposed, exit_rate_asymptomatic, exit_rate_symptomatic,
        exit_rate_critical, probability_symptomatic, probability_critical, probability_to_die, hospital_capacity))

    # integrals is T by 63, needs to be split in compartments, each disease compartments has 9 age groups
    susceptible = integrals[:, 0:9].sum(axis=1)
    exposed = integrals[:, 9:18].sum(axis=1)
    asymptomatic = integrals[:, 18:27].sum(axis=1)
    symptomatic = integrals[:, 27:36].sum(axis=1)
    critical = integrals[:, 36:45].sum(axis=1)
    recovered = integrals[:, 45:54].sum(axis=1)
    dead = integrals[:, 54:63].sum(axis=1)

    infected = exposed + asymptomatic + symptomatic + critical + dead + recovered
    active_infections = exposed + asymptomatic + symptomatic + critical
    click.echo('Peak of disease:')
    click.echo('peak critical = {}'.format(round(max(critical))))
    click.echo('peak infected = {}'.format(round(max(active_infections))))
    click.echo('time-period at peak = day {}'.format(np.argmax(active_infections)))
    click.echo('At end of simulation:')
    click.echo('total infected = {} ({} percent of population)'.format(round(infected[T - 1]),
                                                                       round(infected[T - 1] * 100 / population, 2)))
    click.echo('total deceased = {}, ({} percent of infected)'.format(round(dead[T - 1]),
                                                                      round(dead[T - 1] * 100 / infected[T - 1], 2)))

    # export data
    pd.DataFrame({'s': susceptible, 'e': exposed, 'i1': asymptomatic,
                  'i2': symptomatic, 'c': critical, 'r': recovered, 'd': dead}).to_csv(
        os.path.join(output_folder_path, 'DE_quantities_state_time.csv'))

    click.echo('DE model simulation done, check out the output data here: {}'.format(output_folder_path))


if __name__ == '__main__':
    args = sys.argv
    if "--help" in args or len(args) == 1:
        print("SABCoM")
    main()
