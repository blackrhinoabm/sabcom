import sys
import click
import pickle
import os
import time
import copy
import logging
import json
import pandas as pd
import networkx as nx
import numpy as np
from scipy.integrate import odeint
from SALib.sample import latin

from sabcom.updater import updater
from sabcom.estimation import ls_model_performance, constrNM
from sabcom.differential_equation_model import differential_equations_model
from sabcom.environment import Environment
from sabcom.helpers import generate_district_data


@click.group()
@click.version_option("1.0.0")
def main():
    """
    An open source, easy-to-use-and-adapt, spatial network, multi-agent, model
    that can be used to simulate the effects of different lockdown policy measures
    on the spread of the Covid-19 virus in several (South African) cities.
    """
    pass


@main.command()
@click.option('--input_folder_path', '-i', type=click.Path(exists=True), required=True,
              help="This should contain all necessary input files, specifically an initialisation folder")
@click.option('--output_folder_path', '-o', type=click.Path(exists=True), required=True,
              help="All simulation output will be deposited here")
@click.option('--seed', '-s', type=int, required=True,
              help="Integer seed number that is used for Monte Carlo simulations")
@click.option('--data_output_mode', '-d', default='csv-light', show_default=True,
              type=click.Choice(['csv-light', 'csv', 'network'],  case_sensitive=False,))
@click.option('--scenario', '-sc', default='no-intervention', show_default=True,
              type=click.Choice(['no-intervention', 'lockdown', 'ineffective-lockdown'],  case_sensitive=False,))
@click.option('--days', '-day', default=None, type=int, required=False,
              help="integer that sets the number of simulation days")
@click.option('--probability_transmission', '-pt', default=None, type=float, required=False,
              help="change the probability of transmission between two agents.")
@click.option('--visiting_recurring_contacts_multiplier', '-cont', default=None, type=float, required=False,
              help="change the percentage of contacts agent may have.")
@click.option('--likelihood_awareness', '-la', default=None, type=float, required=False,
              help="change the likelihood that an agent is aware it is infected.")
@click.option('--gathering_max_contacts', '-maxc', default=None, type=int, required=False,
              help="change maximum number of contacts and agent is allowed to have.")
@click.option('--initial_infections', '-ini', default=None, type=int, required=False,
              help="number of initial infections")
@click.option('--sensitivity_config_file_path', '-scf', type=click.Path(exists=True), required=False,
              help="Config file that contains parameter combinations for sensitivity analysis on HPC")
def simulate(**kwargs):
    """
    This function is used to run / simulate the model. It will first load and, optionally, change the initialisation.
    Then, it will call the runner function and simulate the model over time. Finally, it will output data one or
    multiple files.

    :param kwargs: dictionary containing the following parameters
    input_folder_path or -i: path that contain all necessary input files, String
    output_folder_path or -o: path to output folder, String
    seed or -s: used to initialise the random generators to ensure reproducibility, int
    scenario or -sc: one of three possible scenarios, 'no-intervention', 'lockdown', 'ineffective-lockdown'
    days or -day:  sets the number of simulation days, int
    probability_transmission or -pt: change the probability of transmission between two agents
    visiting_recurring_contacts_multiplier, or -cont: change the percentage of contacts agent may have, float
    likelihood_awareness or -la: change the likelihood that an agent is aware it is infected., float
    gathering_max_contacts or -maxc: change maximum number of contacts and agent is allowed to have, int
    initial_infections or -ini: number of initial infections, int
    sensitivity_config_file_path or -scf: path to config file with parameters for sensitivity analysis on HPC, str
    :return: None
    """
    start = time.time()

    # simulate the model and return an updated environment
    environment = updater(**kwargs)

    output_folder_path = kwargs.get('output_folder_path')
    seed = kwargs.get('seed')

    if environment.parameters["data_output"] == 'network':
        for idx, network in enumerate(environment.infection_states):
            for i, node in enumerate(network.nodes):
                network.nodes[i]['agent'] = network.nodes[i]['agent'].status
            idx_string = '{0:04}'.format(idx)
            nx.write_graphml(network, os.path.join(output_folder_path, "seed{}network_time{}.graphml".format(seed,
                                                                                                              idx_string)))
    elif environment.parameters["data_output"] == 'csv-light':
        pd.DataFrame(environment.infection_quantities).to_csv(os.path.join(output_folder_path,
                                                                           'seed{}quantities_state_time.csv'.format(seed)))

    end = time.time()
    hours_total, rem_total = divmod(end - start, 3600)
    minutes_total, seconds_total = divmod(rem_total, 60)
    click.echo("TOTAL RUNTIME {:0>2}:{:0>2}:{:05.2f}".format(int(hours_total), int(minutes_total), seconds_total))
    click.echo('Simulation done, check out the output data here: {}'.format(output_folder_path))


@main.command()
@click.option('--input_folder_path', '-i', type=click.Path(exists=True), required=True,
              help="Folder containing parameters file, input data and an empty initialisations folder")
@click.option('--seed', '-s', type=int, required=True,
              help="Integer seed number that is used for Monte Carlo simulations")
def initialise(**kwargs):
    """
    This function is used to initialise the model with a particular random seed and input folder data files.
    It will output the initialisation as a .pkl file in the initialisations folder in the input folder path.

    :param kwargs: dictionary containing the following parameters
    input_folder_path or -i: path that contain all necessary input files, String
    seed or -s: used to initialise the random generators to ensure reproducibility, int
    :return: None
    """
    start = time.time()
    seed = kwargs.get('seed')

    input_folder_path = kwargs.get('input_folder_path')

    # format optional arguments
    parameters_path = os.path.join(input_folder_path, 'parameters.json')
    if not os.path.exists(parameters_path):
        click.echo(parameters_path + ' not found', err=True)
        click.echo('No parameter file found')
        return

    initialisations_folder_path = os.path.join(input_folder_path, 'initialisations')
    if not os.path.exists(initialisations_folder_path):
        click.echo(initialisations_folder_path + ' not found', err=True)
        click.echo('No initialisation folder to place initialisation pickle')
        return

    # logging initialisation
    logging.basicConfig(filename=os.path.join(initialisations_folder_path,
                                              'initialise_seed{}.log'.format(seed)), filemode='w', level=logging.DEBUG)
    logging.info('Start of initialisation seed{} with arguments -i ={}'.format(seed, input_folder_path))

    # open parameters from json file
    with open(parameters_path) as json_file:
        parameters = json.load(json_file)
        for param in parameters:
            logging.debug('Parameter {} is {}'.format(param, parameters[param]))

    age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
                  'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

    # 2 load district data
    # transform input data to general district data for simulations
    district_data = generate_district_data(int(parameters['number_of_agents']), path=input_folder_path)

    # 2.2 age data
    age_distribution = pd.read_csv(os.path.join(input_folder_path, 'f_age_distribution.csv'), sep=',', index_col=0)
    age_distribution_per_ward = dict(age_distribution.transpose())

    # 2.3 household size distribution
    household_size_distribution = pd.read_csv(os.path.join(input_folder_path, 'f_household_size_distribution.csv'),
                                              index_col=0)

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
                              hh_contact_matrix, other_contact_matrix, household_size_distribution, travel_matrix)

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
@click.option('--problem_file_path', '-prob', type=click.Path(exists=True), required=True,
              help="Link to a json that contains the problem with parameters and bounds")
@click.option('--n_samples', '-n', type=int, default=1, required=False,
              help="The number of parameter samples that should be returned")
@click.option('--output_file_name', '-ofn',  required=False,
              help="Change the name of the output file to one of your liking")
def sample(**kwargs):
    """
    :param kwargs: dictionary containing the following parameters
    problem_file_path or -prob: path  to a json that contains the problem with parameters and bounds, String
    n_samples or -n: the number of parameter samples that should be returned,
    :return: None
    """
    n_samples = kwargs.get('n_samples')
    problem_file_path = kwargs.get('problem_file_path')
    if kwargs.get('output_file_name'):
        output_file_name = kwargs.get('output_file_name')
    else:
        output_file_name = 'hypercube.json'

    with open(problem_file_path) as json_file:
        problem = json.load(json_file)

    latin_hyper_cube = latin.sample(problem=problem, N=n_samples)
    latin_hyper_cube = latin_hyper_cube.tolist()
    # create list of problems with initial values
    problems = []
    for pars in latin_hyper_cube:
        # transform the necessary parameters into integers
        for idx, p in enumerate(pars):
            if problem['integer'][idx]:
                pars[idx] = round(int(pars[idx]))

        new_problem = copy.deepcopy(problem)
        new_problem['initial'] = pars
        problems.append(new_problem)

    with open(output_file_name, 'w') as f:
        json.dump(problems, f)

    click.echo('Sampling {} samples done, check out the samples here: {}'.format(n_samples, output_file_name))


@main.command()
@click.option('--input_folder_path', '-i', type=click.Path(exists=True), required=True,
              help="This should contain all necessary input files, specifically an initialisation folder")
@click.option('--output_folder_path', '-o', type=click.Path(exists=True), required=True,
              help="the estimated parameters will be deposited in this folder")
@click.option('--scenario', '-sc', default='no-intervention', show_default=True,
              type=click.Choice(['no-intervention', 'lockdown', 'ineffective-lockdown'],  case_sensitive=False,))
@click.option('--problems_file_path', '-pfp', type=click.Path(exists=True), required=True,
              help="All simulation output will be deposited here")
@click.option('--n_seeds', '-n', type=int, default=1, required=False,
              help="The number of seeds that should be simulated")
@click.option('--iterations', '-iter', type=int, default=1, required=False,
              help="The number iterations the Nelder-Mead optimisers should do")
@click.option('--output_file_name', '-ofn', required=False,
              help="Change the name of the output file to one of your liking, default is estimated_parameters")
def estimate(**kwargs):
    """
    Estimates uncertain parameters with Nelder-Mead optimisation by fitting simulated deaths to observed excess deaths
    and stores them as a json file
    :param kwargs:
    :return: None
    """
    if kwargs.get('output_file_name'):
        output_file_name = kwargs.get('output_file_name')
    else:
        output_file_name = 'estimated_parameters.json'

    # 1 load problems
    with open(kwargs.get('problems_file_path')) as json_file:
        problems = json.load(json_file)

    # 2 for every initial parameter set find optimal output
    estimated_parameters = []
    average_costs = []
    for pr in problems:
        LB = [x[0] for x in pr['bounds']]
        UB = [x[1] for x in pr['bounds']]
        init_vars = [x for x in pr['initial']]

        args = (kwargs.get('input_folder_path'), kwargs.get('n_seeds'),
                kwargs.get('output_folder_path'), kwargs.get('scenario'))

        output = constrNM(ls_model_performance, init_vars, LB, UB, args=args,
                          maxiter=kwargs.get('iterations'), full_output=True) #TODO debug

        estimated_parameters.append(output['xopt'])
        average_costs.append(output['fopt'])
        click.echo('Average cost was {}'.format(output['fopt']))

    # 3 output in file the optimal uncertain parameters
    lowest_cost_idx = np.argmin(np.array(average_costs))
    estimated_parameters = {'names': list(problems[0]['names']), 'estimates': list(estimated_parameters[lowest_cost_idx]),
                            'cost': average_costs[lowest_cost_idx]}

    click.echo('Estimated parameter values are {}'.format(estimated_parameters['estimates']))
    click.echo('Estimated parameter cost is {}'.format(average_costs[lowest_cost_idx]))

    parameters_path = os.path.join(kwargs.get('input_folder_path'), 'parameters.json')
    with open(parameters_path) as json_file:
        standard_params = json.load(json_file)

    for x, y in zip(estimated_parameters['names'], estimated_parameters['estimates']):
        if x == 'visiting_recurring_contacts_multiplier':
            standard_params[x] = [y for i in standard_params[x]]
        elif x in ["gathering_max_contacts", 'total_initial_infections']:
            standard_params[x] = int(round(y))
        else:
            standard_params[x] = y

    with open(os.path.join(kwargs.get('input_folder_path'), output_file_name), 'w') as f:
        json.dump(standard_params, f)


@main.command()
@click.option('--input_folder_path', '-i', type=click.Path(exists=True), required=True,
              help="This should contain all necessary input files, specifically a parameter file")
@click.option('--output_folder_path', '-o', type=click.Path(exists=True), required=True,
              help="All simulation output will be deposited here")
@click.option('--r_zero', '-rz', type=float, required=True,
              help="The reproductive rate of the virus in a fully susceptible population")
def demodel(**kwargs):
    """
    This function is used to run / simulate a differential equation version of Sabcom.
    Finally, it will output data a data file with data on how the virus spread over time.

    :param kwargs: dictionary containing the following parameters
    input_folder_path or -i: path that contain all necessary input files, String
    output_folder_path or -o: path to output folder, String
    r_zero or -rz: the reproductive rate of the virus in a fully susceptible population, float
    :return: None
    """
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
    simulation_time = parameters['time']  # total number of period simulated

    # Set Covid-19 Parameters:
    r_zero = kwargs.get('r_zero')
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
    probability_critical = np.array([x for x in parameters["probability_critical"].values()])
    probability_to_die = np.array([x for x in parameters["probability_critical"].values()])

    # total population:
    district_population = pd.read_csv(os.path.join(input_folder_path, 'f_population.csv'), index_col=0)
    district_population = district_population.values
    population = district_population.sum()  # sum over wards to obtain city population

    # set city specific parameters
    hospital_capacity = int(round(parameters["health_system_capacity"] * population))

    # population by age group (N_age(i) is the population of age group i)
    ward_age_distribution = pd.read_csv(os.path.join(input_folder_path, 'f_age_distribution.csv'),
                                        index_col=0)  # the datafile contains ward level fractions in each age group
    age_groups = ward_age_distribution * district_population  # convert to number of people in age group per ward
    age_groups = age_groups.sum()  # sum over wards
    age_groups = age_groups.values  # store city level population sizes of each age group

    # load contact matrices
    household_contacts = pd.read_csv(os.path.join(input_folder_path, 'f_household_contacts.csv'), index_col=0)
    other_contacts = pd.read_csv(os.path.join(input_folder_path, 'f_nonhousehold_contacts.csv'), index_col=0)
    contact_matrix = household_contacts + other_contacts
    contact_matrix = contact_matrix.values

    # replicate last row and column to change the 8 category contact matrix to a 9 category matrix
    contact_matrix = np.vstack((contact_matrix, contact_matrix[7, 0:8]))
    last_column = contact_matrix[0:9, 7]
    last_column.shape = (9, 1)
    contact_matrix = np.hstack((contact_matrix, last_column))

    # Apply reciprocity correction (see Towers and Feng (2012))
    # C_corrected(j,k) = (C(j,k)*N(j) + C(k,j)*N(k))/(2*N(j))
    for j in range(contact_matrix.shape[0]):
        for k in range(contact_matrix.shape[0]):
            contact_matrix[j, k] = (contact_matrix[j, k] * age_groups[j] + contact_matrix[
                k, j] * age_groups[k]) / (2 * age_groups[j])

    # Scale contact matrix by population size
    # - each column is normalized by the population of that age group: X(i,j)=C(i,j)/N_age(j)
    age_row_vector = np.array(age_groups)
    age_row_vector.shape = (1, 9)
    contact_probability_matrix = np.divide(contact_matrix,
                                           age_row_vector)  # X(i,j)=C(i,j)/N_age(j) - entries now measure fraction of each age group contacted on average per day

    # Compute infection_rate from R0, exit_rate_asymptomatic, e_S and dominant eigenvalue of matrix X(i,j)*N_age(i)
    age_column_vector = np.array(age_groups)
    age_column_vector.shape = (9, 1)
    eigen_values, eigen_vectors = np.linalg.eig(np.multiply(contact_probability_matrix, age_column_vector))
    dom_eig_val = max(eigen_values)

    infection_rate = (((1 - probability_symptomatic
                        ) * exit_rate_asymptomatic + probability_symptomatic * exit_rate_symptomatic) * r_zero
                      ) / dom_eig_val
    click.echo('infection rate (beta) is {}'.format(round(infection_rate, 4)))

    # Set initial conditions
    # spread initial infections (exposed individuals) across age groups equally
    initial_exposed = (initial_infected / 9) * np.ones(9)
    # compute remaining initial populations in susceptible compartments
    initial_susceptible = age_groups - initial_exposed
    # initiallise other compartments at zero
    initial_asymptomatic = np.zeros(9)
    initial_symptomatic = np.zeros(9)
    initial_critical = np.zeros(9)
    initial_recovered = np.zeros(9)
    initial_dead = np.zeros(9)

    # Solve model over time from initial conditions, using ODE solver from scipy:
    time_points = np.linspace(1, simulation_time, simulation_time)  # Grid of time points (in days)
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
    click.echo('total infected = {} ({} percent of population)'.format(round(infected[simulation_time - 1]),
                                                                       round(infected[simulation_time - 1] * 100 / population, 2)))
    click.echo('total deceased = {}, ({} percent of infected)'.format(round(dead[simulation_time - 1]),
                                                                      round(dead[simulation_time - 1] * 100 / infected[simulation_time - 1], 2)))

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
