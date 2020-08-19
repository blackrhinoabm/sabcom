import sys
import click
import pickle
import os
import time
import json
import pandas as pd
import networkx as nx

from sabcom.runner import runner
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


@main.command()
@click.option('--input_folder_path', '-i', type=click.Path(exists=True), required=True,
              help="This should contain all nescessary input files, specifically an initialisation folder")
@click.option('--output_folder_path', '-o', type=click.Path(exists=True), required=True,
              help="All simulation output will be deposited here")
@click.option('--seed', '-s', type=int, required=True,
              help="Integer seed number that is used for Monte Carlo simulations")
@click.option('--data_output_mode', '-d', default='csv-light', show_default=True,
              type=click.Choice(['csv-light', 'csv', 'network'],  case_sensitive=False,))
@click.option('--scenario', '-sc', default='no-intervention', show_default=True,
              type=click.Choice(['no-intervention', 'lockdown', 'ineffective-lockdown'],  case_sensitive=False,))
@click.option('--days', '-day', default=None, type=int, required=False,
              help="change the number of simulation days here with the caveat that simulation time can only be shortened compared to what was initialised.")
@click.option('--probability_transmission', '-pt', default=None, type=float, required=False,
              help="change the probability of transmission between two agents.")
@click.option('--visiting_recurring_contacts_multiplier', '-cont', default=None, type=float, required=False,
              help="change the percentage of contacts agent may have.")
@click.option('--likelihood_awareness', '-la', default=None, type=float, required=False,
              help="change the likelihood that an agent is aware it is infected.")
@click.option('--gathering_max_contacts', '-maxc', default=None, type=int, required=False,
              help="change maximum number of contacts and agent is allowed to have.")
def simulate(**kwargs):
    """Simulate the model"""
    start = time.time()

    # format arguments
    seed = kwargs.get('seed')
    output_folder_path = kwargs.get('output_folder_path')
    input_folder_path = kwargs.get('input_folder_path')

    inititialisation_path = os.path.join(input_folder_path, 'initialisations')

    seed_path = os.path.join(inititialisation_path, 'seed_{}.pkl'.format(seed))

    if not os.path.exists(seed_path):
        click.echo(seed_path + ' not found', err=True)
        click.echo('Specify a valid seed')
        return

    data = open(seed_path, "rb")
    list_of_objects = pickle.load(data)
    environment = list_of_objects[0]

    # update optional parameters
    if kwargs.get('days'):
        max_time = environment.parameters['time']  # you cannot simulate longer than initialised
        environment.parameters['time'] = min(kwargs.get('days'), max_time)
        click.echo('Time has been set to {}'.format(environment.parameters['time']))

    if kwargs.get('probability_transmission'):
        environment.parameters['probability_transmission'] = kwargs.get('probability_transmission')
        click.echo('Transmission probability has been set to {}'.format(environment.parameters['probability_transmission']))

    if kwargs.get('visiting_recurring_contacts_multiplier'):
        environment.parameters['visiting_recurring_contacts_multiplier'] = [kwargs.get('visiting_recurring_contacts_multiplier') for x in environment.parameters['visiting_recurring_contacts_multiplier']]
        click.echo('Recurring contacts has been set to {}'.format(environment.parameters['visiting_recurring_contacts_multiplier'][0]))

    if kwargs.get('likelihood_awareness'):
        environment.parameters['likelihood_awareness'] = [kwargs.get('likelihood_awareness') for x in environment.parameters['visiting_recurring_contacts_multiplier']]
        click.echo('Likelihood awareness has been set to {}'.format(environment.parameters['likelihood_awareness'][0]))

    if kwargs.get('gathering_max_contacts'):
        environment.parameters['gathering_max_contacts'] = [kwargs.get('gathering_max_contacts') for x in environment.parameters['visiting_recurring_contacts_multiplier']]
        click.echo('Max contacts has been set to {}'.format(environment.parameters['gathering_max_contacts'][0]))

    # transform input data to general district data for simulations
    district_data = generate_district_data(environment.parameters['number_of_agents'], path=input_folder_path)

    # set scenario specific parameters
    scenario = kwargs.get('scenario', 'no-intervention')  # if no scenario was provided no_intervention is used
    print('scenario is ', scenario)
    if scenario == 'no-intervention':
        environment.parameters['likelihood_awareness'] = [
            0.0 for x in environment.parameters['likelihood_awareness']]
        environment.parameters['visiting_recurring_contacts_multiplier'] = [
            1.0 for x in environment.parameters['visiting_recurring_contacts_multiplier']]
        environment.parameters['gathering_max_contacts'] = [
            float('inf') for x in environment.parameters['gathering_max_contacts']]
        environment.parameters['physical_distancing_multiplier'] = [
            1.0 for x in environment.parameters['physical_distancing_multiplier']]
        environment.parameters['informality_dummy'] = 0.0
    elif scenario == 'lockdown':
        environment.parameters['informality_dummy'] = 0.0
    elif scenario == 'ineffective-lockdown':
        environment.parameters['informality_dummy'] = 1.0

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


@main.command()
@click.option('--input_folder_path', '-i', type=click.Path(exists=True), required=True,
              help="Folder containing parameters file, input data and an empty initialisations folder")
#@click.option('--output_folder_path', '-o', type=click.Path(exists=True), required=True)
@click.option('--seed', '-s', type=int, required=True, help="Integer seed number that is used for Monte Carlo simulations")
def initialise(**kwargs):  # input output seed
    """Initialise the model in specified directory"""
    start = time.time()

    seed = kwargs.get('seed')
    #output_folder_path = kwargs.get('output_folder_path') TODO remove
    input_folder_path = kwargs.get('input_folder_path')
    #default_data_path = os.path.join(os.path.dirname(sys.path[0]), 'example_data')

    # format optional arguments
    parameters_path = os.path.join(input_folder_path, 'parameters.json')
    initialisations_folder_path = os.path.join(input_folder_path, 'initialisations')

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

    # Change parameters depending on experiment
    data_output_mode = kwargs.get('data_output_mode', 'csv-light')  # TODO is this still nescessary?
    #print('data output mode = ', data_output_mode)

    parameters['data_output'] = data_output_mode

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

    # make new folder if it does not exist TODO remove this if the above is required
    if not os.path.exists('{}'.format(initialisations_folder_path)):
        os.makedirs('{}'.format(initialisations_folder_path))

    # initialisation
    environment = Environment(int(seed), parameters, district_data, age_distribution_per_ward,
                              hh_contact_matrix, other_contact_matrix, HH_size_distribution, travel_matrix)

    # save environment objects as pickls
    #file_name = initialisations_folder_path + "/seed_" + str(seed) + '.pkl'
    file_name = os.path.join(initialisations_folder_path, "seed_{}.pkl".format(str(seed))) #TODO debug
    save_objects = open(file_name, 'wb')
    pickle.dump([environment, seed], save_objects)
    save_objects.close()

    end = time.time()
    hours_total, rem_total = divmod(end - start, 3600)
    minutes_total, seconds_total = divmod(rem_total, 60)
    click.echo("TOTAL RUNTIME {:0>2}:{:0>2}:{:05.2f}".format(int(hours_total), int(minutes_total), seconds_total))
    click.echo('Initialisation done, check out the output data here: {}'.format(initialisations_folder_path))


#initialise(seed=3, initialisation_path='../example_data/initialisations')
#simulate(seed=3, output_folder_path='../example_data/output_data')

if __name__ == '__main__':
    args = sys.argv
    if "--help" in args or len(args) == 1:
        print("SABCoM")
    main()
