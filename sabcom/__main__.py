import sys
import click
import pickle
import re
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
@click.argument('output_folder_path', required=False)
@click.argument('initialisation_path', required=False)
@click.argument('parameters_path', required=False)
@click.argument('input_folder_path', required=False)
@click.argument('data_output_mode', required=False)
@click.argument('scenario', required=False)
def simulate(**kwargs): #output_folder_path, initialisation_path, parameters_path, input_folder_path, data_output_mode, scenario
    """Simulate the model"""
    initialisation_path = kwargs.get('initialisation_path', 'example_data/initialisations/seed_20.pkl') #  TODO add default

    data = open(initialisation_path, "rb")
    list_of_objects = pickle.load(data)
    environment = list_of_objects[0]

    seed = int(re.findall(r'\d+', initialisation_path)[0])

    input_folder_path = kwargs.get('input_folder_path', 'example_data/input_data/')

    # update time and output format in the environment TODO remove?
    #max_time = environment.parameters['time']  # you cannot simulate longer than initialised
    #environment.parameters['time'] = min(SIMULATION_TIME, max_time)

    # transform input data to general district data for simulations
    district_data = generate_district_data(environment.parameters['number_of_agents'], path=input_folder_path)

    # set scenario specific parameters
    scenario = kwargs.get('scenario', 'no_intervention')  # if no scenario was provided no_intervention is used
    if scenario == 'no_intervention':
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
    elif scenario == 'ineffective_lockdown':
        environment.parameters['informality_dummy'] = 1.0

    for agent in environment.agents:
        agent.informality = what_informality(agent.district, district_data
                                             ) * environment.parameters["informality_dummy"]

    # make new folder for seed, if it does not exist
    output_folder_path = kwargs.get('output_folder_path', 'example_data/output_data/')  # if no output folder was provided cwd is used

    if not os.path.exists('{}seed{}'.format(output_folder_path, seed)):
        os.makedirs('{}seed{}'.format(output_folder_path, seed))

    input_folder_path = kwargs.get('input_folder_path', 'example_data/input_data')

    initial_infections = pd.read_csv('{}/Cases_With_Subdistricts.csv'.format(input_folder_path), index_col=0)
    environment = runner(environment, initial_infections, seed, data_folder=input_folder_path,
                         data_output=output_folder_path)

    # save csv light or network data
    data_output_mode = kwargs.get('data_output_mode', 'csv_light')  # default output mode is csv_light

    if data_output_mode == 'network':
        for idx, network in enumerate(environment.infection_states):
            for i, node in enumerate(network.nodes):
                network.nodes[i]['agent'] = network.nodes[i]['agent'].status
            idx_string = '{0:04}'.format(idx)
            nx.write_graphml(network, "{}seed{}/network_time{}.graphml".format(output_folder_path, seed, idx_string))
    elif data_output_mode == 'csv_light':
        pd.DataFrame(environment.infection_quantities).to_csv('{}seed{}/quantities_state_time.csv'.format(
            output_folder_path, seed))

    click.echo('Simulation done, check out the output data here: {}'.format(output_folder_path))

#
# @main.command()
# @click.argument('seed', required=True)
# @click.argument('initialisation_path', required=True)
# @click.argument('parameters_path', required=False)
# @click.argument('input_folder_path', required=False)
# @click.argument('data_output_mode', required=False)
def initialise(**kwargs):  # seed, initialisation_path, parameters_path, input_folder_path, data_output_mode
    """Initialise the model in specified directory"""
    start = time.time()
    # 1 load general the parameters
    parameters_path = kwargs.get('parameters_path', 'example_data/parameters.json')

    with open(parameters_path) as json_file:
        parameters = json.load(json_file)

    # Change parameters depending on experiment
    data_output_mode = kwargs.get('data_output_mode', 'csv_light')  # default output mode is csv_light
    parameters['data_output'] = data_output_mode

    age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
                  'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

    input_folder_path = kwargs.get('input_folder_path', 'example_data/input_data/')

    # 2 load district data
    # transform input data to general district data for simulations
    district_data = generate_district_data(int(parameters['number_of_agents']), path=input_folder_path)

    # 2.2 age data
    age_distribution = pd.read_csv('{}age_dist.csv'.format(input_folder_path), sep=';', index_col=0)
    age_distribution_per_ward = dict(age_distribution.transpose())

    # 2.3 household size distribution
    HH_size_distribution = pd.read_excel('{}HH_Size_Distribution.xlsx'.format(input_folder_path), index_col=0)

    # 3 load travel matrix
    travel_matrix = pd.read_csv('{}Travel_Probability_Matrix.csv'.format(input_folder_path), index_col=0)

    # 4 load contact matrices
    # 4.1 load household contact matrix
    hh_contact_matrix = pd.read_excel('{}ContactMatrices_10year.xlsx'.format(input_folder_path), sheet_name="Home", index_col=0)
    # add a col & row for 80 plus. Rename columns to mathc our age categories
    hh_contact_matrix['80plus'] = hh_contact_matrix['70_80']
    row = hh_contact_matrix.xs('70_80')
    row.name = '80plus'
    hh_contact_matrix = hh_contact_matrix.append(row)
    hh_contact_matrix.columns = age_groups
    hh_contact_matrix.index = age_groups

    # 4.2 load other contact matrix
    other_contact_matrix = pd.read_excel('{}ContactMatrices_10year.xlsx'.format(input_folder_path), sheet_name="OutsideOfHome",
                                         index_col=0)
    other_contact_matrix['80plus'] = other_contact_matrix['70_80']
    row = other_contact_matrix.xs('70_80')
    row.name = '80plus'
    other_contact_matrix = other_contact_matrix.append(row)
    other_contact_matrix.columns = age_groups
    other_contact_matrix.index = age_groups

    # make new folder for seed, if it does not exist
    initialisations_folder_path = kwargs.get('initialisation_path', os.getcwd())  # TODO make this required?

    # make new folder if it does not exist
    if not os.path.exists('{}'.format(initialisations_folder_path)):
        os.makedirs('{}'.format(initialisations_folder_path))

    seed = kwargs.get('seed')
    # initialisation
    environment = Environment(seed, parameters, district_data, age_distribution_per_ward,
                              hh_contact_matrix, other_contact_matrix, HH_size_distribution, travel_matrix)

    # save environment objects as pickls
    file_name = initialisations_folder_path + "/seed_" + str(seed) + '.pkl'
    save_objects = open(file_name, 'wb')
    pickle.dump([environment, seed], save_objects)
    save_objects.close()

    end = time.time()
    hours_total, rem_total = divmod(end - start, 3600)
    minutes_total, seconds_total = divmod(rem_total, 60)
    print("TOTAL RUNTIME", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours_total), int(minutes_total), seconds_total))


initialise(seed=3, initialisation_path='example_data/initialisations')

if __name__ == '__main__':
    args = sys.argv
    if "--help" in args or len(args) == 1:
        print("SABCoM")
    main()
