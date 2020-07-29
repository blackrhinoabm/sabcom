import sys
import click
import pickle
import os
import time
import json
import pandas as pd
import networkx as nx

from runner import runner
from environment import Environment
from helpers import generate_district_data, what_informality


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
@click.argument('seed', required=True)
@click.argument('output_folder_path', type=click.Path(exists=True), required=True)
# @click.argument('initialisation_folder_path', required=False)
# @click.argument('parameters_path', required=False)
# @click.argument('input_folder_path', required=False)
# @click.argument('data_output_mode', required=False)
# @click.argument('scenario', required=False)
def simulate(**kwargs): #seed, output_folder_path, initialisation_folder_path, parameters_path, input_folder_path, data_output_mode, scenario
    """Simulate the model"""
    start = time.time()
    default_data_path = os.path.join(os.path.dirname(sys.path[0]), 'example_data')
    seed = kwargs.get('seed')

    inititialisation_path = kwargs.get('initialisation_folder_path', os.path.join(default_data_path, 'initialisations'))

    seed_path = os.path.join(inititialisation_path, 'seed_{}.pkl'.format(seed))

    data = open(seed_path, "rb") # TODO add custom error message here informing that the specified seed is not in folder
    list_of_objects = pickle.load(data)
    environment = list_of_objects[0]

    #seed = int(re.findall(r'\d+', initialisation_path)[0]) TODO remove

    input_folder_path = kwargs.get('input_folder_path', os.path.join(default_data_path, 'input_data'))

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
    output_folder_path = kwargs.get('output_folder_path')

    input_folder_path = kwargs.get('input_folder_path', os.path.join(default_data_path, 'input_data'))

    initial_infections = pd.read_csv(os.path.join(input_folder_path, 'Cases_With_Subdistricts.csv'), index_col=0)

    # save csv light or network data
    data_output_mode = kwargs.get('data_output_mode', 'csv_light')  # default output mode is csv_light
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
    elif data_output_mode == 'csv_light':
        pd.DataFrame(environment.infection_quantities).to_csv(os.path.join(output_folder_path,
                                                                           'seed{}quantities_state_time.csv'.format(seed)))

    end = time.time()
    hours_total, rem_total = divmod(end - start, 3600)
    minutes_total, seconds_total = divmod(rem_total, 60)
    click.echo("TOTAL RUNTIME {:0>2}:{:0>2}:{:05.2f}".format(int(hours_total), int(minutes_total), seconds_total))
    click.echo('Simulation done, check out the output data here: {}'.format(output_folder_path))


@main.command()
@click.argument('seed', required=True)
@click.argument('initialisation_path', type=click.Path(exists=True), required=True)
#@click.argument('parameters_path', required=False)
#@click.argument('input_folder_path', required=False)
#@click.argument('data_output_mode', required=False)
def initialise(**kwargs):  # seed, initialisation_path, parameters_path, input_folder_path, data_output_mode
    """Initialise the model in specified directory"""
    start = time.time()

    default_data_path = os.path.join(os.path.dirname(sys.path[0]), 'example_data')

    # 1 load general the parameters
    parameters_path = kwargs.get('parameters_path', os.path.join(default_data_path, 'parameters.json'))
    print(parameters_path)

    with open(parameters_path) as json_file:
        parameters = json.load(json_file)

    # Change parameters depending on experiment
    data_output_mode = kwargs.get('data_output_mode', 'csv_light')  # default output mode is csv_light
    parameters['data_output'] = data_output_mode

    age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
                  'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

    input_folder_path = kwargs.get('input_folder_path', os.path.join(default_data_path, 'input_data'))

    # 2 load district data
    # transform input data to general district data for simulations
    district_data = generate_district_data(int(parameters['number_of_agents']), path=input_folder_path)

    # 2.2 age data
    age_distribution = pd.read_csv(os.path.join(input_folder_path, 'age_dist.csv'), sep=';', index_col=0)
    age_distribution_per_ward = dict(age_distribution.transpose())

    # 2.3 household size distribution
    HH_size_distribution = pd.read_excel(os.path.join(input_folder_path, 'HH_Size_Distribution.xlsx'), index_col=0)

    # 3 load travel matrix
    travel_matrix = pd.read_csv(os.path.join(input_folder_path, 'Travel_Probability_Matrix.csv'), index_col=0)

    # 4 load contact matrices
    # 4.1 load household contact matrix
    hh_contact_matrix = pd.read_excel(os.path.join(input_folder_path, 'ContactMatrices_10year.xlsx'),
                                      sheet_name="Home", index_col=0)
    # add a column & row for 80 plus. Rename columns to match our age categories
    hh_contact_matrix['80plus'] = hh_contact_matrix['70_80']
    row = hh_contact_matrix.xs('70_80')
    row.name = '80plus'
    hh_contact_matrix = hh_contact_matrix.append(row)
    hh_contact_matrix.columns = age_groups
    hh_contact_matrix.index = age_groups

    # 4.2 load other contact matrix
    other_contact_matrix = pd.read_excel(os.path.join(input_folder_path, 'ContactMatrices_10year.xlsx'),
                                         sheet_name="OutsideOfHome",
                                         index_col=0)
    other_contact_matrix['80plus'] = other_contact_matrix['70_80']
    row = other_contact_matrix.xs('70_80')
    row.name = '80plus'
    other_contact_matrix = other_contact_matrix.append(row)
    other_contact_matrix.columns = age_groups
    other_contact_matrix.index = age_groups

    # make new folder for seed, if it does not exist
    initialisations_folder_path = kwargs.get('initialisation_path', os.getcwd())  # TODO make this required!

    # make new folder if it does not exist TODO remove this is the above is required
    if not os.path.exists('{}'.format(initialisations_folder_path)):
        os.makedirs('{}'.format(initialisations_folder_path))

    seed = kwargs.get('seed')
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
