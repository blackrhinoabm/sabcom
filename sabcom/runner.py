import click
import os
import pickle
import json
import logging
import numpy as np
import random
import pandas as pd
import scipy.stats as stats

from sabcom.updater import updater
from sabcom.helpers import generate_district_data, what_informality


def runner(**kwargs):
    """
    This function is used to update parameters

    :param kwargs:
    :return:
    """

    # store often used arguments in temporary variable
    seed = kwargs.get('seed')
    output_folder_path = kwargs.get('output_folder_path')
    input_folder_path = kwargs.get('input_folder_path')

    # formulate paths to initialisation folder and seed within input folder
    if kwargs.get('initial_seeds_folder'):
        inititialisation_path = kwargs.get('initial_seeds_folder')
        click.echo('Modified initialisations path: {}'.format(inititialisation_path))
    else:
        inititialisation_path = os.path.join(input_folder_path, 'initialisations')

    seed_path = os.path.join(inititialisation_path, 'seed_{}.pkl'.format(seed))
    if not os.path.exists(seed_path):
        click.echo(seed_path + ' not found', err=True)
        click.echo('Error: specify a valid seed')
        return

    # open the seed pickle object as an environment
    data = open(seed_path, "rb")
    list_of_objects = pickle.load(data)
    environment = list_of_objects[0]

    # add contacts section to csv light DataFrame TODO remove after re-initialisation
    environment.infection_quantities['contacts'] = []

    if kwargs.get('initial_seeds_folder'):
        environment.infection_quantities = {key: [] for key in
                                            ['e', 's', 'i1', 'i2', 'c', 'r', 'd', 'compliance', 'contacts']}

    # initialise logging
    logging.basicConfig(filename=os.path.join(output_folder_path,
                                              'simulation_seed{}.log'.format(seed)), filemode='w', level=logging.DEBUG)
    logging.info('Start of simulation seed{} with arguments -i ={}, -o={}'.format(seed,
                                                                                  input_folder_path,
                                                                                  output_folder_path))

    # update optional parameters
    if kwargs.get('sensitivity_config_file_path'):
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

    if kwargs.get('days'):
        environment.parameters['time'] = kwargs.get('days')
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
        click.echo(
            'Transmission probability has been set to {}'.format(environment.parameters['probability_transmission']))
        logging.debug(
            'Transmission probability has been set to {}'.format(environment.parameters['probability_transmission']))

    if kwargs.get('second_infection_n'):
        environment.parameters['second_infection_n'] = kwargs.get('second_infection_n')
        click.echo(
            'Second infection number has been set to {}'.format(environment.parameters['second_infection_n']))
        logging.debug(
            'Second infection number has been set to {}'.format(environment.parameters['second_infection_n']))

    if kwargs.get('learning_scenario'):
        environment.parameters['learning_scenario'] = kwargs.get('learning_scenario')
        click.echo(
            'learning_scenario has been set to {}'.format(environment.parameters['learning_scenario']))
        logging.debug(
            'learning_scenario has been set to {}'.format(environment.parameters['learning_scenario']))

    if kwargs.get('time_4_new_infections'):
        environment.parameters['time_4_new_infections'] = kwargs.get('time_4_new_infections')
        click.echo(
            'Second infection time has been set to {}'.format(environment.parameters['time_4_new_infections']))
        logging.debug(
            'Second infection time has been set to {}'.format(environment.parameters['time_4_new_infections']))

    if kwargs.get('new_infections_scenario'):
        environment.parameters['new_infections_scenario'] = kwargs.get('new_infections_scenario')
        click.echo(
            'New infections scenario has been set to {}'.format(environment.parameters['new_infections_scenario']))
        logging.debug(
            'New infections scenario has been set to {}'.format(environment.parameters['new_infections_scenario']))

    if kwargs.get('visiting_recurring_contacts_multiplier'):
        environment.parameters['visiting_recurring_contacts_multiplier'] = kwargs.get('visiting_recurring_contacts_multiplier')
        click.echo('Recurring contacts has been set to {}'.format(
            environment.parameters['visiting_recurring_contacts_multiplier']))
        logging.debug(
            'Recurring contacts has been set to {}'.format(
                environment.parameters['visiting_recurring_contacts_multiplier']))

    if kwargs.get('initial_infections'):
        environment.parameters['total_initial_infections'] = round(int(kwargs.get('initial_infections')))
        click.echo('Initial infections have been set to {}'.format(environment.parameters['total_initial_infections']))
        logging.debug(
            'Initial infections have been set to {}'.format(environment.parameters['total_initial_infections']))

    # TODO debug!!! add extra parameters here!
    if kwargs.get('vaccination_scenario'):
        scenario = kwargs.get('vaccination_scenario')
        if scenario == 'risk_based':
            environment.parameters['agent_priority_list'] = ['age_80_plus', 'age_70_80',  'age_60_70',  'age_50_60',
                                                             'age_40_50',  'age_30_40',  'age_20_30',  'age_10_20',
                                                             'age_0_10']
        elif scenario == 'connection_based':
            environment.parameters['agent_priority_list'] = ['age_10_20', 'age_20_30', 'age_30_40',
                                                             'age_40_50', 'age_0_10', 'age_50_60', 'age_60_70',
                                                             'age_70_80', 'age_80_plus']
        else:
            environment.parameters['agent_priority_list'] = random.sample(['age_80_plus', 'age_70_80', 'age_60_70', 'age_50_60',
             'age_40_50', 'age_30_40', 'age_20_30', 'age_10_20',
             'age_0_10'], 9)
        click.echo('Vaccination scenario is {}'.format(environment.parameters['agent_priority_list']))
        logging.debug(
            'Initial infections have been set to {}'.format(environment.parameters['agent_priority_list']))

    else:
        environment.parameters['agent_priority_list'] = random.sample(['age_80_plus', 'age_70_80', 'age_60_70', 'age_50_60',
             'age_40_50', 'age_30_40', 'age_20_30', 'age_10_20',
             'age_0_10'], 9)

    # TODO debug 2
    if kwargs.get('death_response_intensity'):
        environment.parameters['death_response_intensity'] = kwargs.get('death_response_intensity')
    else:
        try:
            click.echo('keeping death response intensity at {}'.format(environment.parameters['death_response_intensity']))

        except:
            environment.parameters['death_response_intensity'] = 0
            click.echo('No death_response_intensity found... initiating as 0')



    if kwargs.get('daily_vaccinations'):
        environment.parameters['daily_vaccinations'] = kwargs.get('daily_vaccinations')
        click.echo('Vaccination scenario is {}'.format(environment.parameters['daily_vaccinations']))
        logging.debug(
            'Initial infections have been set to {}'.format(environment.parameters['daily_vaccinations']))
    else:
        environment.parameters['daily_vaccinations'] = 0


    # check if the stringency index has changed in the parameter file
    sringency_index_updated = False
    if environment.stringency_index != environment.parameters['stringency_index'] or len(environment.stringency_index) < environment.parameters['time']:
        # initialise stochastic process in case stringency index has changed
        click.echo('change in stringency index detected and updated for all agents')
        environment.stringency_index = environment.parameters['stringency_index']
        if len(environment.stringency_index) < environment.parameters['time']:
            environment.stringency_index += [environment.stringency_index[-1] for x in range(len(
                environment.stringency_index), environment.parameters['time'])]

        lower, upper = -(environment.stringency_index[0] / 100), \
                       (1 - (environment.parameters['stringency_index'][0] / 100))
        mu, sigma = 0.0, environment.parameters['private_shock_stdev']
        shocks = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma,
                                     size=len(environment.agents))
        sringency_index_updated = True

    # transform input data to general district data for simulations
    district_data = generate_district_data(environment.parameters['number_of_agents'], path=input_folder_path)

    # set scenario specific parameters
    scenario = kwargs.get('scenario', 'no-intervention')  # if no input was provided use no-intervention
    click.echo('scenario is {}'.format(scenario))
    if scenario == 'no-intervention':
        environment.parameters['stringency_index'] = [0.0 for x in environment.parameters['stringency_index']]
        environment.stringency_index = environment.parameters['stringency_index']
        environment.parameters['informality_dummy'] = 0.0
    elif scenario == 'lockdown':
        environment.parameters['informality_dummy'] = 0.0
    elif scenario == 'ineffective-lockdown':
        environment.parameters['informality_dummy'] = 1.0

    # log parameters used after scenario called
    for param in environment.parameters:
        logging.debug('Parameter {} has the value {}'.format(param, environment.parameters[param]))

    # update agent informality based on scenario
    for i, agent in enumerate(environment.agents):
        agent.informality = what_informality(agent.district, district_data
                                             ) * environment.parameters["informality_dummy"]
        # optionally also update agent initial compliance if stringency was changed.
        if sringency_index_updated:
            agent.compliance = environment.stringency_index[0] / 100 + shocks[i]
            agent.previous_compliance = agent.compliance

        if agent.status in ['e', 'i1', 'i2', 'c']:
            agent.status = 'r'

        # next option update susceptibility ...  TODO debug!
        if kwargs.get('newly_susceptible_percentage'):
            if agent.status == 'r':
                if np.random.random() < kwargs.get('newly_susceptible_percentage'):
                   agent.status = 's'
                   logging.debug(
                       'Agent {} status has been changed from r to s'.format(agent))

    initial_infections = pd.read_csv(os.path.join(input_folder_path, 'f_initial_cases.csv'), index_col=0)
    environment.parameters["data_output"] = kwargs.get('data_output_mode',
                                                       'csv-light')  # default output mode is csv_light

    # Simulate the model
    environment = updater(environment=environment, initial_infections=initial_infections, seed=int(seed),
                          data_folder=output_folder_path,
                          data_output=environment.parameters["data_output"])

    return environment
