import click
import os
import pickle
import json
import logging
import pandas as pd
import scipy.stats as stats

from sabcom.runner import runner
from sabcom.helpers import generate_district_data, what_informality


def updater(**kwargs):
    """

    :param kwargs:
    :return:
    """

    # store often used arguments in temporary variable
    seed = kwargs.get('seed')
    output_folder_path = kwargs.get('output_folder_path')
    input_folder_path = kwargs.get('input_folder_path')

    # formulate paths to initialisation folder and seed within input folder
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
        environment.parameters['visiting_recurring_contacts_multiplier'] = [
            kwargs.get('visiting_recurring_contacts_multiplier') for x in range(environment.parameters['time'])]
        click.echo('Recurring contacts has been set to {}'.format(
            environment.parameters['visiting_recurring_contacts_multiplier'][0]))
        logging.debug(
            'Recurring contacts has been set to {}'.format(
                environment.parameters['visiting_recurring_contacts_multiplier'][0]))

    if type(environment.parameters['visiting_recurring_contacts_multiplier']) == list:
        if len(environment.parameters['visiting_recurring_contacts_multiplier']) < environment.parameters['time']:
            environment.parameters['visiting_recurring_contacts_multiplier'] += [
                environment.parameters['visiting_recurring_contacts_multiplier'][-1] for x in range(
                    len(environment.parameters['visiting_recurring_contacts_multiplier']), environment.parameters['time'])]
            logging.debug('visiting_recurring_contacts_multiplier has been lengthened by {}'.format(
                environment.parameters['time'] - len(environment.parameters['visiting_recurring_contacts_multiplier'])))

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
        environment.parameters['total_initial_infections'] = round(int(kwargs.get('initial_infections')))
        click.echo('Initial infections have been set to {}'.format(environment.parameters['total_initial_infections']))
        logging.debug(
            'Initial infections have been set to {}'.format(environment.parameters['total_initial_infections']))

    # check if the stringency index has changed in the parameter file
    sringency_index_updated = False
    if environment.stringency_index != environment.parameters['stringency_index']:
        # initialise stochastic process in case stringency index has changed
        click.echo('change in stringency index detected and updated for all agents')
        environment.stringency_index = environment.parameters['stringency_index']
        if len(environment.parameters['stringency_index']) < environment.parameters['time']:
            environment.stringency_index += [environment.parameters['stringency_index'][-1] for x in range(len(
                environment.parameters['stringency_index']), environment.parameters['time'])]

        lower, upper = -(environment.parameters['stringency_index'][0] / 100), \
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

    # update agent informality based on scenario
    for i, agent in enumerate(environment.agents):
        agent.informality = what_informality(agent.district, district_data
                                             ) * environment.parameters["informality_dummy"]
        # optionally also update agent initial compliance if stringency was changed.
        if sringency_index_updated:
            agent.compliance = environment.parameters['stringency_index'][0] / 100 + shocks[i]
            agent.previous_compliance = agent.compliance

    initial_infections = pd.read_csv(os.path.join(input_folder_path, 'f_initial_cases.csv'), index_col=0)
    environment.parameters["data_output"] = kwargs.get('data_output_mode',
                                                       'csv-light')  # default output mode is csv_light

    # Simulate the model
    environment = runner(environment=environment, initial_infections=initial_infections, seed=int(seed),
                         data_folder=output_folder_path,
                         data_output=environment.parameters["data_output"])

    return environment
