import pandas as pd
import json
import os

from src.environment import Environment
from src.helpers import generate_district_data
from src.runner import runner


def test_model():
    """Basic test to see if the model runs with T=90, 100 agents and 5 Monte Carlo simulations"""
    os.chdir('../sabcom')
    data_folder = 'output_data/baseline/'

    # load parameters
    with open('parameters/parameters_cape_town.json') as json_file:
        parameters = json.load(json_file)

    # set parameters to time = 90, 5 MC runs and 100 agents
    parameters["time"] = 90
    parameters["monte_carlo_runs"] = 5
    parameters["number_of_agents"] = 100

    # Change parameters depending on experiment
    age_groups = ['age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
                  'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus']

    parameters['data_output'] = 'csv_light'

    # load district data
    district_data = generate_district_data(10000)#parameters['number_of_agents']) TODO debug

    # load travel matrix
    travel_matrix = pd.read_csv('input_data/Travel_Probability_Matrix.csv', index_col=0)

    # load age data
    age_distribution = pd.read_csv('input_data/age_dist.csv', sep=';', index_col=0)
    age_distribution_per_ward = dict(age_distribution.transpose())

    # load household contact matrix
    hh_contact_matrix = pd.read_excel('input_data/ContactMatrices_10year.xlsx', sheet_name="Home", index_col=0)
    # add a col & row for 80 plus. Rename columns to mathc our age categories
    hh_contact_matrix['80plus'] = hh_contact_matrix['70_80']
    row = hh_contact_matrix.xs('70_80')
    row.name = '80plus'
    hh_contact_matrix = hh_contact_matrix.append(row)
    hh_contact_matrix.columns = age_groups
    hh_contact_matrix.index = age_groups

    # load other contact matrix
    other_contact_matrix = pd.read_excel('input_data/ContactMatrices_10year.xlsx', sheet_name="OutsideOfHome",
                                         index_col=0)
    other_contact_matrix['80plus'] = other_contact_matrix['70_80']
    row = other_contact_matrix.xs('70_80')
    row.name = '80plus'
    other_contact_matrix = other_contact_matrix.append(row)
    other_contact_matrix.columns = age_groups
    other_contact_matrix.index = age_groups

    # load household size distribution data
    HH_size_distribution = pd.read_excel('input_data/HH_Size_Distribution.xlsx', index_col=0)

    # load initial infections:
    initial_infections = pd.read_csv('input_data/Cases_With_Subdistricts.csv', index_col=0)

    # Monte Carlo simulations
    for seed in range(parameters['monte_carlo_runs']):
        # make new folder for seed, if it does not exist
        if not os.path.exists('{}seed{}'.format(data_folder, seed)):
            os.makedirs('{}seed{}'.format(data_folder, seed))

        # initialization
        environment = Environment(seed, parameters, district_data, age_distribution_per_ward,
                                  hh_contact_matrix, other_contact_matrix, HH_size_distribution, travel_matrix)

        # running the simulation
        environment = runner(environment, initial_infections, seed, data_output=parameters["data_output"],
                             data_folder=data_folder,
                             calculate_r_naught=False)
