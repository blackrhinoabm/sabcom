# To use this script comment out the @click decorators in the __main__ and place input & output folders in root

from sabcom.__main__ import initialise, simulate#, de_model

CITIES = ['cape_town', 'johannesburg']
CITY = CITIES[0]
seed = 31
# debug initialise

#initialise(input_folder_path='{}'.format(CITY), seed=seed)
#
# # # debug simulation
simulate(seed=seed, input_folder_path='{}'.format(CITY), output_folder_path='output_data/{}'.format(CITY),
         data_output_mode='csv-light', scenario='ineffective-lockdown',
         probability_transmission=0.4, likelihood_awareness=0.99,
         days=200,
         visiting_recurring_contacts_multiplier=0.9, gathering_max_contacts=32,
         #sensitivity_config_file_path='config_ct.json'
         )
#

#de_model(input_folder_path='cape_town', r_zero=2, output_folder_path='output_data')