import pandas as pd
import networkx as nx

from src.analyselib.read import *
from src.analyselib.calculate import *


population = pd.read_csv('../population.csv')
datapath = '../measurement/'


time = get_time(datapath)
# infection_states = get_networklist(datapath+"network_time{}.graphml",time) # Get network
 
# get data and manipulate columns 
csvlist = get_csvlist_old(datapath, '2agent_data0.csv') #pass in first day, attention:work around
data = pd.concat(csvlist).sort_values(by=['t', 'agent'])
data = add_statuses_columns(data)


data = disease_progression_withwards(data, time)
data.to_csv('../experiments/testing_groupby.csv')











