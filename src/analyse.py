
import networkx as nx
from analyselib.read  import *
import pandas as pd
import numpy as np

datapath='../measurement/'
time = get_time(datapath)  

# infection_states = get_networklist(datapath+"network_time{}.graphml",time) # Get network

# get data, pass in first day, datapath
csvlist = get_csvlist(datapath,'2agent_data0.csv') #this is a work around
data=pd.concat(csvlist).sort_values(by=['t','agent'])









