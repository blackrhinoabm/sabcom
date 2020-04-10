

def get_time(data_path):
	import os
	count =0
	for filename in os.listdir(data_path):
	    if filename.endswith(".csv"): 
	         # print(os.path.join(directory, filename))
	        count+=1
	    else:
	        continue
	return count

def get_networklist(networkpath, time):
	import networkx as nx
	infection_states = []
	for idx in range(time):
		infection_states.append(nx.read_graphml(networkpath.format(idx), node_type=int))
	return infection_states

def get_csvlist(data_path, template):
	import os
	import pandas as pd
	import re
	import sys

	# check if template matches first day
	temp =[]
	for filename in os.listdir(data_path):
	    if filename.endswith(".csv"):
	    	temp.append(filename)
	    	
	if template !=(sorted(temp)[0]):
		print("Template",template,len(template),'letters',type(template))
		print("data name",sorted(temp)[0],len(sorted(temp)[0]),'letters',type(sorted(temp)[0]))
		print('firstdaydata incorreclty passed in')
		sys.exit()
	#read csv
	pd_list = []

	for filename in (os.listdir(data_path)):
	    if filename.endswith(".csv"): 
	         # print(os.path.join(directory, filename))
	        df = pd.read_csv(data_path+filename)

	        x = re.findall(r'\d+',filename)
	        day = x[1]
	       	df['t'] = x[1]
	        pd_list.append(df)
	    else:
	        continue
	        
	return pd_list
