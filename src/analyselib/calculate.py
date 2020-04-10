

def add_statuses_columns(data):
	for elem in data.status.unique():
		data[str(elem)] = data.status == elem
		data[str(elem)] = data[str(elem)] *1
	return data

def disease_progression(df,time):
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt 

	df.t = pd.to_numeric(df.t, errors='coerce')
	df=df[['t', 's', 'i1', 'i2', 'r','d', 'c','status']]
	# print(df.columns)
	data=df.groupby('t').sum()
	return data



 
