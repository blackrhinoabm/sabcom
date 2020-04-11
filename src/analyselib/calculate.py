
def add_statuses_columns(data):
	for elem in data.status.unique():
		data[str(elem)] = data.status == elem
		data[str(elem)] = data[str(elem)] *1
	return data

def disease_progression(df,time):
	import pandas as pd

	df.t = pd.to_numeric(df.t, errors='coerce')
	df=df[['t', 's', 'i1', 'i2', 'r','d', 'c','status']]
	data=df.groupby('t').sum()
	return data

def disease_progression_withwards(df,time):
	# to do!
	import pandas as pd
	df.t = pd.to_numeric(df.t, errors='coerce')
	df = df[['t',	'WardID',	'lon',	'lat','s',	'i1',	'i2',	'r',	'd',	'c']]
	data=df.groupby(['t', 'WardID','lon','lat']).sum()
	return data



 
