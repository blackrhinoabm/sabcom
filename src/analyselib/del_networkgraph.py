import os


def delete_networkgraph(data_path):
	for filename in os.listdir(data_path):
		if filename.endswith(".graphml"): 
		         # print(os.path.join(directory, filename))
		    
		    try:
		    	os.remove(data_path+filename)
		    except:
		    	print(filename,'did not work')

	print("Files removed!")

seeds = list(range(500))
#Be careful!
for s in seeds:
	path ='/Users/admin/git_repos/sabcom/experiments/montecarlo/measurement/'+str(s)+'/'
	# delete_networkgraph(path)