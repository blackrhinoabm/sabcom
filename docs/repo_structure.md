# SABCoM repository lay-out	
	
    .
    ├── /docs			    	# documentation files, also used to host documentation site
    ├──	/.github/workflows		# used for continuous integration  
	├── /tests              	# folder that contains tests
        └── test_SABCoM.py		# test script to make sure the application runs
	├── /data					# contains example input data
	    ├── /initialisations    # pickle files of initialised model version with 100k agents
		├── /input_data			# data files that serve as input for the model along with the scripts used to generate them
		└── parameters.json		# example parameters file
    ├── requirements.txt    	# text file containing python package dependencies
    ├── LICENSE			    	# MIT licence information	
    ├── README.md				# model description
    ├── .gitignore				# list of files to ignore for Github 
	├── setup.py				# used to install the package
	└── /sabcom
        ├── initialisation.py	# script used to initialise the pickle files in the initialisation folder
        ├── __main__.py			# main script that contains the functions to simulate and initialise the model
		├── environment.py		# defines the ennvironment object used in the simulations
		├── runner.py			# defines the simulation function
		├── initialisation.py   # defines the initialisation function
		├── helpers.py          # defines all helper functions
        └── agent.py            # defines the agent object
	
