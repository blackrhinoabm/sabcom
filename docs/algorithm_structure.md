# Model algorithm structure

The model can be seen to consist of two parts. The first is used to initialise the model in src/environment.py and the 
second is used to run the simulations in runner.py.  

The initialisation in the environment script can further be divided in three parts:

1. create modelled districts,
2. create household network structure,
3. create other contacts network structure.

The second part of the script in the runner can also be divided in three parts:

4. initialisation of infections,
5. update the infection status of all agents,
6. new infections.