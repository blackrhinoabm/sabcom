[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python application](https://github.com/blackrhinoabm/sabcom/workflows/Python%20application/badge.svg)

<img src="https://pbs.twimg.com/profile_images/1270246832015314953/CW4YcWdd_400x400.jpg" width="125">

![](https://cogeorg.github.io/images/black_rhino_logo.jpg)

 __The Spatial Agent-Based Covid-19 Model (SABCOM)__

SABCOM is an open source, easy-to-use-and-adapt, spatial network, multi-agent, model that can be used to simulate the effects of different lockdown policy measures on the spread of the Covid-19 virus in several (South African) cities. 

# Installation

## Using Pip

```bash
  $ pip install sabcom
```

or, alternatively 

```bash
  $ pip3 install sabcom
```

## Manual

```bash
  $ git clone https://github.com/blackrhinoabm/sabcom
  $ cd sabcom
  $ python setup.py install
```

# Usage

The application can be used to simulate the progression of Covid-19 over a city of choice. Before running
the application, the user needs that make sure that all dependencies are installed. This can be done by 
installing the files in the requirements.txt file on Github or on your system if you did a manual installation.
Given that you are in the folder that contains this file use:

```bash
  $ python -m pip install -r requirements.txt
```

Next, there are two options. Simulating the model (using an existing initialisation) or initialising a new model environment that can be 
used for the simulation.

## Simulation
Five arguments need to be provided to simulate the model: a path for the input folder (-i), a path for the output
folder (-o), a seed (-s), a data output mode (-d), and a scenario (-sc).

`simulate -i <input folder path> -o <output folder path> -s <seed> -d <data output mode> -sc <scenario>`

For example, say you want to simulate the model using input folder `example_data`, 
output folder `example_data/output_data`, seed `2`, data output mode `csv-light`, and scenario `no-intervention`. 
First, make sure that all the files and folders are in your current location. Next, you type in the command line:  

```bash
$ sabcom simulate -i example_data -o example_data/output_data -s 2 -d csv-light -sc no-intervention
```

This will simulate a no_intervention scenario for the seed_2.pkl initialisation. input files for the city of your choice, 
and output a csv light data file in the specified output folder.

Note how this assumes that there is already an initialisation file. If this is not the case, 
sabcom can be used to produce one given the input files. 

## Initialisation
`initialise <input folder path> <seed number>`

If an initialisation file is not present, you can create one using the sabcom initialise function. 
For example, if you want to create an initialisation with the files in input folder (assumed to be in your current working directory) `example_data`, 
Monte Carlo seed 3, the following command can be used:

```bash
$ sabcom initialise -i example_data -s 3
```

As a rule, creating a model initialisation takes much longer than simulating one.

# Requirements
The program requires Python 3, and the packages listed in the requirements.txt file.

# Website and Social Media
https://sabcom.co.za

https://twitter.com/SABCOM5

# Disclaimer

This software is intended for educational and research purposes. Despite best efforts,
we cannot fully rule out the possibility of errors and bugs. The use of SABCoM
is entirely at your own risk.
