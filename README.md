[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python application](https://github.com/blackrhinoabm/sabcom/workflows/Python%20application/badge.svg)

<img src="https://pbs.twimg.com/profile_images/1270246832015314953/CW4YcWdd_400x400.jpg" width="125">

![](https://cogeorg.github.io/images/black_rhino_logo.jpg)
![](https://github.com/joerischasfoort/joerischasfoort.github.io/blob/master/images/covi-id.png)


[comment]: <> (One paragraph overview of the project, TODO add link to blog?)
 __The Spatial Agent-Based Covid-19 Model (SABCOM)__

SABCOM is an open source, easy-to-use-and-adapt, spatial network, multi-agent, model that can be used to simulate the effects of different lockdown policy measures on the spread of the Covid-19 virus in several (South African) cities. 

# Installation

## Using Pip

```bash
  $ pip install sabcom
```

## Manual

```bash
  $ git clone https://github.com/blackrhinoabm/sabcom
  $ cd sabcom
  $ python setup.py install
```

# Usage

The application can be used to simulate the progression of Covid-19 over a city of choice. To run it, an initialised environment is needed. 

## Simulation
To simulate the model three 

`simulate <initialisation path> <parameters path> <input folder pah> <output folder path> <data output mode> <scenario>`

For example, say you want to simulate the model using initialisation `seed_2.pkl`, parameter file `parameters.json`, input folder `/input_folder`, output folder `/output_folder`, output mode "csv_light", and scenario "no_intervention". 
First, make sure that all the files and folders are in your current location. Next, you type in the command line:  

```bash
$ sabcom simulate seed_2.pkl parameters.json /input_folder /output_folder "csv_light" "no_intervention"
```

This will simulate a no_intervention scenario for the seed_2 initialisation with the specified parameters, input files for the city of your choice, and output a csv light data file in the specified output folder.

Note how this assumes that there is already an initialisation file. If this is not the case, sabcom can be used to produce one given the input files. 

## Initialisation

If an initialisation file is not present, you can create one using sabcom. For example, if you want to create an initialisation with Monte Carlo seed 2, parameter file `parameters.json`, and the files in input folder `/input_folder`, the following command can be used:

```bash
$ sabcom initialise 2 parameters.json /input_folder /output_folder "csv_light" "no_intervention"
```

As a rule, creating a model initialisation takes much longer than simulating one.

# Requirements
The program requires Python 3, and the packages listed in the requirements.txt file.

# Contact and Social Media
https://twitter.com/SABCOM5

# Disclaimer

This software is intended for educational and research purposes. Despite best efforts,
we cannot fully rule out the possibility of errors and bugs. The use of SABCoM
is entirely at your own risk.
