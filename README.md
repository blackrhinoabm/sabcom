[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python application](https://github.com/blackrhinoabm/sabcom/workflows/Python%20application/badge.svg)

<img src="https://pbs.twimg.com/profile_images/1270246832015314953/CW4YcWdd_400x400.jpg" width="125">

![](https://cogeorg.github.io/images/black_rhino_logo.jpg)
![](https://github.com/joerischasfoort/joerischasfoort.github.io/blob/master/images/covi-id.png)


[comment]: <> (One paragraph overview of the project, TODO add link to blog?)
 __The Spatial Agent-Based Covid-19 Model (SABCOM)__

SABCOM is an open source, easy-to-use-and-adapt, spatial network, multi-agent, model that can be used to simulate the effects of different lockdown policy measures on the spread of the Covid-19 virus in several South African cities. The model is designed to simulate and analyse the spread of Covid-19 at the level of individual people. This distinguishes the model from most epidemiology models which operate top down and at a higher level of abstraction. This makes our model particularly useful for researchers and policy makers needing to study the impact of Covid-19 in highly heterogenous and unequal populations, e.g. neighbourhoods with vulnerable populations alongside wealthy populations. The bottom up granularity further allows for the analysis of targeted measures such as quarantining certain neighbourhoods, transmission reduction measures (such as wearing masks), and targeted social distancing measures aimed at specific segments of the population, .

The model is inspired by the canonical SEIR structure and generates curves that reflect the number of agents that are susceptible (s) infected without symptoms (i1), with symptoms (i2), critically ill (c), and are recovered (r). 


[comment]: <> (The output of a simulation run might look something like this:  ) 

[comment]: <> (<img src="https://github.com/joerischasfoort/joerischasfoort.github.io/blob/master/images/the_curve.png" height="512px"/> ) 

Due to the unique spatial structure of the model, we can track how a virus spreads spatially. For example through Cape Town, our preliminary case study city. The figure below shows the proportion of the population infected in different wards in the City of Cape town. Note, this is a hypothetical simulation of a non-calibrated model and is only used to give an idea of possible dynamics. 

<img src="https://github.com/joerischasfoort/joerischasfoort.github.io/blob/master/images/Infected.gif" height="768px"/>

 __Getting started__

You can **install** the SABCom model by cloning this repository to your system. After that, you are ready to start using the model.

__Running the model__

Just run the file: ```SABCoModel.py```.

__Analysis__
Running the model in the notebook provides the added advantage that it comes with code to generate graphs to understand the model dynamics. 

__Requirements__
The program requires Python 3, and the packages listed in the requirements.txt file.

__Contact and Social Media__
https://twitter.com/SABCOM5

__Disclaimer__

This software is intended for educational and research purposes. Despite best efforts,
we cannot fully rule out the possibility of errors and bugs. The use of SABCoM
is entirely at your own risk.
