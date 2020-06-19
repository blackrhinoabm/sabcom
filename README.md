[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python application](https://github.com/blackrhinoabm/sabcom/workflows/Python%20application/badge.svg)

<img src="https://pbs.twimg.com/profile_images/1270246832015314953/CW4YcWdd_400x400.jpg" width="125">

![](https://cogeorg.github.io/images/black_rhino_logo.jpg)
![](https://github.com/joerischasfoort/joerischasfoort.github.io/blob/master/images/covi-id.png)


[comment]: <> (One paragraph overview of the project, TODO add link to blog?)
 __The Spatial Agent-Based Covid-19 Model (SABCOM)__

SABCOM is an open source, easy-to-use-and-adapt, spatial network, multi-agent, simulation model of the spread of the Covid-19 virus. The model is designed to simulate and analyse the spread of Covid-19 at the level of individual people. This distinguishes the model from most epidemiology models which operate top down and at a higher level of abstraction. This makes our model particularly useful for researchers and policy makers needing to study the impact of Covid-19 in highly heterogenous and unequal populations, e.g. neighbourhoods with vulnerable populations alongside wealthy populations. The bottom up granularity further allows for the analysis of targeted measures such as quarantining certain neighbourhoods, transmission reduction measures (such as wearing masks), and targeted social distancing measures aimed at specific segments of the population, .

The model is inspired by the canonical SEIR structure and generates curves that reflect the number of agents that are susceptible (s) infected without symptoms (i1), with symptoms (i2), critically ill (c), and are recovered (r). 


[comment]: <> (The output of a simulation run might look something like this:  ) 

[comment]: <> (<img src="https://github.com/joerischasfoort/joerischasfoort.github.io/blob/master/images/the_curve.png" height="512px"/> ) 

Due to the unique spatial structure of the model, we can track how a virus spreads spatially. For example through Cape Town, our preliminary case study city. The figure below shows the proportion of the population infected in different wards in the City of Cape town. Note, this is a hypothetical simulation of a non-calibrated model and is only used to give an idea of possible dynamics. 

<img src="https://github.com/joerischasfoort/joerischasfoort.github.io/blob/master/images/Infected.gif" height="768px"/>

 __Getting started__

You can **install** the SABCom model by cloning this repository to your system. After that, you are ready to start using the model.

__Running the model__

There are two files in the main folder that will allow you to run the model. In the first step, you set the parameters for the simulation and store them in a json file. Open SABCoModel_notebook.ipynb and set

```python
parameters = {
    # general simulation parameters
    "time": 150,
    "number_of_agents": 10000,
    ...
}
```
and save as json file with

```
with open('parameters.json', 'w') as outfile:
    json.dump(parameters, outfile)
```


Next, you also need to set the location and number of neigborhoods. We use population density data from 116 Wards in and around Cape Town.

```
population = pd.read_csv('population.csv')
smallest_size = population['Population'].sum() / parameters['number_of_agents']
neighbourhood_data = []
for i in range(len(population)):
    if population['Population'].iloc[i] > smallest_size:
        neighbourhood_data.append(
            [int(population['WardID'].iloc[i]), {'Population': population['Population'].iloc[i],
                                                            'Density': population['Density'].iloc[i],
                                                            'lon': population['lon'].iloc[i],
                                                            'lat': population['lat'].iloc[i]}])
max_neighbourhoods = len(neighbourhood_data)
with open('neighbourhood_data.json', 'w') as outfile:
    json.dump(neighbourhood_data[:max_neighbourhoods], outfile)
```

The configuration files __parameters.json__ and __neighbourhood_data.json__ can then be fed into the main program.
There are two options, with the second more straightforward :

1) you run the __Initialization__, __Simulation__ and  __Save network data__ cells from within the notebook
2) you run  ```python SABCoModel.py```


__Analysis__
The notebook contains a section that produces the output graphs


__Requirements__
The program requires python 3, the geopy, and the networkx non-standard packages.

__Contact and Social Media__
https://twitter.com/SABCOM5

__Disclaimer__

This software is intended for educational and research purposes. Despite best efforts,
we cannot fully rule out the possibility of errors and bugs. The use of SABCoM
is entirely at your own risk.
