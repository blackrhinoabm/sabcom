[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python application](https://github.com/blackrhinoabm/sabcom/workflows/Python%20application/badge.svg)
![](https://cogeorg.github.io/images/black_rhino_logo.jpg)
![](https://github.com/joerischasfoort/joerischasfoort.github.io/blob/master/images/covi-id.png)


[comment]: <> (One paragraph overview of the project, TODO add link to blog?)
 __The Spatial Agent-Based Covid Model (SABCom)__

SABCoM is an open source easy-to-use-and-adapt spatial network multi-agent simulation model of the spread of the Covid-19 virus. The model can
be used to simulate and analyse a spread of the Covid-19 virus on the level of the individual.
This distinguishes the model from most models, that have a more high level perspective. This makes
our model particularly usefull for researchers and policy makers that want to study the impact of
heterogeneity in the population (e.g. by modelling neighbourhoods with vulnerable populations) and
who want to study the impact of targeted measures (e.g. quarantining certain neighbourhoods,
or targetted social distancing measures at aimed at specific segments of the population).

The model is inspired by the canonical SEIRS structure and generates curves that reflect the amount of agents that are susceptible (s) infected without symptoms (i1), with symptoms (i2), critically ill (c), and are recovered (r). 


[comment]: <> (The output of one simulation might look something like this:  ) 

[comment]: <> (<img src="https://github.com/joerischasfoort/joerischasfoort.github.io/blob/master/images/the_curve.png" height="512px"/> ) 

Due to the unique spatial structure of the model, we can track how a virus spreads spatially. For example through Cape Town. The figure below shows the quantity of the population infected in different Wards in the City of Cape town. Note that thisconcerns a hypothetical simulation of the non-calibrated model and is only used to give an idea of possible dynamics. 

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
There are two options:

1) you run the __Initialization__, __Simulation__ and  __Save network data__ cells from within the notebook
2) you run  ```python SABCoModel.py```


__Analysis__
The notebook contains a section that produces the output graphs


__Requirements__
The program requires python 3 and the geopy, networkx non-standard packages.


__Disclaimer__

This software is intended for educational and research purposes. Despite best efforts,
we cannot fully rule out the possibility of errors and bugs. The use of SABCoM
is entirely at your own risk.
