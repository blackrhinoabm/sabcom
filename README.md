<img src="https://cogeorg.github.io/images/black_rhino_logo.jpg" height="64px"/>
<img src="https://github.com/joerischasfoort/joerischasfoort.github.io/blob/master/images/covi-id.png" height="64px"/>

[comment]: <> (One paragraph overview of the project, TODO add link to blog?)
**The Spatial Agent-Based Covid Model (SABCom)** is an open source easy-to-use-and-adapt 
spatial network multi-agent simulation model of the spread of the Covid-19 virus. The model can 
be used to simulate and analyse a spread of the Covid-19 virus on the level of the individual.
This distinguishes the model from most models, that have a more high level perspective. This makes 
our model particularly usefull for researchers and policy makers that want to study the impact of 
heterogeneity in the population (e.g. by modelling neighbourhoods with vulnerable populations) and
who want to study the impact of targeted measures (e.g. quartantaining certain neighbourhoods,
or targetted social distancing measures at aimed at specific segments of the population). 
 
 __Getting started__

You can **install** the SABCom model by cloning this repository to your system. After that, you are ready to start using the model.

__Running the model__
There are two files in the main folder that will allow you to run the model. The first is the 
SABCoModel.py model. This will run the model using the neighbourhood_data.json and parameters.json
files. These are the calibration files for Cape Town. The second file to run the model is the Jupyter notebook:
SABCoModel_notebook.ipynb . Running the model using this file allows you to change the parameters. Furthermore,
it contains a section that produces the output graphs. 

__Disclaimer__

This software is intended for educational and research purposes. Despite best efforts, 
we cannot fully rule out the possibility of errors and bugs. The use of SABCoM
is entirely at your own risk.