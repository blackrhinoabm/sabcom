[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python application](https://github.com/blackrhinoabm/sabcom/workflows/Python%20application/badge.svg)

<img src="https://pbs.twimg.com/profile_images/1270246832015314953/CW4YcWdd_400x400.jpg" width="125">

![](https://cogeorg.github.io/images/black_rhino_logo.jpg)

 __The Social Agent-Based Covid-19 Model (SABCOM)__

SABCOM is an open source, easy-to-use-and-adapt, spatial network, multi-agent, model that can be used to simulate the effects of different lockdown policy measures on the spread of the Covid-19 virus in several (South African) cities. 

# Installation

The first step of working with sabcom is installing it. The easiest way to install it is using the Python pip package installer.

Using pip, type the following command in your terminal:

```bash
$ pip install sabcom
```

or alternatively

```bash
$ pip3 install sabcom
```

It is also possible to install sabcom manually via git using the following command:

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

Then, make sure you have correctly formatted input data and the model should be ready for simulation. For an exact description on how to do so, we refer to the [documentation](https://sabcom.co.za/docs/build/html/index.html).


# Requirements
The program requires Python 3, and the packages listed in the requirements.txt file.

# Website and Social Media
https://sabcom.co.za

https://twitter.com/SABCOM5

# Disclaimer

This software is intended for educational and research purposes. Despite best efforts,
we cannot fully rule out the possibility of errors and bugs. The use of SABCoM
is entirely at your own risk.
