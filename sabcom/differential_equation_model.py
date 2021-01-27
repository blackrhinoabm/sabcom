import numpy as np


def differential_equations_model(compartments, t, transmissibility, contact_probability_matrix,
                                 exit_rate_exposed, exit_rate_asymptomatic,
                                 exit_rate_symptomatic, exit_rate_critical,
                                 probability_symptomatic, probability_critical, probability_to_die, hospital_capacity,
                                 time_varying_contact_rates):
    # reshape 63 element vector Z into [7 x 9] matrix
    compartments = compartments.reshape(7, -1)

    # assign rows to disease compartments
    susceptible, exposed, asymptomatic, symptomatic, critical, recovered, dead = compartments

    health_overburdened_multiplier = 1

    # health system can be overburdened which will increase the probability of death
    if critical.sum() > hospital_capacity:
        health_overburdened_multiplier = 1.79 #TODO add this as a parameter
        probability_to_die = np.minimum(health_overburdened_multiplier * probability_to_die, np.ones(9))

    contact_rate = time_varying_contact_rates[int(t)] ** 2

    # construct differential equation evolution equations
    delta_susceptible = -transmissibility * contact_rate * susceptible * contact_probability_matrix.dot((asymptomatic + symptomatic))
    delta_exposed = transmissibility * contact_rate * susceptible * contact_probability_matrix.dot((
            asymptomatic + symptomatic)) - exit_rate_exposed * exposed
    delta_asymptomatic = (1 - probability_symptomatic
                          ) * exit_rate_exposed * exposed - exit_rate_asymptomatic * asymptomatic
    delta_symptomatic = probability_symptomatic * exit_rate_exposed * exposed - exit_rate_symptomatic * symptomatic
    delta_critical = probability_critical * exit_rate_symptomatic * symptomatic - exit_rate_critical * critical
    delta_recovered = exit_rate_asymptomatic * asymptomatic + (
            1 - probability_critical) * exit_rate_symptomatic * symptomatic + (1 - probability_to_die
                                                                               ) * exit_rate_critical * critical
    delta_dead = probability_to_die * exit_rate_critical * critical

    # store differentials as 63 element vector
    delta_compartments = np.concatenate((delta_susceptible, delta_exposed, delta_asymptomatic,
                                         delta_symptomatic, delta_critical, delta_recovered, delta_dead), axis=0)

    return delta_compartments
