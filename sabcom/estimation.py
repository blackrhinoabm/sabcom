import numpy as np
import pandas as pd
import json
import math
import os
import scipy.stats as stats
import scipy.optimize as sciopt

from sabcom.runner import runner


def ls_model_performance(input_params, input_folder_path, mc_runs, output_folder_path, scenario, names,
                         sensitivity_parameters, learning_scenario, inititialisation_path, newly_susceptible_percentage):
    """
    Simple function calibrate uncertain model parameters
    :param input_parameters: list of input parameters
    :return: cost
    """
    parameter_json_path = os.path.join(input_folder_path, 'parameters.json')
    mc_runs = mc_runs

    # open estimation_parameter.json file to extract special parameters used for the estimation
    with open(parameter_json_path) as json_file:
        param_file = json.load(json_file)

    emp_fatality_curve = param_file['empirical_fatalities']
    empirical_population = param_file['empirical_population']

    for name, par in zip(names, input_params):
        if name not in ['visiting_recurring_contacts_multiplier', 'total_initial_infections']:
            param_file[name] = par
        if name == 'visiting_recurring_contacts_multiplier':
            vis_rec = par
        else:
            vis_rec = None
        if name == 'total_initial_infections':
            initial_infections = par
        else:
            initial_infections = None
        if name == 'death_response_intensity':
            dri = par
        else:
            dri = None

    for key in sensitivity_parameters:
        param_file[key] = sensitivity_parameters[key]
        print('Parameter {} has been updated to {}'.format(key, param_file[key]))

    param_file["time"] = len(emp_fatality_curve)

    # dump in config file
    with open('estimation_parameters.json', 'w') as outfile:
        json.dump(param_file, outfile)

    costs = []
    for seed in range(mc_runs):
        # check if the seed can be found in the input folder, if not skip seed
        seed_path = os.path.join(inititialisation_path, 'seed_{}.pkl'.format(seed))

        if not os.path.exists('{}'.format(seed_path)):
            print('Path {} not found, will continue loop'.format(seed_path))
            continue

        # run model with parameters.
        environment = runner(input_folder_path=input_folder_path,
                             output_folder_path=output_folder_path, seed=seed,
                             scenario=scenario,
                             learning_scenario=learning_scenario,
                             initial_infections=initial_infections,
                             death_response_intensity=dri,
                             visiting_recurring_contacts_multiplier=vis_rec,
                             newly_susceptible_percentage=newly_susceptible_percentage,
                             stringency_changed=True,
                             sensitivity_config_file_path='estimation_parameters.json',
                             initial_seeds_folder=inititialisation_path)

        sim_dead_curve = pd.DataFrame(environment.infection_quantities)['d'] * (empirical_population / param_file['number_of_agents'])
        sim_dead_curve = sim_dead_curve.diff().ewm(span=10).mean()

        # calculate the cost
        costs.append(ls_cost_function(emp_fatality_curve, sim_dead_curve))
        print(costs[-1])

    return np.mean(costs)


def confidence_interval(data, av):
    sample_stdev = np.std(data)
    sigma = sample_stdev/math.sqrt(len(data))
    return stats.t.interval(alpha = 0.95, df= 24, loc=av, scale=sigma)


def ls_cost_function(observed_values, average_simulated_values):
    """
    Simple cost function to calculate average squared deviation of simulated values from observed values
    :param observed_values: list of observed data points
    :param average_simulated_values: list of corresponding observed data points
    :return:
    """
    score = 0
    for x, y in zip(observed_values, average_simulated_values):
        if x > 0.0:
            score += np.true_divide((x - y), x)**2
        else:
            score += 0

    if np.isnan(score):
        return np.inf
    else:
        return score


def cost_function(observed_values, average_simulated_values):
    """
    Simple cost function to calculate average squared deviation of simulated values from observed values
    :param observed_values: dictionary of observed stylized facts
    :param average_simulated_values: dictionary of corresponding simulated stylized facts
    :return:
    """
    score = 0
    for key in observed_values:
        score += np.true_divide((observed_values[key] - average_simulated_values[key]), observed_values[key])**2

    if np.isnan(score):
        return np.inf
    else:
        return score


# =====================================================================================================================================
# Copyright
# =====================================================================================================================================

# Copyright (C) 2017 Alexander Blaessle.
# This software is distributed under the terms of the GNU General Public License.

# This file is part of constNMPY.

# constNMPy is a small python package allowing to run a Nelder-Mead optimization via scipy's fmin function.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# ===========================================================================================================================================================================
# Module Functions
# ===========================================================================================================================================================================

def constrNM(func, x0, LB, UB, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=0, disp=0,
             retall=0, callback=None):
    """Constrained Nelder-Mead optimizer.
    Transforms a constrained problem
    Args:
        func (function): Objective function.
        x0 (numpy.ndarray): Initial guess.
        LB (numpy.ndarray): Lower bounds.
        UB (numpy.ndarray): Upper bounds.
    Keyword Args:
        args (tuple): Extra arguments passed to func, i.e. ``func(x,*args).``
        xtol (float) :Absolute error in xopt between iterations that is acceptable for convergence.
        ftol(float) : Absolute error in ``func(xopt)`` between iterations that is acceptable for convergence.
        maxiter(int) : Maximum number of iterations to perform.
        maxfun(int) : Maximum number of function evaluations to make.
        full_output(bool) : Set to True if fopt and warnflag outputs are desired.
        disp(bool) : Set to True to print convergence messages.
        retall(bool): Set to True to return list of solutions at each iteration.
        callback(callable) : Called after each iteration, as ``callback(xk)``, where xk is the current parameter vector.
    """

    # Check input
    if len(LB) != len(UB) or len(LB) != len(x0):
        raise ValueError('Input arrays have unequal size.')

    # Check if x0 is within bounds
    for i, x in enumerate(x0):

        if LB[i] is not None:
            if x < LB[i]:
                errStr = 'Initial guess x0[' + str(i) + ']=' + str(x) + ' out of bounds.'
                raise ValueError(errStr)

        if UB[i] is not None:
            if x > UB[i]:
                errStr = 'Initial guess x0[' + str(i) + ']=' + str(x) + ' out of bounds.'
                raise ValueError(errStr)

    # Transform x0
    x0 = transformX0(x0, LB, UB)

    # Stick everything into args tuple
    opts = tuple([func, LB, UB, args])

    # Call fmin
    res = sciopt.fmin(constrObjFunc, x0, args=opts, ftol=ftol, xtol=xtol, maxiter=maxiter, disp=disp,
                      full_output=full_output, callback=callback, maxfun=maxfun, retall=retall)

    # Convert res to list
    res = list(res)

    # Dictionary for results
    rDict = {'fopt': None, 'iter': None, 'funcalls': None, 'warnflag': None, 'xopt': None, 'allvecs': None}

    # Transform back results
    if full_output or retall:
        r = transformX(res[0], LB, UB)
    else:
        r = transformX(res, LB, UB)
    rDict['xopt'] = r

    # If full_output is selected, enter all results in dict
    if full_output:
        rDict['fopt'] = res[1]
        rDict['iter'] = res[2]
        rDict['funcalls'] = res[3]
        rDict['warnflag'] = res[4]

    # If retall is selected, transform back all values and append to dict
    if retall:
        allvecs = []
        for r in res[-1]:
            allvecs.append(transformX(r, LB, UB))
        rDict['allvecs'] = allvecs

    return rDict


def constrObjFunc(x, func, LB, UB, args):
    r"""Objective function when using Constrained Nelder-Mead.
    Calls :py:func:`TransformX` to transform ``x`` into
    constrained version, then calls objective function ``func``.
    Args:
        x (numpy.ndarray): Input vector.
        func (function): Objective function.
        LB (numpy.ndarray): Lower bounds.
        UB (numpy.ndarray): Upper bounds.
    Keyword Args:
        args (tuple): Extra arguments passed to func, i.e. ``func(x,*args).``
    Returns:
         float: Return value of ``func(x,*args)``.
    """

    # print x
    x = transformX(x, LB, UB)
    # print x
    # raw_input()

    return func(x, *args)


def transformX(x, LB, UB, offset=1E-20):
    r"""Transforms ``x`` into constrained form, obeying upper bounds ``UB`` and lower bounds ``LB``.
    .. note:: Will add tiny offset to LB if ``LB[i]=0``, to avoid singularities.
    Idea taken from http://www.mathworks.com/matlabcentral/fileexchange/8277-fminsearchbnd--fminsearchcon
    Args:
        x (numpy.ndarray): Input vector.
        LB (numpy.ndarray): Lower bounds.
        UB (numpy.ndarray): Upper bounds.
    Keyword Args:
        offset (float): Small offset added to lower bound if LB=0.
    Returns:
        numpy.ndarray: Transformed x-values.
    """

    # Make sure everything is float
    x = np.asarray(x, dtype=np.float64)
    # LB=np.asarray(LB,dtype=np.float64)
    # UB=np.asarray(UB,dtype=np.float64)

    # Add offset if necessary to avoid singularities
    for l in LB:
        if l == 0:
            l = l + offset

    # Determine number of parameters to be fitted
    nparams = len(x)

    # Make empty vector
    xtrans = np.zeros(np.shape(x))

    # k allows some variables to be fixed, thus dropped from the
    # optimization.
    k = 0

    for i in range(nparams):

        # Upper bound only
        if UB[i] != None and LB[i] == None:

            xtrans[i] = UB[i] - x[k] ** 2
            k = k + 1

        # Lower bound only
        elif UB[i] == None and LB[i] != None:

            xtrans[i] = LB[i] + x[k] ** 2
            k = k + 1

        # Both bounds
        elif UB[i] != None and LB[i] != None:

            xtrans[i] = (np.sin(x[k]) + 1.) / 2. * (UB[i] - LB[i]) + LB[i]
            xtrans[i] = max([LB[i], min([UB[i], xtrans[i]])])
            k = k + 1

        # No bounds
        elif UB[i] == None and LB[i] == None:

            xtrans[i] = x[k]
            k = k + 1

        # NOTE: The original file has here another case for fixed variable. We might need to add this here!!!

    return np.array(xtrans)


def transformX0(x0, LB, UB):
    r"""Transforms ``x0`` into constrained form, obeying upper bounds ``UB`` and lower bounds ``LB``.
    Idea taken from http://www.mathworks.com/matlabcentral/fileexchange/8277-fminsearchbnd--fminsearchcon
    Args:
        x0 (numpy.ndarray): Input vector.
        LB (numpy.ndarray): Lower bounds.
        UB (numpy.ndarray): Upper bounds.
    Returns:
        numpy.ndarray: Transformed x-values.
    """

    # Turn into list
    x0u = list(x0)

    k = 0
    for i in range(len(x0)):

        # Upper bound only
        if UB[i] != None and LB[i] == None:
            if UB[i] <= x0[i]:
                x0u[k] = 0
            else:
                x0u[k] = np.sqrt(UB[i] - x0[i])
            k = k + 1

        # Lower bound only
        elif UB[i] == None and LB[i] != None:
            if LB[i] >= x0[i]:
                x0u[k] = 0
            else:
                x0u[k] = np.sqrt(x0[i] - LB[i])
            k = k + 1


        # Both bounds
        elif UB[i] != None and LB[i] != None:
            if UB[i] <= x0[i]:
                x0u[k] = np.pi / 2
            elif LB[i] >= x0[i]:
                x0u[k] = -np.pi / 2
            else:
                x0u[k] = 2 * (x0[i] - LB[i]) / (UB[i] - LB[i]) - 1;
                # shift by 2*pi to avoid problems at zero in fmin otherwise, the initial simplex is vanishingly small
                x0u[k] = 2 * np.pi + np.arcsin(max([-1, min(1, x0u[k])]));
            k = k + 1

        # No bounds
        elif UB[i] == None and LB[i] == None:
            x0u[k] = x0[i]
            k = k + 1

    return np.array(x0u)


def printAttr(name, attr, maxL=5):
    """Prints single attribute in the form attributeName = attributeValue.
    If attributes are of type ``list`` or ``numpy.ndarray``, will check if the size
    exceeds threshold. If so, will only print type and dimension of attribute.
    Args:
        name (str): Name of attribute.
        attr (any): Attribute value.

    Keyword Args:
        maxL (int): Maximum length threshold.
    """

    if isinstance(attr, (list)):
        if len(attr) > maxL:
            print(name, " = ", getListDetailsString(attr))
            return True
    elif isinstance(attr, (np.ndarray)):
        if min(attr.shape) > maxL:
            print(name, " = ", getArrayDetailsString(attr))
            return True

    print(name, " = ", attr)

    return True


def getListDetailsString(l):
    """Returns string saying "List of length x", where x is the length of the list.
    Args:
        l (list): Some list.
    Returns:
        str: Printout of type and length.
    """

    return "List of length " + str(len(l))


def getArrayDetailsString(l):
    """Returns string saying "Array of shape x", where x is the shape of the array.
    Args:
        l (numpy.ndarray): Some array.
    Returns:
        str: Printout of type and shape.
    """

    return "Array of shape " + str(l.shape)


def printDict(dic, maxL=5):
    """Prints all dictionary entries in the form key = value.
    If attributes are of type ``list`` or ``numpy.ndarray``, will check if the size
    exceeds threshold. If so, will only print type and dimension of attribute.
    Args:
        dic (dict): Dictionary to be printed.

    Returns:
        bool: True
    """

    for k in dic.keys():
        printAttr(k, dic[k], maxL=maxL)

    return True
