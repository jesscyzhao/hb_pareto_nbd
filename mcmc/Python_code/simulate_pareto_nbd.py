################### BEGIN TWO SIX HEADER #########################

# add top level ballast to system path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

# import ballast as a check
import numpy as np
import pandas as pd
import datetime as dt

################### END TWO SIX HEADER ###########################

import os

N_MIN = 1
N_MAX = 100
NUM_TRANS_SCALAR = [1, 2, 4, 800]


def simulate_customer(lambda_param, mu, n):
    """
    Simulate a single customer given their lambda, mu, and lifetime (n).

    :param lambda_param: float
    :param mu: float
    :param n: float
    :return: x = frequency (float), tx = recency (float)
    """
    tau = np.random.exponential(1 / mu)

    minT = min(n, tau)
    num_draws = max(10, int(minT*(1/lambda_param)))

    itts = np.zeros(num_draws)
    count = 0
    while sum(itts) < minT:
        scalar = NUM_TRANS_SCALAR[count]
        itts = np.random.exponential(1 / lambda_param, num_draws * scalar)
        if sum(itts) > minT:
            break
        else:
            count += 1
            if scalar == NUM_TRANS_SCALAR[3]:
                raise UserWarning('not enough inter-transaction time sampled')

    trans_times = np.cumsum([0] + list(itts))
    valid_trans_times = trans_times[trans_times < tau]
    train_trans_times = valid_trans_times[valid_trans_times < n]

    x = len(train_trans_times) - 1
    tx = max(train_trans_times)
    if x > 0 and tx == 0.0:
        raise UserWarning('when x > 0, tx should be greater than 0')
    if n < tx:
        raise UserWarning('n should be greater than tx, something is wrong')
    return x, tx


def generate_pareto_nbd_data(parameters, num_customer):
    rng = np.random.RandomState()
    rng.seed(1)
    # notice that numpy gamma parametrizes differently: np.random.gamma(k, theta, size), k = r, theta = 1/a
    lambda_params = rng.gamma(parameters['r'], 1 / parameters['a'], num_customer)
    rng.seed(2)
    mus = rng.gamma(parameters['s'], 1 / parameters['b'], num_customer)
    rng.seed(3)
    ns = [rng.uniform(0.001, 1) * N_MAX for _ in range(num_customer)]

    param_list = list(zip(lambda_params, mus, ns))
    xtxn = list()
    counter = 0
    for param in param_list:
        counter += 1
        lambda_param, mu, n = param
        x, tx = simulate_customer(lambda_param, mu, n)

        if counter % 10000 == 0:
            print x, tx, n

        xtxn.append((x, tx, n))

    xtxn_df = pd.DataFrame(xtxn, columns=['x', 't.x', 'T.cal'])

    return xtxn_df


