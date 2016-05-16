import math
import numpy as np
import pandas as pd
import copy
import os
from simulate_pareto_nbd import generate_pareto_nbd_data

SIMULATE = False

def run_single_chain(data, init_param, chain_id=1, num_iter=3000, burn_in=1000, thin=50, trace=10):

    N = data.shape[0]
    NUM_DRAWS = (num_iter-1)/thin + 1

    latent_variables_draws = np.zeros(NUM_DRAWS, dtype=[('lambda', float, N), ('mu', float, N),
                                                        ('tau', float, N), ('alive', int, N)])
    gamma_parameters_draws = np.zeros(NUM_DRAWS, dtype=[('r', float), ('alpha', float), ('s', float), ('beta', float)])

    hyper_prior = [1e-3]*8

    gamma_parameters = copy.deepcopy(gamma_parameters_draws[0])
    gamma_parameters['r'] = init_param['r']
    gamma_parameters['alpha'] = init_param['alpha']
    gamma_parameters['s'] = init_param['s']
    gamma_parameters['beta'] = init_param['beta']

    latent_variables = copy.deepcopy(latent_variables_draws[0])
    latent_variables['lambda'] = np.mean(data['x']) / np.mean([data['T.cal'][i] if data['t.x'][i] == 0 else
                                                               data['t.x'][i] for i in range(N)])
    latent_variables['tau'] = [data['t.x'][i] + 0.5/latent_variables['lambda'][i] for i in range(N)]
    latent_variables['alive'] = [int(latent_variables['tau'][i] > data['T.cal'][i]) for i in range(N)]
    latent_variables['mu'] = [1/latent_variables['tau'][i] for i in range(N)]

    for step in range(1, (num_iter + burn_in+1)):
        if step % trace == 0:
            print 'step %s of chain %s' %(step, chain_id)

        if (step > burn_in) and ((step-1-burn_in)%thin==0):
            idx = (step-1-burn_in)/thin
            print gamma_parameters
            latent_variables_draws[idx] = copy.deepcopy(latent_variables)
            gamma_parameters_draws[idx] = copy.deepcopy(gamma_parameters)

        latent_variables = draw_latent_variables(data, latent_variables, gamma_parameters)
        gamma_parameters = draw_gamma_parameters(latent_variables, gamma_parameters, hyper_prior)

    return {'latent_param': latent_variables_draws, 'gamma_param': gamma_parameters_draws}


def draw_latent_variables(data, latent_variables, gamma_parameters):
    x = data['x']
    T_cal = data['T.cal']
    tx = data['t.x']
    r, alpha, s, beta = gamma_parameters

    proposed_latent_variables = copy.deepcopy(latent_variables)
    proposed_latent_variables['lambda'] = draw_lambda(r, alpha, x, T_cal, proposed_latent_variables['tau'])
    proposed_latent_variables['mu'] = draw_mu(s, beta, proposed_latent_variables['tau'], T_cal)
    proposed_latent_variables['tau'] = draw_tau(proposed_latent_variables['lambda'], proposed_latent_variables['mu'],
                                                tx, T_cal)
    proposed_latent_variables['alive'] = (proposed_latent_variables['tau'] > T_cal) * 1

    return proposed_latent_variables


def log_gamma_posterior(prior, hyper_prior, x):

    shape, rate = prior

    shape_hyper_1, shape_hyper_2, rate_hyper_1, rate_hyper_2 = hyper_prior

    sum_x = sum(x)
    sum_log_x = sum([math.log(i) for i in x])
    n = len(x)

    log_post = (n*(shape*math.log(rate) - math.lgamma(shape))+(shape-1)*sum_log_x - rate*sum_x +
                (shape_hyper_1-1)*math.log(shape) - (shape*shape_hyper_2) + (rate_hyper_1 - 1) * math.log(rate) -
                (rate*rate_hyper_2))

    return log_post


def slice_sampling_step(x, shape, rate, hyper_prior, steps=10, lower=1e-5, upper=100, w=1):

    prior = [shape, rate]

    log_post_prior = log_gamma_posterior(prior, hyper_prior, x)

    for i in range(steps):
        for j in range(len(prior)):
            center = copy.deepcopy(prior)
            left = copy.deepcopy(prior)
            right = copy.deepcopy(prior)
            log_post_z = log_post_prior - np.random.exponential(scale=1)

            u = np.random.uniform(0, 1)*w

            left[j] = max(center[j]-u, lower)
            right[j] = min(center[j] + (w-u), upper)

            while left[j] > lower:
                if log_gamma_posterior(left, hyper_prior, x) > log_post_z:
                    left[j] -= w
                else:
                    break

            while right[j] < upper:
                if log_gamma_posterior(right, hyper_prior, x) > log_post_z:
                    right[j] += w
                else:
                    break

            param_lower = max(lower, left[j])
            param_upper = min(upper, right[j])

            propose_prior = copy.deepcopy(center)

            count = 1

            while count < 1e2:

                propose_param = np.random.uniform(param_lower, param_upper)
                propose_prior[j] = copy.deepcopy(propose_param)

                log_propose_prior = log_gamma_posterior(propose_prior, hyper_prior, x)

                if log_propose_prior > log_post_prior:
                    break

                if propose_param < prior[j]:
                    param_lower = copy.deepcopy(propose_param)

                else:
                    param_upper = copy.deepcopy(propose_param)

                count += 1

            prior = copy.deepcopy(propose_prior)
            log_post_prior = copy.deepcopy(log_propose_prior)

    return prior


def draw_gamma_parameters(latent_variables, gamma_parameters, all_hyper_prior):

    r_alpha_update = slice_sampling_step(latent_variables['lambda'], gamma_parameters['r'], gamma_parameters['alpha'],
                                         all_hyper_prior[:4])
    gamma_parameters['r'], gamma_parameters['alpha'] = r_alpha_update

    s_beta_update = slice_sampling_step(latent_variables['mu'], gamma_parameters['s'], gamma_parameters['beta'],
                                        all_hyper_prior[4:])
    gamma_parameters['s'], gamma_parameters['beta'] = s_beta_update

    return gamma_parameters


#@vectorize(["float64(float64, float64, int64, float64, float64)"], nopython=True, target='parallel')
def draw_lambda(r, alpha, x, T_cal,tau):
    N = len(x)
    Lambda = [np.random.gamma(shape=r + x[i], scale=1/(alpha + min(T_cal[i], tau[i]))) for i in range(N)]
    return Lambda


def draw_mu(s, beta, tau, T_cal):
    N = len(T_cal)
    mu = np.zeros(N)
    alive_customers = np.where(tau > T_cal)[0]
    dead_customers = np.where(tau < T_cal)[0]
    if len(alive_customers) > 0:
        mu[alive_customers] = [np.random.gamma(shape=s, scale=1/(beta+T_cal[i])) for i in alive_customers]

    if len(dead_customers) > 0:
        mu[dead_customers] = [np.random.gamma(shape=s+1, scale=1/(beta+tau[i])) for i in dead_customers]

    return mu


def draw_tau(Lambda, mu, tx, T_cal):
    N = len(tx)
    mu_lam = mu + Lambda
    t_diff = np.array(T_cal - tx)

    mu_lam_times_t_diff = mu_lam * t_diff
    p_alive = 1 / (1+(mu/mu_lam)*(np.array([math.exp(i) for i in mu_lam_times_t_diff])-1))

    rand_unif = np.random.uniform(0, 1, N)
    alive_customers = np.where(p_alive > rand_unif)[0]
    dead_customers = np.where(p_alive < rand_unif)[0]

    tau = np.zeros(N)

    if len(alive_customers) > 0:
        tau[alive_customers] = [T_cal[i] + np.random.exponential(1/mu[i]) for i in range(N)]

    if len(dead_customers) > 0:
        mu_lam_tx = mu_lam[dead_customers] * tx[dead_customers]
        mu_lam_tx = np.array([item if item <= 700 else 700 for item in mu_lam_tx])
        mu_lam_T_cal = mu_lam[dead_customers] * T_cal[dead_customers]
        mu_lam_T_cal = np.array([item if item <= 700 else 700 for item in mu_lam_T_cal])

        rand = np.random.uniform(0, 1, len(dead_customers))

        term_1 = np.exp(mu_lam_tx * -1.0)
        term_2 = np.exp(mu_lam_T_cal * -1.0)

        numerator = -1.0 * np.log((1-rand)*term_1 + rand*term_2)
        tau[dead_customers] = numerator / mu_lam[dead_customers]

    return tau


if __name__ == '__main__':

    if SIMULATE:
        sim_shape = 1
        sim_rate = 5
        N = 5000
        sim_data = np.random.gamma(shape=sim_shape, scale=1.0/sim_rate, size=N)

        hyper_prior = [1e-3] * 4
        test_gamma_slice_sampling = slice_sampling_step(sim_data, 0.5, 0.5, hyper_prior, steps=5)

        # slice sampling seems to be working.

        sim_parameters = {'r': 2.0,
                          'a': 10.0,
                          's': 2.0,
                          'b': 10.0
                          }

        num_customer = 5000
        data = generate_pareto_nbd_data(sim_parameters, num_customer)


    data = pd.read_csv(os.path.join('..', 'test_data', 'pnbd_gibbs_sampling_sim_data.csv'))
    data.index = range(data.shape[0])
    print data

    init_param = {'r': 0.5, 'alpha': 0.5, 's': 0.5, 'beta': 0.5}
    mcmc_sample = run_single_chain(data, init_param, num_iter=100, burn_in=10)



