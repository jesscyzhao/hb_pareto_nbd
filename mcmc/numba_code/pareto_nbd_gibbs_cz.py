import math
import numpy as np
import copy


def run_single_chain(data, init_param, chain_id=1, num_iter=3000, burn_in=1000, thin=50, trace=100):

    def draw_latent_variables(data, latent_variables, gamma_parameters):
        pass

    def draw_gamma_parameters(latent_variables, gamma_parameters, hyper_prior):
        pass

    N = data.shape[0]
    NUM_DRAWS = (num_iter-1)/thin + 1

    latent_variables_draws = np.zeros(NUM_DRAWS, dtype=[('lambda', float, (1, N)), ('mu', float, (1, N)),
                                                        ('tau', float, (1, N)), ('alive', int, (1, N))])
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
            idx = (step-1-burn_in)/thin + 1
            latent_variables_draws[idx] = copy.deepcopy(latent_variables)
            gamma_parameters_draws[idx] = copy.deepcopy(gamma_parameters)

        latent_variables = draw_latent_variables(data, latent_variables, gamma_parameters)
        gamma_parameters = draw_gamma_parameters(latent_variables, gamma_parameters, hyper_prior)

    return {'latent_param': latent_variables_draws, 'gamma_param': gamma_parameters_draws}

