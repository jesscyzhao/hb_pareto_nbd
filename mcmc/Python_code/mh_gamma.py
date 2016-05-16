import math
import numpy as np
import copy


def mh_gamma_chain(data, init_prior, hyper_prior, burnin=1000, num_iter=5000, thin=50, trace=100):

    num_draws = (num_iter-1)/thin + 1
    output = np.zeros((num_draws, 2))

    # Initialization
    current_param = init_prior
    counter = 0

    for step in range(1, burnin + num_iter+1):

        if (step > burnin) and ((step-1-burnin)%thin==0):
            idx = (step-1-burnin)/thin + 1
            output[idx-1, ] = copy.deepcopy(current_param)
            print output[idx-1, ]

        if step% trace == 0:
            print 'step ' + str(step) + ' of total ' + str(burnin+num_iter) + ' steps'


        one_mh_step_result = mh_gamma_step(data, current_param[0],current_param[1], hyper_prior)

        if one_mh_step_result[0] != current_param[0]:
            counter += 1

        if one_mh_step_result[1] != current_param[1]:
            counter += 1

        current_param = one_mh_step_result

    return {'output': output, 'accept_rate': counter/(2*(burnin+num_iter))}


def mh_gamma_step(x, shape, rate, hyper_prior):

    prior = [shape, rate]
    lower = 0
    upper = 100

    for i in range(len(prior)):
        propose_prior = copy.deepcopy(prior)
        current_log_post = log_gamma_posterior(prior, hyper_prior, x)
        # proposal distribution is normal
        # sample within boundary (0, 100)
        propose_param = min(max(1e-10, prior[i] + np.random.normal(0, 1, 1)), 100)
        propose_prior[i] = propose_param
        propose_log_post = log_gamma_posterior(propose_prior, hyper_prior, x)

        u = np.random.uniform(0, 1, 1)
        r = 0
        try:
            r = math.exp(max(propose_log_post-current_log_post, -700))
        except OverflowError:
            r = 0

        if u < r:
            prior = copy.deepcopy(propose_prior)

    return prior


def log_gamma_posterior(prior, hyper_prior, x):

    shape, rate = prior

    shape_hyper_1, shape_hyper_2, rate_hyper_1, rate_hyper_2 = hyper_prior
    
    sum_x = sum(x)
    sum_log_x = sum([math.log(i) for i in x])
    n = len(x)
    
    log_post = (n * (shape * math.log(rate) - math.lgamma(shape)) + (shape-1) * sum_log_x - rate * sum_x +
      (shape_hyper_1 - 1) * math.log(shape) - (shape * shape_hyper_2) +
      (rate_hyper_1 - 1) * math.log(rate) - (rate * rate_hyper_2))

    return log_post


def density(sample, num_bins=1e5):
    sample_range = max(sample) - min(sample)
    band_width = sample_range / num_bins
    bin_height_unit = 1/(band_width * len(sample))

    bin_lower = min(sample)
    bin_upper = bin_lower+band_width

    density_list = list()
    while bin_upper < max(sample):
        greater_than_lower = sample[sample>bin_lower]
        smaller_than_uppder = greater_than_lower[greater_than_lower<=bin_upper]
        density_list = density_list + [len(smaller_than_uppder) * bin_height_unit]
        bin_upper += band_width
        bin_lower += band_width
        if bin_upper == max(sample):
            print "Finished calculating density"
    density_list = [density_list[0] +1] + density_list[1:]
    assert sum(density_list) == 1, 'Something is wrong'

    return density_list


if __name__=='__main__':

    shape = 3
    rate = 5
    n = 5000
    data = np.random.gamma(shape, scale=1.0/rate, size=n)

    hyper_prior = [1e-3, 1e-3, 1e-3, 1e-3]
    init_prior = [0.5, 0.5]

    num_iter = 5000

    chain_1 = mh_gamma_chain(data, init_prior, hyper_prior, num_iter=num_iter)

    # simplified version of Geweke Diagnostics
    mcmc_sample = chain_1['output']
    mid_point = mcmc_sample.shape[0]/2
    for i in range(mcmc_sample.shape[1]):
        print 'param %s' %i
        first_half_mean = np.mean(mcmc_sample[:mid_point, i])
        second_half_mean = np.mean(mcmc_sample[mid_point:, i])
        overall_mean = np.mean(mcmc_sample[:, i])
        print 'overall mean=%s, first half mean=%s, second half mean=%s' %(overall_mean, first_half_mean,
                                                                           second_half_mean)
        first_half_var = np.std(mcmc_sample[:mid_point, i])
        second_half_var = np.std(mcmc_sample[mid_point:, i])
        overall_var = np.std(mcmc_sample[:, i])
        print 'overall var=%s, first half var=%s, second half var=%s' %(overall_var, first_half_var, second_half_var)

