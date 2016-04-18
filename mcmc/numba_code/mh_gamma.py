# import numba 
import math 
import numpy as np
import copy 
import pymc

def mh_gamma_chain(data, init_prior, hyper_prior, burnin=1000, num_iter=5000, thin=50, trace=100):
	

	num_draws = (num_iter-1)/thin + 1
	output = np.zeros((num_draws, 2))
	
	#Initialization 
	current_param = init_prior
	counter = 0

	for step in range(1, burnin + num_iter+1):

		if step% trace ==0: 
			print 'step ' + str(step) + ' of total ' + str(burnin+num_iter) + ' steps'
		
		if (step > burnin) and ((step-1-burnin)%thin==0):
			idx = (step-1-burnin)/thin + 1
			output[idx-1, ] = copy.deepcopy(current_param)

		one_mh_step_result = mh_gamma_step(data, current_param[0],current_param[1], hyper_prior)

		if one_mh_step_result[0] != current_param[0]:
			counter+1

		if one_mh_step_result[1] != current_param[1]:
			counter+1

		current_param = one_mh_step_result

	return {'output': output, 'accept_rate': counter/(2*(burnin+num_iter))}

def mh_gamma_step(x, shape, rate, hyper_prior):

	prior = [shape, rate]
	lower = 0
	upper = 100

	for i in range(len(prior)):
		propose_prior = prior 
		current_log_post = log_gamma_posterior(prior, hyper_prior, x)
	    # proposal distribution is normal 
	    # sample within boundary (0, 100)
		propose_param = min(max(1e-10, prior[i] + np.random.normal(0, 1, 1)),100)
		propose_prior[i] = propose_param
		propose_log_post = log_gamma_posterior(propose_prior, hyper_prior, x)

		u = np.random.uniform(0, 1, 1)
		r = 0 
		try:
			r = math.exp(max(propose_log_post-current_log_post, -700))
		except OverflowError:
			r = 0


		if u < r:
			prior = propose_prior 

	return prior 

def log_gamma_posterior(prior, hyper_prior, x):

	shape, rate = prior

	shape_hyper_1,shape_hyper_2, rate_hyper_1,rate_hyper_2 = hyper_prior
    
	sum_x = sum(x)
	sum_log_x = sum([math.log(i) for i in x])
	n = len(x)
    
	log_post = (n * (shape * math.log(rate) - math.lgamma(shape)) + (shape-1) * sum_log_x - rate * sum_x + 
      (shape_hyper_1 - 1) * math.log(shape) - (shape * shape_hyper_2) +
      (rate_hyper_1 - 1) * math.log(rate) - (rate * rate_hyper_2))

	return log_post


if __name__=='__main__':

	shape = 3
	rate = 5
	n = 5000
	data = np.random.gamma(shape, scale=1.0/rate, size=n)

	hyper_prior = [1e-3, 1e-3, 1e-3, 1e-3]
	init_prior = [0.5, 0.5]

	chain_1 = mh_gamma_chain(data, init_prior, hyper_prior)

	print chain_1['output']

	mcmc_object = pymc.MCMC(chain_1)
	print pymc.utils.hpd(mcmc_object, 1.-0.95)


