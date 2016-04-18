# test gamma_mh step 

library(coda)
source('three-model-comparison/cz_pareto_mcmc.r')
sim.shape = 3
sim.rate = 5

sim.data = rgamma(n=5000, shape=sim.shape, rate=sim.rate)
burnin = 5000
# Try num_iter = 20,000, 100,000, 200,000 See the difference. 
num_iter = 200000
thin=50

hyper_prior = c(1e-3, 1e-3, 1e-3, 1e-3)

init_prior = c(0.5,0.5)
run_1= mh_gamma_chain(sim.data, init_prior, hyper_prior, burnin, num_iter, thin)
init_prior = c(0.5, 10)
run_2= mh_gamma_chain(sim.data, init_prior, hyper_prior, burnin, num_iter, thin)

# Diagnosis 
# accept rate = # of accept moves / (burn_in + num_iter)
cat('accept rate for run 1 is', run_1$accept_rate)
cat('accept rate for run 2 is', run_2$accept_rate)

# Use coda 

mcmc_samples_1 = as.mcmc(run_1$sample_output, burnin, thin=thin)
mcmc_samples_2 = as.mcmc(run_2$sample_output, burnin, thin=thin)
gamma_mcmc_sample_list = list(mcmc_samples_1, mcmc_samples_2)

# Quantile 
lapply(gamma_mcmc_sample_list, summary) 

# HPD interval 
lapply(gamma_mcmc_sample_list, HPDinterval)

# Distribution 
hist(mcmc_samples_1[, 1], 100, main='Distribution of alpha')
hist(mcmc_samples_1[, 2], 100, main='Distribution of beta')

# Trace plot 
traceplot(mcmc_samples_1[, 1], main='Tranceplot of alpha')
traceplot(mcmc_samples_1[, 2], main='Tranceplot of beta')

# Autocorrelation 
autocorr.plot(mcmc_samples_1[,1], main='Autocorrelation of alpha')
autocorr.plot(mcmc_samples_1[,2], main='Autocorrelation of beta')

# Running mean 
plot(calc.running.mean(mcmc_samples_1[, 1])~seq(1, num_iter%/%thin), main='Running mean alpha', type='l', xlab='# of draws', ylab='Running mean')
plot(calc.running.mean(mcmc_samples_1[, 2])~seq(1, num_iter%/%thin), main='Running mean beta', type='l', xlab='# of draws', ylab='Running mean')

gelman.diag(gamma_mcmc_sample_list)

# Conclusion, MH is not very efficient. 

# Test out slice sampling 

slice_sampling_chain = function(x, shape, rate, hyper_prior, num_call=1, thin = 10){
  output = array(NA_real_, dim=c(num_call, 2))
  
  for (call in 1:num_call){
    prior = slice_sampling_step(x, shape, rate, hyper_prior, thin)
    output[call, 1] = prior$shape
    output[call, 2] = prior$rate
    shape = prior$shape
    rate = prior$rate
  }
  
  return (output)
}

init_prior = c(0.5, 0.5)
slice_sampling_output = slice_sampling_chain(sim.data, init_prior[1], init_prior[2], hyper_prior, num_call = 5)

# slice sampling is so much faster 

