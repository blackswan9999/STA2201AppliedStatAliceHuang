//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

data {
  int<lower=0> N;
  vector[N] log_gest;    // 
  vector[N] log_weight;     // 
  vector[N] preterm;//
  vector[N] log_gest_preterm;//
  vector[N] sex;//
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  vector[5] beta;           // coefs
  real<lower=0> sigma;  // error sd for Gaussian likelihood
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  // Log-likelihood
  log_weight ~ bernoulli_logit(beta[1] + beta[2] * log_gest + beta[3]*preterm + beta[4]*log_gest_preterm + beta[5]*sex, sigma);

  // Log-priors
  target += normal_lpdf(sigma | 0, 1);
  target += normal_lpdf(beta | 0, 1);
}

generated quantities {
  vector[N] log_lik;    // pointwise log-likelihood for LOO
  vector[N] log_weight_rep; // replications from posterior predictive dist

  for (n in 1:N) {
    real log_weight_hat_n = beta[1] + beta[2] * log_gest[n] + beta[3]*preterm[n] + beta[4]*log_gest_preterm[n] + beta[5]*sex[n];
    log_lik[n] = bernouilli_lpdf(log_weight[n] | log_weight_hat_n, sigma);
    log_weight_rep[n] = normal_rng(log_weight_hat_n, sigma);
  }
}
