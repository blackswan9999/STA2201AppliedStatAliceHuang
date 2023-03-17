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

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  int<lower=0, upper=1> switch_well[N];
  vector[N] cent_dist;
  vector[N] cent_log_arsenic;
  vector[N] cent_dist_cent_log_arsenic;
}

parameters {
  vector[4] beta;
}

model {
  // Log-likelihood
  // needs to be 1 and 0 going in
  switch_well ~ bernoulli_logit(beta[1] + beta[2]*cent_dist + beta[3]*cent_log_arsenic + beta[4]*cent_dist_cent_log_arsenic);

  // Log-priors
  target += normal_lpdf(beta | 0, 1);
}

generated quantities {
  vector[N] log_lik;    // pointwise log-likelihood for LOO
  vector[N] log_switch_rep; // replications from posterior predictive dist

  for (n in 1:N) {
    real log_switch_hat_n = beta[1] + beta[2]*cent_dist[n] + beta[3]*cent_log_arsenic[n] + beta[4]*cent_dist_cent_log_arsenic[n];
    log_lik[n] = bernoulli_logit_lpmf(switch_well[n] | inv_logit(log_switch_hat_n));
    log_switch_rep[n] = bernoulli_rng(inv_logit(log_switch_hat_n));
  }
}
