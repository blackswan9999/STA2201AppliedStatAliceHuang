data {
  int<lower=0> N; // number of years
  int<lower=0> S; // number of states
  matrix[N, S] y; // log entries per captia
  int<lower=0> K; // number of splines
  matrix[N, K] B; //splines 
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  matrix[K,S] alpha;
  vector<lower=0>[S] sigma_alpha;
  vector<lower=0>[S] sigma;
}

transformed parameters{
  matrix[N,S] mu;
  for(i in 1:N){
    for(s in 1:S){
      mu[i,s] = B[i,]*alpha[,s];
    }
  }
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  for(s in 1:S){
    y[,s] ~ normal(mu[,s], sigma_y[s]);
    alpha[1,s] ~ normal(0, sigma_alpha);
    alpha[2,s] ~ normal(alpha[1,s], sigma_alpha[s]);
  }
  
}

