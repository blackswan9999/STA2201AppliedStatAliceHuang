data {
  int<lower=0> N; // number of age categories
  int<lower=0> M; // number of year categories
  int deaths[N,M];
  matrix[N,M] age;
  matrix[N,M] pop;
}

parameters {
  vector<lower=0>[M] alpha;
  vector<lower=0>[M] beta;
}

transformed parameters{
  // matrix containing mortality*pop estimates, each column will contain estimates for one year
  matrix[N,M] mort;
  // loop through years
  for(m in 1:M){
    // make a column of mortality*pop estimates for year m
      mort[:,m] = log(alpha[m]) + beta[m]*age[:,m] + log(pop[:,m]);
    }
}

model {
  // update priors for each year
  for(m in 1:M){
    target += normal_lpdf(alpha[m]| 0.02, 0.01);
    target += normal_lpdf(beta[m]|0.0025, 0.001);
  }
  
  for(n in 1:N){
    for(m in 1:M){
      target += poisson_log_lpmf(deaths[n,m]|mort[n,m]);
    }
  }
}
