data {
  int<lower=1> n; // number of sites
  int<lower=1> t; // number of weeks
  int<lower=1> nobs;// total number of observed values
  int<lower=1> nmis; // total number of missing values
  real east[n]; // easting coordinates
  real north[n]; // northing coordinates
  vector<lower=0>[nobs] y; // vector of observed values
  vector[n] zerosn; // vector of zeros of dimension n
  int<lower=1> vectimeobs[nobs]; //vector of observed weeks at each position in the vector
  int<lower=1> vecsiteobs[nobs]; //vector of observed sites at each position in the vector
  int<lower=0> vectimemis[nmis]; //vector of missing weeks at each position in the vector
  int<lower=0> vecsitemis[nmis]; //vector of missing sites at each position in the vector
  int<lower=0> ncovs; // number of covariates
  vector[ncovs] covs[n]; // matrix of land-use variables
  vector[3] t_covs[t]; // matrix of climatic variables
  vector[t] tc_brightness[n]; // matrix of TC brightness
  vector[t] tc_greenness[n]; // matrix of TC greenness

}


parameters{
  real theta0; // initial value state vector
  vector [t] theta; //state vector
  real<lower=0, upper = 1> rho; // probability of zero
  vector[3] alpha; // coefficient of climatic factors
  real<lower=0> w_sd; // variance of state vector
  real<lower=0> tau; // variance
  vector<lower=0>[nmis] y_miss; // vector of missing values
  vector[2] betas; //coordinates coefficients
  vector[2] tc; // TC coefficients
  vector[ncovs] lur; // land use variables coefficients
}


transformed parameters{
  
  vector[nobs] mu; // vector of the mean for the observed values
  vector[nmis] mu_mis; //vector of the mean values for missing values
  
  real tau_sq = pow(tau,2);
  
  // computing the mean of the process for the observed values
  for(i in 1:nobs){
    mu[i] =  theta[vectimeobs[i]]  + 
    betas[1]*east[vecsiteobs[i]] + betas[2]*north[vecsiteobs[i]] + 
    lur'*covs[vecsiteobs[i]] + alpha'*t_covs[vectimeobs[i]] +
    tc[1]*tc_brightness[vecsiteobs[i], vectimeobs[i]] + 
    tc[2]*tc_greenness[vecsiteobs[i], vectimeobs[i]];
  }
  
  // computing the mean of the process for the missing values
  for(i in 1:nmis){
    mu_mis[i] = theta[vectimemis[i]]  +   
    betas[1]*east[vecsitemis[i]] + betas[2]*north[vecsitemis[i]] +
    lur'*covs[vecsitemis[i]] + alpha'*t_covs[vectimemis[i]]+ 
    tc[1]*tc_brightness[vecsitemis[i], vectimemis[i]] + 
    tc[2]*tc_greenness[vecsitemis[i], vectimemis[i]];
  }
  
  
} 

model{
  
  //state equations
  theta0 ~ normal(0,10);
  theta[1] ~ normal(theta0,w_sd);
  for(j in 2:t){
    theta[j] ~ normal(theta[j-1],w_sd);
  }
  
  betas ~ normal(0,10);
  tau ~ cauchy(0,1);
  w_sd ~ normal(0,1);
  rho ~ beta(1,1);
  
  for(i in 1:nobs){
    if(y[i] == 0)
    1 ~ bernoulli(rho);
    else{
      0 ~ bernoulli(rho);
      y[i] ~ lognormal(mu[i],tau);
    }
    
  }
  
  //model for the missing values
  for(i in 1:nmis){
    if(y_miss[i] == 0)
    1 ~ bernoulli(rho);
    else{
      0 ~ bernoulli(rho);
      y_miss[i] ~ lognormal(mu_mis[i],tau);
    }
  }
  
} 

generated quantities {
  vector[nobs] log_lik;
  real logit_rho = logit(rho);
  
  
  for (i in 1:nobs) {
    if(y[i] == 0) {
      log_lik[i] = bernoulli_logit_lpmf(1 | logit_rho);
    }else{
      log_lik[i] = lognormal_lpdf(y[i] | mu[i], tau);
    }      
  }
}
