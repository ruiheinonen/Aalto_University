data {
        // train
        int <lower=0> N; // number of data points
        vector[N] y;  // price
        vector[N] x1; // sqft_living
        vector[N] x2; // grade
        
        // test
        int <lower=0> N_test; // number of data points
        vector[N_test] x1_test; // sqft_living
        vector[N_test] x2_test; // grade
}
parameters {
        real alpha; // intercept
        vector[2] beta; // coefficients
        real <lower=0> sigma; 
}
model {
        y ~ normal(alpha + beta[1]*x1 + beta[2]*x2, sigma);
        alpha ~ normal(10, 10);
        beta[1] ~ normal(0, 1);
        beta[2] ~ normal(0, 1);
        sigma ~ normal(0, 100);
}
generated quantities {
        vector[N] log_lik;
        vector[N_test] y_preds;
        for (i in 1:N_test){
            y_preds[i] = normal_rng(alpha + beta[1]*x1_test[i] + beta[2]*x2_test[i], sigma);
        }
        for (i in 1:N){
            log_lik[i] = normal_lpdf(y[i] | alpha + beta[1]*x1[i] + beta[2]*x2[i], sigma);
        }
}