function yhat=logistic(beta,t)
yhat=beta(1)./(1+beta(2).*exp(-beta(3).*t));

