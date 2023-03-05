# Heteroscedasticity for CNN
## This is a my project for predicting heteroscedasticity data by using Convolutional Neural Network.

Traditionally, to do a prediction like linear regression, we have to make some assumptions about an error term.
1. Independent and identically distributed (IID)
2. Normality
3. Homoscedasticity (constant variance)

For example, we can do a regression on below formula

$$
\begin{align*}
y & =sin4x\times sin5x+\epsilon\\
x & \in[0,\frac{1}{2}\pi],\epsilon\sim N(0,1)
\end{align*}
$$


But we can't do regression on the next heteroscedasticity data (changing variance) 

$$
\begin{align*}
y & =sin4x\times sin5x+\epsilon\\
x\in[0,\frac{1}{2}\pi], \epsilon &\sim N(0,0.02+0.02\times(1-sin4x)^{2})
\end{align*}
$$

$\epsilon$ is dependent on $x$ variable.

In this report, I'm going to do a prediction on the heteroscedasticity data by using Convolutional Neural Network.
More precisely, I would use two CNNs comprised of two different loss function: Mean Sqaure Error (MSE) and Negative Log Likelihood (NLL). Because error terms follow a Gaussian distribution, NLL also have to be set accordingly. Belows are two formuals each.

1. MSE = $\large \frac{1}{n_{obs}}\sum(y_{true}-y_{pred})^{2}$
2. NLL = $\large \frac{1}{2n_{obs}}\sum_{i}\frac{(y_{i,true}-y_{i,pred})^{2}}{\sigma_{i}^{2}}+\text{log}(\sigma_{i}^{2})$
