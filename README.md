# Heteroscedasticity_CNN
##This is a project for predicting heteroscedasticity data by using Convolutional Neural Network.

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
More precisely, I would use 
