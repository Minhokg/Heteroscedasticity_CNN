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


But we can't 

$$
\begin{align*}
y & =sin4x\times sin5x+\epsilon\\
x\in[0,\frac{1}{2}\pi] & ,\epsilon\sim N(0,0.02+0.02\times(1-sin4x)^{2})
\end{align*}
$$
