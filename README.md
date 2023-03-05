# Heteroscedasticity_CNN
This is a project for predicting heteroscedasticity data by using CNN
$$
\begin{align}
    \Phi(0,x) = \max_{u \in \mathcal{D}} \bigg[
        \mathbb{E} & \Phi\left(1, 
        x + \int_0^1 \sigma^2(s) \, \zeta(s) \, u_s \, ds
        + \int_0^1 \sigma(s) \, dW_s
    \right) \\
        &- \frac{1}{2} \int_0^1 \sigma^2(s) \, \zeta(s) \,
        \mathbb{E} u_s^2  \, ds
    \bigg].
\end{align}
$$
