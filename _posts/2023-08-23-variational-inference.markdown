---
layout: post
title:  Variational Inference
categories: Mathematics
tags: AI ML Mathematics
share: True
comments: True
---

In generative modeling, we are interested in modeling $$\boldsymbol{p_{\theta}(x)}$$, where $$\boldsymbol{x}\in\mathbb{R}^{d}$$ is usually high dimensional. So, it is intractable to compute $$\boldsymbol{p_{\theta}(x)}$$ directly. Instead, we introduce a latent variable $$\boldsymbol{z}$$ which simplifies the likelihood estimation of $$\boldsymbol{x}$$. $$\boldsymbol{p_{\theta}(x)}$$ can be intractable to describe within itself, but if the latent variable can take only finitely many values, then we can describe $$\boldsymbol{p_{\theta}(x)}$$ as a marginal distribution of the joint distribution $$\boldsymbol{p_{\theta}(x,z)}$$ over all z.

In latent variable models (LVMs), data are generated using two-step process:
1. Sample the latent variable (prior) 
    \begin{equation}
    \boldsymbol{z} \thicksim \boldsymbol{p_{\theta}(z)} \nonumber
    \end{equation}
2. Generate the data conditioned on the latent variable (likelihood)
    \begin{equation}
    \boldsymbol{x} \thicksim \boldsymbol{p_{\theta}(x|z)} \nonumber
    \end{equation}

With the above procedure, the joint distribution can be defined as: 
\begin{equation}
\boldsymbol{p_{\theta}(x,z)} = \boldsymbol{p_{\theta}(z)}\boldsymbol{p_{\theta}(x|z)} \nonumber
\end{equation}

Therefore, the marginal distribution is:

$$
\boldsymbol{p_{\theta}(x)} = \int\boldsymbol{p_{\theta}(x,z)}dz = \int\boldsymbol{p_{\theta}(z)}\boldsymbol{p_{\theta}(x|z)} dz = \mathbb{E}_{\boldsymbol{z}\thicksim\boldsymbol{p_{\theta}(z)}}\boldsymbol{[p_{\theta}(x|z)]}
$$

# Maximum Likelihood Estimation in LVMs
We want to maximize the log-likelihood of a single data point $$\boldsymbol{x}$$ with respect to the parameters $$\boldsymbol{\theta}$$:

\begin{equation}
\underset{\boldsymbol{\theta}}{\max}\log\boldsymbol{p_{\theta}(x)} = \underset{\boldsymbol{\theta}}{\max}\log(\int \boldsymbol{p_{\theta}(x,z) dz}) = \underset{\boldsymbol{\theta}}{\max}\underbrace{\log(\int \boldsymbol{p_{\theta}(z)}\boldsymbol{p_{\theta}(x|z)} dz)}_{f(\boldsymbol{\theta})} = \underset{\boldsymbol{\theta}}{\max}f(\boldsymbol{\theta}) \nonumber
\end{equation}

Usually, the integral $$\int\boldsymbol{p_{\theta}(x, z)}dz$$ is intractable to compute. That is, we can't even evaluate the function $$f(\boldsymbol{\theta})$$ that we want to maximize. Thus we maximize the lower bound of $$f(\boldsymbol{\theta})$$ instead.

We find some $$g(\boldsymbol(\theta))$$ that is lower bound of $$f(\boldsymbol{\theta})$$. That is
\begin{equation}
f(\boldsymbol{\theta}) \geq g(\boldsymbol{\theta}), \quad\text{for all $\boldsymbol{\theta}$}. \nonumber
\end{equation}

Then, we maximize $$g(\boldsymbol{\theta})$$ instead of $$f(\boldsymbol{\theta})$$. Maximizing $$g(\boldsymbol{\theta})$$ would give us the lower bound on the solution of the original optimization problem.
\begin{equation}
\underset{\boldsymbol{\theta}}{\max}f(\boldsymbol{\theta}) \geq \underset{\boldsymbol{\theta}}{\max}g(\boldsymbol{\theta}) \nonumber
\end{equation}

We hope that $$\underset{\boldsymbol{\theta}}{\max}f(\boldsymbol{\theta})$$ is close to $$\underset{\boldsymbol{\theta}}{\max}g(\boldsymbol{\theta})$$. If so, why just limiting with only one lower bound $$g(\boldsymbol{\theta})$$? We instead consider a collection of lower bound
$$
\mathcal{G} = \{g(\boldsymbol{\theta})|g(\boldsymbol{\theta}) \text{is a lower bound on} f(\boldsymbol{\theta})\}
$$

# Multiple lower bounds
We maximize the lower bound that is tighest to $$f(\boldsymbol{\theta})$$.

\begin{equation}
\underset{\boldsymbol{\theta}}{\max}f(\boldsymbol{\theta}) \geq \underset{g  \in \mathcal{G}}{\max}\underset{\boldsymbol{\theta}}{\max} g(\boldsymbol{\theta}) \nonumber
\end{equation}

This is called variational optimization, because we are optimizing over the functions($$g$$) themselves.

# References
