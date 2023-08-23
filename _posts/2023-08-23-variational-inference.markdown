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
\begin{aligned}
\boldsymbol{p_{\theta}(x)} &= \int\boldsymbol{p_{\theta}(x,z)}d\boldsymbol{z} \\
                           &= \int\boldsymbol{p_{\theta}(z)}\boldsymbol{p_{\theta}(x|z)} d\boldsymbol{z}\\
                           &= \mathbb{E}_{\boldsymbol{z}\thicksim\boldsymbol{p_{\theta}(z)}}\boldsymbol{[p_{\theta}(x|z)]} \nonumber
\end{aligned}
$$

# Maximum Likelihood Estimation in LVMs
We want to maximize the log-likelihood of a single data point $$\boldsymbol{x}$$ with respect to the parameters $$\boldsymbol{\theta}$$:

$$
\begin{aligned}
\underset{\boldsymbol{\theta}}{\max}\log\boldsymbol{p_{\theta}(x)} &= \underset{\boldsymbol{\theta}}{\max}\log(\int \boldsymbol{p_{\theta}(x,z) d\boldsymbol{z}})\\
    & = \underset{\boldsymbol{\theta}}{\max}\underbrace{\log(\int \boldsymbol{p_{\theta}(z)}\boldsymbol{p_{\theta}(x|z)} d\boldsymbol{z})}_{f(\boldsymbol{\theta})}\\
    & = \underset{\boldsymbol{\theta}}{\max}f(\boldsymbol{\theta}) \nonumber
\end{aligned}
$$

Usually, the integral $$\int\boldsymbol{p_{\theta}(x, z)}d\boldsymbol{z}$$ is intractable to compute. That is, we can't even evaluate the function $$f(\boldsymbol{\theta})$$ that we want to maximize. Thus we maximize the lower bound of $$f(\boldsymbol{\theta})$$ instead.

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

$$
\begin{aligned}
\underset{\boldsymbol{\theta}}{\max}f(\boldsymbol{\theta}) &\geq \underset{g  \in \mathcal{G}}{\max}\underset{\boldsymbol{\theta}}{\max} g(\boldsymbol{\theta})\\
 & = \underset{g  \in \mathcal{G}, \boldsymbol{\theta}}{\max} g(\boldsymbol{\theta}) \nonumber
\end{aligned}
$$

This is called variational optimization, because we are optimizing over the functions($$g$$) themselves.

# Evidence Lower Bound (ELBO)
We need to find a lower bound for $$\log\boldsymbol{p_{\theta}(x)}$$. Let $$\boldsymbol{q(z)}$$ be an arbitraty distribution over $$\boldsymbol{z}$$. Then, we can write

$$
\begin{aligned}
\log\boldsymbol{p_{\theta}(x)} &= \mathbb{E}_{\boldsymbol{z} \thicksim \boldsymbol{q(z)}}[\log\boldsymbol{p_{\theta}(x)}] && (\boldsymbol{p_{\theta}(x)} \text{ is independent of $z$})\\
    &= \int \boldsymbol{q(z)} \log\boldsymbol{p_{\theta}(x)} d\boldsymbol{z}\\
    &= \int \boldsymbol{q(z)} \log\frac{\boldsymbol{p_{\theta}(x,z)}}{\boldsymbol{p_{\theta}(z|x)}} d\boldsymbol{z}\\
    &= \int \boldsymbol{q(z)} \log(\frac{\boldsymbol{p_{\theta}(x,z)}\boldsymbol{q(z)}}{\boldsymbol{p_{\theta}(z|x)}\boldsymbol{q(z)}}) d\boldsymbol{z}\\
    &= \int \boldsymbol{q(z)} \log\frac{\boldsymbol{p_{\theta}(x,z)}}{\boldsymbol{q(z)}} d\boldsymbol{z} + \int \boldsymbol{q(z)} \log\frac{\boldsymbol{q(z)}}{\boldsymbol{p_{\theta}(z|x)}} d\boldsymbol{z}\\
    &= \mathbb{E}_{\boldsymbol{z} \thicksim \boldsymbol{q(z)}}[\log\frac{\boldsymbol{p_{\theta}(x,z)}}{\boldsymbol{q(z)}}] + \mathbb{KL}(\boldsymbol{q(z)}||\boldsymbol{p_{\theta}(z|x)}) && (\mathbb{KL}(q(z)||p(z)) \overset{\Delta}{=} \int \boldsymbol{q(z)} \log\frac{\boldsymbol{q(z)}}{\boldsymbol{p(z)}} d\boldsymbol{z})
\end{aligned}
$$
# ELBO as a lower bound
Observe that, $$\mathbb{KL}(q(z)||p(z)) \geq 0$$. We have $$\mathbb{KL}(q(z)||p(z)) = 0$$ if and only if $$\boldsymbol{q(z)} = \boldsymbol{p(z)}$$.

Therefore, we have
$$
\log\boldsymbol{p_{\theta}(x)} =  \underbrace{\mathbb{E}_{\boldsymbol{z} \thicksim \boldsymbol{q(z)}}[\log\frac{\boldsymbol{p_{\theta}(x,z)}}{\boldsymbol{q(z)}}]}_{ELBO = \mathcal{L}(\theta, q)} + \underbrace{\mathbb{KL}(\boldsymbol{q(z)}||\boldsymbol{p_{\theta}(z|x)})}_{\geq 0}
$$

Since, $$\mathbb{KL}(q(z)||p(z)) \geq 0$$, we have 
\begin{equation} \label{eq:elbo}
\log\boldsymbol{p_{\theta}(x)} \geq \mathcal{L}(\theta, q)
\end{equation}
<br>
Thus, $$\mathcal{L}(\theta, q)$$ is a lower bound on $$\log\boldsymbol{p_{\theta}(x)}. (\ref{eq:elbo})$$ is tight if and only if $$\boldsymbol{q(z)} = \boldsymbol{p_{\theta}(z|x)}$$.

# References
