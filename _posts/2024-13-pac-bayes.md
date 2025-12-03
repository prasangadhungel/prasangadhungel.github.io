---
layout: post
title:  PAC-Bayes Bound?
categories: Mathematics
tags: AI ML Mathematics
share: True
comments: True
---

### **The Supervised Learning Setup**

Let $\mathcal{X}$ denote the input space (e.g., $\mathbb{R}^d$) and $\mathcal{Y}$ denote the label space (e.g., $\{-1, +1\}$ for binary classification). We observe a training dataset $\mathcal{S} = \{(x_i, y_i)\}_{i=1}^{m}$, where each sample is drawn independently and identically distributed (i.i.d.) from an **unknown** joint probability distribution $\mathcal{D}$ over $\mathcal{X} \times \mathcal{Y}$.

Formally, a learning algorithm $\mathcal{A}$ is a map from a dataset to a hypothesis. If we let $\mathcal{Z} = \mathcal{X} \times \mathcal{Y}$, the algorithm is defined as:
$$
\mathcal{A}: \bigcup_{m=1}^\infty \mathcal{Z}^m \to \mathcal{H}
$$
where $\mathcal{H} \subseteq \mathcal{Y}^\mathcal{X}$ is the hypothesis class (the set of functions $h: \mathcal{X} \to \mathcal{Y}$ considered by the algorithm).

### **Risk and The Learning Objective**

To measure performance, we define a loss function $\ell: \mathcal{Y} \times \mathcal{Y} \to [0, B]$. Typically, we assume the loss is bounded (e.g., $B=1$ for the 0-1 loss). The quantity $\ell(h(x), y)$ represents the penalty for predicting $h(x)$ when the true label is $y$.

Our goal is to find a hypothesis that minimizes the **true risk** (or generalization error), defined as the expected loss over $\mathcal{D}$:
$$
L_\mathcal{D}(h) \triangleq \mathbb{E}_{(x, y) \sim \mathcal{D}}[\ell(h(x), y)]
$$

The theoretical limit of performance is achieved by the **Bayes optimal predictor** $h^*$, which minimizes the risk over the set of all measurable functions $\mathcal{Y}^\mathcal{X}$:
$$
h^*(x) = \underset{\hat{y} \in \mathcal{Y}}{\mathrm{argmin}} \ \mathbb{E} [\ell(\hat{y}, y) \mid X=x]
$$
The risk of this predictor, $L^* = L_\mathcal{D}(h^*)$, is called the **Bayes error** (or irreducible error).

### **From True Risk to Empirical Risk**

Since $\mathcal{D}$ is unknown, $L_\mathcal{D}(h)$ is uncomputable. We rely on the sample $\mathcal{S}$ to compute the **empirical risk** (training error):
$$
L_\mathcal{S}(h) = \frac{1}{m}\sum_{i=1}^{m}\ell(h(x_i), y_i)
$$

A standard approach, **Empirical Risk Minimization (ERM)**, searches for a predictor that minimizes training error. However, searching over the entire space of functions $\mathcal{Y}^\mathcal{X}$ is ill-posed; it is trivial to find a function $h$ such that $L_\mathcal{S}(h)=0$ (memorization) while $L_\mathcal{D}(h)$ remains high.

### **The Need for Inductive Bias**

To enable generalization, we restrict our search to a specific **hypothesis class** $\mathcal{H} \subset \mathcal{Y}^\mathcal{X}$. The choice of $\mathcal{H}$ (e.g., linear half-spaces, neural networks of depth $k$) imposes an **inductive bias**.

The **No Free Lunch Theorem** implies that without inductive bias, for any learning algorithm, there exists a distribution $\mathcal{D}$ on which the algorithm fails.

Let $h^*_\mathcal{H}$ denote the risk-minimizer within our class:
$$
h^*_\mathcal{H} \in \underset{h \in \mathcal{H}}{\mathrm{argmin}} \ L_\mathcal{D}(h)
$$
Our learning algorithm returns the ERM solution restricted to this class:
$$
\hat{h}_{S} \in \underset{h \in \mathcal{H}}{\mathrm{argmin}} \ L_\mathcal{S}(h)
$$

### **Error Decomposition**

We analyze the **excess risk** of our learned predictor compared to the Bayes optimal. We decompose this error into two components:
$$ 
L_\mathcal{D}(\hat{h}_{S}) -  L^* = \underbrace{(L_\mathcal{D}(\hat{h}_{S}) - L_\mathcal{D}(h^*_\mathcal{H}))}_{\text{Estimation Error}} + \underbrace{(L_\mathcal{D}(h^*_\mathcal{H}) - L^*)}_{\text{Approximation Error}}
$$

1.  **Approximation Error:** The penalty for restricting our search to $\mathcal{H}$. This is a property of the class $\mathcal{H}$ and cannot be reduced by collecting more data.
2.  **Estimation Error:** The penalty incurred because we minimize $L_\mathcal{S}$ rather than $L_\mathcal{D}$. This term is a random variable depending on the sample $\mathcal{S}$ and typically scales with the complexity of $\mathcal{H}$.

This highlights the **bias-complexity tradeoff**: as the complexity of $\mathcal{H}$ increases, approximation error decreases, but estimation error increases.

### **Generalization Bounds via Uniform Convergence**

To bound the estimation error, we analyze the worst-case difference between empirical and true risk over $\mathcal{H}$.

**1. Finite Hypothesis Class**
If $|\mathcal{H}| < \infty$ and loss is bounded in $[0,1]$, Hoeffding's inequality and the union bound guarantee that with probability at least $1 - \delta$ over the choice of $\mathcal{S}$:
$$
L_\mathcal{D}(\hat{h}_S) \leq L_\mathcal{S}(\hat{h}_S) + \sqrt{\frac{\ln|\mathcal{H}| + \ln(1/\delta)}{2m}}
$$

**2. Infinite Hypothesis Class**
For infinite classes, we replace $\ln|\mathcal{H}|$ with the **VC Dimension**, denoted $d_{\text{VC}}$. By the Vapnik-Chervonenkis theorem (via Sauer’s Lemma), if $d_{\text{VC}}(\mathcal{H}) = d < \infty$, then for $m \ge d$, with probability at least $1-\delta$:
$$
L_\mathcal{D}(\hat{h}_S) \leq L_\mathcal{S}(\hat{h}_S) + O\left(\sqrt{\frac{d \ln(m/d) + \ln(1/\delta)}{m}}\right)
$$

Crucially, learning is possible if the **Uniform Convergence** property holds:
$$ 
\sup_{h \in \mathcal{H}} |L_\mathcal{S}(h) - L_\mathcal{D}(h)| \xrightarrow{m \to \infty} 0 
$$

### **The Evolution of Learning Criteria**

**1. Probably Approximately Correct (PAC) Learning**
This framework, introduced by Valiant (1984), provides the standard definition of learnability. A class $\mathcal{H}$ is **PAC-learnable** if there exists an algorithm $\mathcal{A}$ and a sample complexity function $m_\mathcal{H}(\epsilon, \delta)$ such that for every $\epsilon, \delta \in (0, 1)$ and every distribution $\mathcal{D}$:

If $m \ge m_\mathcal{H}(\epsilon, \delta)$, then with probability at least $1-\delta$:
$$
L_\mathcal{D}(\hat{h}_S) \leq \inf_{h \in \mathcal{H}} L_\mathcal{D}(h) + \epsilon
$$

Standard bounds derived from VC theory usually take the form:
$$ 
L_\mathcal{D}(\hat{h}) \leq L_\mathcal{S}(\hat{h}) + \sqrt{\frac{C \cdot \text{complexity}(\mathcal{H})}{m}} 
$$
While rigorous, these bounds have limitations. The complexity term (VC-dimension or Rademacher complexity) is often a worst-case measure independent of the actual distribution or the specific hypothesis learned. In practice, deep networks with infinite VC dimension generalize well, suggesting these bounds are too loose to explain modern ML performance.

This motivates an alternative framework that incorporates prior knowledge and data-dependent posteriors.

---

### **PAC-Bayes Bounds**

PAC-Bayes theory (McAllester, 1999) bridges the gap between the frequentist PAC analysis and Bayesian inference. Instead of learning a single hypothesis $h$, we consider a learning algorithm that outputs a **distribution** $Q$ over the hypothesis class $\mathcal{H}$ (the posterior).

**The Setup**
1.  **Prior ($P$):** A distribution over $\mathcal{H}$ chosen *before* seeing the data. It encodes our preference for certain hypotheses (e.g., small weights).
2.  **Posterior ($Q$):** A distribution over $\mathcal{H}$ learned *after* observing $\mathcal{S}$. Note that $Q$ can be any distribution, not necessarily a Bayesian posterior update.

We define the risk of the randomized classifier (Gibbs classifier) as the expected risk under $Q$:
$$
L_\mathcal{D}(Q) \triangleq \mathbb{E}_{h \sim Q} [L_\mathcal{D}(h)]
$$
Similarly, the empirical risk is $L_\mathcal{S}(Q) \triangleq \mathbb{E}_{h \sim Q} [L_\mathcal{S}(h)]$.

**McAllester’s Bound**
For any prior $P$ independent of the data, and any loss bounded in $[0,1]$, with probability at least $1-\delta$ over the draw of $\mathcal{S}$, the following inequality holds for **all** posteriors $Q$ simultaneously:

$$
L_\mathcal{D}(Q) \leq L_\mathcal{S}(Q) + \sqrt{\frac{KL(Q \| P) + \ln(2m/\delta)}{2(m-1)}}
$$
where $KL(Q \| P)$ is the Kullback-Leibler divergence, measuring the "distance" between the learned posterior and the prior.

Naturally, we might wonder in what sense this is actually an improvement over uniform convergence. Well first it that we have lost the dependendcy on the class $\mathcal{H}$. Unlike VC dimension (a constant for the class), the complexity term here is $KL(Q\|P)$. If the algorithm finds a solution $Q$ close to the prior $P$, the bound is tight, even if the ambient space $\mathcal{H}$ is massive. For certain choices of $P$ and $Q$ (e.g., Gaussians), the KL term can be computed analytically, allowing us to explicitly optimize the bound.

This framework helps explain why overparameterized models generalize: even if the parameter space is large, the volume of solutions (the posterior $Q$) might be close to the initialization (prior $P$), resulting in a low effective complexity cost.

Optimization of the PAC-Bayes bound leads to objectives that trade off empirical error with a regularization term (the KL divergence), providing a rigorous justification for regularized training.
