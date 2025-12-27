---
layout: post
title:  Convolution and Fourier Transform on Graph
categories: Mathematics
tags: AI ML Mathematics
share: True
comments: True
---

### 1. The Classical Foundation: Fourier and Convolution
In classical signal processing on $$\mathbb{R}^n$$, the **Fourier Transform** decomposes a signal into orthonormal basis functions. For example, consider a discrete signal $$x$$ of length $$N=4$$. This signal lives in $$\mathbb{R}^4$$.

$$ x = \begin{bmatrix} 1 \\ 1 \\ 0 \\ 0 \end{bmatrix} $$

In the **Time Domain**, we represent this using the standard basis (Dirac deltas) $$e_0, \dots, e_3$$:

$$ x = 1 \cdot \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + 1 \cdot \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} + 0 \cdot \dots $$

This basis is orthonormal, but tells us nothing about the global smoothness or periodicity of the signal.

#### 1.2. The Fourier Basis (The Eigenbasis)
The Fourier Transform proposes a different basis: $$\{u_0, u_1, u_2, u_3\}$$. In classical signal processing, these basis vectors are sampled complex exponentials.

For $$N=4$$, the basis vector $$u_k$$ corresponding to frequency $$k$$ is:

$$ u_k(n) = \frac{1}{\sqrt{N}} e^{i \frac{2\pi}{N} k n} $$

Let's write them out explicitly:

*   **$$u_0$$ (DC component / Zero frequency):**
    $$ u_0 = \frac{1}{2} [1, 1, 1, 1]^T $$
    (Constant, perfectly smooth).

*   **$$u_1$$ (Low frequency):**
    $$ u_1 = \frac{1}{2} [1, i, -1, -i]^T $$

*   **$$u_2$$ (High frequency / Nyquist):**
    $$ u_2 = \frac{1}{2} [1, -1, 1, -1]^T $$

*   **$$u_3$$ (Conjugate of $$u_1$$):**
    $$ u_3 = \frac{1}{2} [1, -i, -1, i]^T $$

##### Checking Orthonormality
These vectors form an orthonormal basis. For example, the inner product of $$u_0$$ and $$u_2$$:

$$ \langle u_0, u_2 \rangle = u_0^H u_2 = \frac{1}{4} (1\cdot1 + 1\cdot(-1) + 1\cdot1 + 1\cdot(-1)) = 0 $$

And the norm:

$$ \| u_0 \|^2 = \frac{1}{4} (1+1+1+1) = 1 $$

#### 1.3. The Decomposition (The Transform)
The Fourier Transform is simply the projection of the signal $$x$$ onto these basis vectors.
$$ \hat{x}_k = \langle x, u_k \rangle = u_k^H x $$

Calculating the coefficients for our signal $$x = [1, 1, 0, 0]^T$$:

1.  **$$\hat{x}_0$$ (DC):** $$\frac{1}{2} [1, 1, 1, 1] \cdot [1, 1, 0, 0]^T = \frac{1}{2}(2) = 1$$
2.  **$$\hat{x}_1$$:** $$\frac{1}{2} [1, -i, -1, i] \cdot [1, 1, 0, 0]^T = \frac{1}{2}(1 - i)$$
3.  **$$\hat{x}_2$$:** $$\frac{1}{2} [1, -1, 1, -1] \cdot [1, 1, 0, 0]^T = \frac{1}{2}(0) = 0$$
4.  **$$\hat{x}_3$$:** $$\frac{1}{2} [1, i, -1, -i] \cdot [1, 1, 0, 0]^T = \frac{1}{2}(1 + i)$$

The **Spectrum** of the signal is the vector $$\hat{x} = [1, \frac{1-i}{2}, 0, \frac{1+i}{2}]^T$$.

#### 2. The Laplacian Connection
In the classical 1D example above, the basis vectors $$u_k$$ are not arbitrary. They are the **eigenvectors of the 1D Laplacian operator**.

If you discretize the second derivative (Laplacian or $$2x_i - x_{i-1} - x_{i+1}$$) on a 1D grid with a ring graph, you get:

$$ L_{1D} = D - A = \begin{bmatrix} 2 & -1 & 0 & -1 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ -1 & 0 & -1 & 2 \end{bmatrix} $$

If you calculate the eigendecomposition of this specific matrix $$L_{1D}$$, **you get exactly the Fourier basis vectors $$u_0, \dots, u_3$$ defined above.**

Let $$A$$ be an $$N \times N$$ symmetric matrix (the operator) acting on vectors $$x \in \mathbb{R}^N$$.
If we write $$y = Ax$$ using the standard basis (Cartesian coordinates), $$A$$ is typically a **dense matrix**.
*   **Coupling:** To compute the output $$y_i$$, you need a linear combination of *all* input elements $$x_1 \dots x_N$$. The dimensions are "coupled."

However, if we find the eigenvectors $$U = [u_1, \dots, u_N]$$ such that $$A u_i = \lambda_i u_i$$, we can decompose $$A$$:

$$ A = U \Lambda U^T $$

where $$\Lambda$$ is a **diagonal matrix** of eigenvalues.

If we transform our input $$x$$ into the eigenbasis ($$\hat{x} = U^T x$$), the operation becomes:

$$ \hat{y} = \Lambda \hat{x} $$

$$ \hat{y}_i = \lambda_i \hat{x}_i $$

**Diagonalization means:** In this specific basis (the eigenbasis), the complex matrix operation becomes simple scalar multiplication. The dimensions are now **decoupled**.

#### 2.1. The Infinite Dimensional View (Operators)
Now, move to the Hilbert space $$L^2$$ (square-integrable functions).
*   **Vector:** A function $$f(t)$$.
*   **Operator:** The Laplacian $$\Delta = \frac{d^2}{dt^2}$$.

Just like a matrix, an operator acts on a basis. Let's see how the Laplacian acts on Fourier Basis.
Let our basis be complex exponentials: $$\{e^{i\omega t}\}$$.
Apply the Laplacian operator to a basis vector:

$$ \frac{d^2}{dt^2}(e^{i\omega t}) = -\omega^2 (e^{i\omega t}) $$

The operator returns the **same** basis vector, just scaled by a scalar ($$-\omega^2$$).
If you were to write this operator as a "matrix" (infinite dimensional), it would have zeros everywhere except on the diagonal. The diagonal entries would be the eigenvalues $$-\omega^2$$.

Why do we use complex exponentials $$e^{i \omega t}$$? Because they are the **eigenfunctions of the Laplace operator** $$\Delta = \nabla^2$$.  Thus, the Fourier modes are the eigenfunctions of the Laplacian, and the eigenvalues correspond to frequencies (squared). So, the operations become element-wise.

#### 2.3 The Convolution Theorem
The convolution of functions $$f$$ and $$g$$, denoted $$f * g$$, is defined as:

$$ (f * g)(t) = \int f(\tau)g(t-\tau) d\tau $$

Note that the **Convolution** is a linear combination of **Shifts**.

Computationally, this is equivalent to element-wise multiplication in the spectral domain:

$$ \mathcal{F}(f * g) = \mathcal{F}(f) \cdot \mathcal{F}(g) $$

$$ f * g = \mathcal{F}^{-1} \big( \mathcal{F}(f) \cdot \mathcal{F}(g) \big) $$

**The Problem:** On a graph, there is no canonical definition of "translation" ($$t - \tau$$), so we cannot apply the spatial definition of convolution directly. However, we *can* define the Laplacian operator on a graph. Therefore, we define graph convolution via the spectral domain. Here is how it works:
1.  **Convolution** is a linear combination of **Shifts**.
2.  The **Laplacian** is also a linear combination of **Shifts** (for eg, $$2x_i - x_{i-1} - x_{i+1}$$).
3.  Therefore, they **Commute** ($$AB = BA$$).
4.  Commuting matrices share the same **Eigenvectors**.
5.  Therefore, the eigenvectors of the Laplacian (the Fourier Basis) are also the eigenvectors of the Convolution matrix.
6.  Expressing an operator in its eigenbasis **diagonalizes** it.

### 3. Spectral Graph Theory

Let $$G = (V, E)$$ be an undirected graph with $$N$$ nodes.
*   **Adjacency Matrix:** $$A \in \mathbb{R}^{N \times N}$$.
*   **Degree Matrix:** $$D_{ii} = \sum_j A_{ij}$$.

#### 3.1 The Graph Laplacian
The combinatorial Laplacian is $$L = D - A$$.
For GNNs, we typically use the **Normalized Laplacian**:

$$ L_{sym} = I_N - D^{-1/2} A D^{-1/2} $$

Since $$L_{sym}$$ is real symmetric positive semi-definite, it admits an eigendecomposition:

$$ L_{sym} = U \Lambda U^T $$
*   $$U \in \mathbb{R}^{N \times N}$$ is the matrix of eigenvectors ($$u_1, \dots, u_N$$). These are the **Graph Fourier Modes**.
*   $$\Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$$ are the eigenvalues (frequencies), usually sorted $$0 = \lambda_1 \le \lambda_2 \dots \le \lambda_N$$. Low $$\lambda$$ indicates smooth signals; high $$\lambda$$ indicates rapidly changing signals across edges.

#### 3.2 Graph Fourier Transform (GFT)
For a signal $$x \in \mathbb{R}^N$$ (a scalar feature per node):
1.  **GFT:** $$\hat{x} = U^T x$$ (Projection onto the eigenbasis).
2.  **Inverse GFT:** $$x = U \hat{x}$$.

#### 3.3. Graph Convolution (Spectral Definition)

Using the Convolution Theorem, we define the convolution of a signal $$x$$ with a filter $$g$$ on a graph as:

$$ x \star_G g = \mathcal{F}^{-1} \big( \mathcal{F}(g) \odot \mathcal{F}(x) \big) $$
$$ x \star_G g = U ( \hat{g} \odot (U^T x) ) $$

where $$\odot$$ is the Hadamard product. While the Hadamard product is intuitive, linear algebra is easier to manipulate using standard matrix multiplication. If you have a vector $$\hat{g} = [\theta_1, \theta_2, \dots, \theta_N]^T$$ and you want to perform an element-wise multiplication with a vector $$\hat{x} = U^T x$$, this is mathematically equivalent to multiplying $$\hat{x}$$ by a **diagonal matrix** where the diagonal elements are the entries of $$\hat{g}$$. So, we can represent the filter as a diagonal matrix $$g_\theta(\Lambda) = \text{diag}(\theta_1, \dots, \theta_N)$$.

Thus, the fundamental **Spectral Graph Convolution** is:

$$ x \star_G g_\theta = U g_\theta(\Lambda) U^T x $$

**The Computational Bottleneck:**
1.  Eigendecomposition of $$L$$ is $$O(N^3)$$.
2.  Multiplication by $$U$$ is $$O(N^2)$$.
This is prohibitively expensive for large graphs and does not generalize across graphs (since $$U$$ is specific to a specific graph structure).

### 4. Bridge to Spatial: Polynomial Approximations

To make this efficient and spatially localized, we parametrize the filter $$g_\theta(\Lambda)$$ not as $$N$$ free parameters, but as a **polynomial function** of the eigenvalues.

Let $$g_\theta(\Lambda) = \sum_{k=0}^K \theta_k \Lambda^k$$.
Then the convolution becomes:

$$ x \star_G g_\theta = U \left( \sum_{k=0}^K \theta_k \Lambda^k \right) U^T x = \sum_{k=0}^K \theta_k (U \Lambda^k U^T) x = \sum_{k=0}^K \theta_k L^k x $$

**This is the critical result.** We no longer need $$U$$. We can compute convolution directly using powers of the Laplacian $$L$$.

#### Why is $$L^k$$ spatially localized?
This relates to the concept of $$K$$-hop neighborhoods.
*   $$L$$ (and $$A$$) describes 1-hop connectivity.
*   $$(A^k)_{ij} \neq 0$$ if and only if there is a path of length $$k$$ between node $$i$$ and node $$j$$.
*   Therefore, $$(L^k x)_i$$ depends only on the values of $$x$$ at nodes within $$K$$ hops of node $$i$$.

This brings us to **ChebNet** [[Defferrard et al., 2016](https://arxiv.org/pdf/1606.09375)], which approximates the filter using Chebyshev polynomials $$T_k(x)$$ for numerical stability:

$$ g_\theta \star x \approx \sum_{k=0}^K \theta_k T_k(\tilde{L}) x $$

This is strictly a spectral convolution, but implemented via spatial aggregation of $$K$$-hop neighbors.

### 5. GCN: A First-Order Approximation
The standard Graph Convolutional Network [[Kipf & Welling, 2017](https://arxiv.org/pdf/1609.02907)] is derived by simplifying ChebNet.
Assume $$K=1$$ (1-hop filters) and constrain parameters such that $$\theta = \theta_0 = -\theta_1$$.
Using the "renormalization trick" (replacing $$I + D^{-1/2}AD^{-1/2}$$ with a normalized adjacency $$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$$), the layer-wise propagation becomes:

$$ H^{(l+1)} = \sigma \left( \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H^{(l)} W^{(l)} \right) $$

Here, $$\tilde{A}$$ acts as a spatial averaging operator. The spectral derivation has collapsed into a spatial weighted average.

### 6. Message Passing vs. Spectral Convolution

#### Message Passing Neural Networks (MPNN)
[[Gilmer et al., 2017](https://arxiv.org/pdf/1704.01212)] formalized the spatial view. A generic MPNN layer updates node features $$h_u$$ as:
1.  **Message:** $$m_{uv} = M(h_u, h_v, e_{uv})$$
2.  **Aggregation:** $$m_u = \text{AGG}_{v \in \mathcal{N}(u)} (m_{uv})$$
3.  **Update:** $$h_u' = \text{UPDATE}(h_u, m_u)$$

#### The Equivalence and the Gap
**Is one a special case of the other?**

**Spectral Convolution $$\subset$$ Message Passing:**
Yes. Any polynomial spectral filter of order $$K$$ (like ChebNet) can be implemented as a Message Passing scheme.
*   Since $$L^K x$$ aggregates information from $$K$$-hops, it can be unrolled into $$K$$ layers of 1-hop message passing where the aggregation function is a specific linear combination (weighted sum) defined by the Laplacian entries.
*   Standard GCN is exactly an MPNN where the message is $$h_v$$ scaled by a normalization constant, and aggregation is a sum.

**Message Passing $$\not\subset$$ Spectral Convolution (in the strict linear sense):**
General Message Passing is strictly more expressive than standard Spectral Convolution.
*   **Non-linearity:** Spectral convolution implies a linear operation on the signal $$x$$ followed by a point-wise non-linearity ($$\sigma(g \star x)$$). MPNNs can apply non-linearities *during* the aggregation (e.g., Graph Attention Networks - GAT [[Veličković, P., et al., 2018](https://arxiv.org/pdf/1710.10903)]. In GAT, the adjacency weights $$A_{ij}$$ are dynamic functions of the features, making the "Laplacian" data-dependent and time-varying.
*   **Isotropism:** Standard spectral filters (polynomials of $$L$$) are **isotropic**. The weights depend on graph structure (distance), not on direction or specific neighbor identity, unless features are incorporated into edge weights. MPNNs can learn anisotropic kernels (treating different neighbors differently based on features).


### References
1.  [Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (ChebNet). *NeurIPS*.](https://arxiv.org/pdf/1606.09375)
2.  [Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.](https://arxiv.org/pdf/1609.02907)
3.  [Gilmer, J., et al. (2017). Neural Message Passing for Quantum Chemistry. *ICML*.](https://arxiv.org/pdf/1704.01212)
4.  [Veličković, P., et al. (2018). Graph Attention Networks. *ICLR*.](https://arxiv.org/pdf/1710.10903)
