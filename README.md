

# Measuring Generalization Impact of Depth and Width on Over-Parameterized Neural Networks maintaining VC Dimension

## Abstract
One theory that explains how randomly initialized first-order methods achieve zero training loss despite having a non-convex and non-smooth objective function is Neural Network Overparameterization. However, previous work attempting to substantiate this theory require extreme overparameterization -- with parameter count exceeding training data size by an exponential factor. Additionally, the impact to convergence rate and generalizability of these overparameterized networks are not well explored, especially when varying depth and width while controlling for network capacity. In this paper, we investigate the trade-offs between network depth and width on convergence rate and test data loss when controlling for network capacity (reflected through VC-dimension bounds).
First, we generate a synthetic data set, $S_1$, imposing strong assumptions, to train a two-layer neural network with ReLU activation and sigmoidal output. This trained neural network witho zero training loss is then used as our control, from which we derive the VC-Dimension bounds. Second, we propose three variants on the network architecture and prove that these alternative models maintain the same VC-Dimension bounds as our control, analyzing convergence rate and generalization of these variants as compared to our control. Third, we repeat the above process on a noisier and larger data set, $S_2$, deriving a new control network and repeating the process above, to analyze convergence rate and generalization of this second set of variant models with similar VC-Dimension bounds.

## 1. Introduction

Overparameterization has emerged as the prevalent theory justifying perfect training performance seen in randomly initialized first-order methods. The central idea of overparameterization is that the parameters in the model exceed the size of the training data set. Once overparameterized, the neural network can fit to any label configuration on the data set, including randomly generated labelling scenarios. Once trained in an overparameterized regime, every local minima the network finds is actually a global minima, though it is unclear why this is the case.

In the binary classification problem, with a set of inputs $x_i \in X$ with dimension $d$ and size $n$, mapped to labels $y_i \in Y = \{0,1\}$, a network $N$ with $\ell$ layers of width $m$ is overparameterized long as $m \cdot d > n$. 

Prior work makes assumptions on the degree of overparameterization relative to the size of the data set, $n$, and fairly stringent assumptions about the data set itself. First, prior work states that to achieve zero-training loss, the degree of overparameterization must be $m=\Omega(\frac{n^6}{\lambda^4_0\delta^3})$. While these assumptions are theoretically reasonable, they pose challenges when scaling with $n$. their impact on the perfect training performance seen by these networks is not well-explored experimentally.

Furthermore, experimental insights on how depth and width of the network impact the convergence rate and network generalization is limited, especially when network capacity remains constant. Network capacity in this context is reflected through the Vapnik-Chervonenkis Dimension (VC-Dimension) of network $N$, defined as the maximum size of set $S \in X$ that is shattered by $N$.

VC-dimension is a common characterization of sample complexity of infinite hypothesis classes relating to the binary classification problem. For a set of input data points, $X$ of size $n$, Sauer's Lemma states that a hypothesis class of functions, $\mathcal{H}$, for set $X$ is learnable if $\mathcal{H}$ has a finite VC-Dimension $d$. For a step/sign function, the VC dimension of $\mathcal{H}_\{V,E,sign\}$ is $O(|E|\log(|E|))$.

## 2. Related Work

Prior work has largely focused on either establishing or defining bounds on the VC-Dimension of neural networks or exploring the requirements for overparameterization to achieve zero training loss. In 2017, Bartlett et al. gives an upper bound on the VC dimension of fully connected neural networks with ReLu activation functions. According to their team, VC-dim=$O(WL\log(W))$ where $W$ is the number of weights and $L$ is the number of layers. 

The following year, Aurora et al. shows acceleration to global minima is a result of overparameterization in each layer, holding the number of weights constant while increasing depth, which yields greater network expressiveness but non-optimized convergence. 

In Du et al.’s 2019 work, the team attempts to provide mathematical proof for the overparameterization theory to explain the phenomena on two-layer, fully connected neural networks with ReLU activation. Beyond the two-layer case, Allen-Zhang et al. shows overparameterization across multiple layers enjoys the same results as the Du et al.'s two layer case. 

## 3. System Model and Problem Formulation
### A. Basic Notation
\begin{itemize}
    \setlength\itemsep{1em}
    \item $N_0$: Two-layer, fully connected neural network with ReLU activation and stochastic gradient descent optimization.
    \item $X \in \mathds{R}^d$: Input data set of size $n$
    \item $m_i$: Number of hidden nodes in layer $i$
    \item $M = \sum^l_{i=1}m_i$: Total number of hidden nodes in network $N$
    \item $w_{ri}$: Input weight vector for layer $i$ in network $N$ where $w_r \in \mathds{R}^d$
    \item $a_{ri}$: Output weight vector for layer $i$ in network $N$ where $w_r \in \mathds{R}^d$
    \item $\sigma(\cdot)$: ReLU Activation Function
    \item $SGD$: Stochastic Gradient Descent, defined as follows:
    \[ f(W,a,x)= \sum^n_{i=1}\frac{1}{\sqrt{M}} \sum^M_{r=1}a_r\sigma(w^\top_r x) \] 
\end{itemize}

### B. Problem Formulation

### C. Hypothesis
TODO

### D. Methodology
To create a data set for our experiments, we used Du et al.’s method to generate 1000 data points from a 1000-dimensional unit sphere and applied labels from a 1-dimensional standard Gaussian distribution. 

While Du et al. did achieve zero training loss, their networks were extremely overparameterized, with m>>n, precisely set as m=(n6043). Thus, to extend Du et al.’s findings while improving upon the overall scalability and ease of experimentation of the neural network, we will first define a control 2-layer network that achieves zero training loss but is overparameterized to a more reasonable degree, starting with m > n, where the number of hidden nodes is only slightly larger than the size of the training data set and then moving to m = cn where the number of hidden nodes is larger than the training corpus size by a multiplier. We trained our model by running 100 epochs of gradient descent and used a fixed step size.

Once we define this control neural network architecture and it’s accompanying VC dimension bounds, we will then select a set of increasing VC dimension values that, for 2 layer networks, would correspond to a set of widths that approach the theoretical bound needed for over parameterization shown in [1].

## 4. Results
4.1 Data Collection 

4.2 Analysis I

4.3 Analysis II

4.4 Analysis III

4.5 Evaluation Criteria

## References
[1]  S.  Du,  X.  Zhai,  B.  Poczos  and  A.  Singh,  ”Gradient  Descent  Provably Optimizes Over-Parameterized Neural Networks”, Conference Paper at International  Conference  on  Learning  Representations  2019,  February 2019.
[2]  P.  Bartlett,  N.  Harvey,  C.  Liaw,  and  A.  Mehrabian,  ”Nearly-tight  VC-dimension  and  pseudodimension  bounds  for  piecewise  Linear  NNs”, Conference  Paper  at  Conference  on  Learning  Theory  2017,  October 2017.
[3] 

