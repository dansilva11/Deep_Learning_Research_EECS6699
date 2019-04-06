

# Measuring Generalization Impact of Depth and Width on Over-Parameterized Neural Networks maintaining VC Dimension

## Abstract
One theory that explains how randomly initialized first-order methods achieve zero training loss despite having a non-convex and non-smooth objective function is Neural Network Overparameterization. However, previous work attempting to substantiate this theory require extreme overparameterization -- with parameter count exceeding training data size by an exponential factor. Additionally, the impact to convergence rate and generalizability of these overparameterized networks are not well explored, especially when varying depth and width while controlling for network capacity. In this paper, we investigate the trade-offs between network depth and width on convergence rate and test data loss when controlling for network capacity (reflected through VC-dimension bounds).
First, we generate a synthetic data set, $S_1$, imposing strong assumptions, to train a two-layer neural network with ReLU activation and sigmoidal output. This trained neural network witho zero training loss is then used as our control, from which we derive the VC-Dimension bounds. Second, we propose three variants on the network architecture and prove that these alternative models maintain the same VC-Dimension bounds as our control, analyzing convergence rate and generalization of these variants as compared to our control. Third, we repeat the above process on a noisier and larger data set, $S_2$, deriving a new control network and repeating the process above, to analyze convergence rate and generalization of this second set of variant models with similar VC-Dimension bounds.

## 1. Introduction

### A. Zero Training Loss in Over-Parameterized Neural Networks
Overparameterization has emerged as the prevalent theory justifying the perfect training performance seen from randomly initialized first-order methods. Prior work makes assumptions on the degree of overparameterization relative to the size of the data set, $n$, and fairly stringent assumptions about the data set itself. While these are theoretically reasonable, these assumptions are challenging to scale with $n$ and their impact on the perfect training performance seen by these networks is not well-explored experimentally.

Furthermore, experimental insights on how depth and width of the network impact the convergence rate and network generalization is limited, especially when network capacity remains constant. Network capacity in this context is reflected through the Vapnik-Chervonenkis Dimension (VC-Dimension) of network $\scriptN$, defined as the maximum size of set $\scriptS \in X$ that is shattered by $N$.

B. Overparameterization

C. Network Generalization

m → # of hidden nodes in neural network Rd
n → # of Training Data points

## 2. Related Work
## 3. Problem Formulation
## 4. Methodology
To create a data set for our experiments, we used Du et al.’s method to generate 1000 data points from a 1000-dimensional unit sphere and applied labels from a 1-dimensional standard Gaussian distribution. 
While Du et al. did achieve zero training loss, their networks were extremely overparameterized, with m>>n, precisely set as m=(n6043). Thus, to extend Du et al.’s findings while improving upon the overall scalability and ease of experimentation of the neural network, we will first define a control 2-layer network that achieves zero training loss but is overparameterized to a more reasonable degree, starting with m > n, where the number of hidden nodes is only slightly larger than the size of the training data set and then moving to m = cn where the number of hidden nodes is larger than the training corpus size by a multiplier. We trained our model by running 100 epochs of gradient descent and used a fixed step size.
Once we define this control neural network architecture and it’s accompanying VC dimension bounds, we will then select a set of increasing VC dimension values that, for 2 layer networks, would correspond to a set of widths that approach the theoretical bound needed for over parameterization shown in [1].

4.1 Data Collection 

4.2 Analysis I

4.3 Analysis II

4.4 Analysis III

4.5 Evaluation Criteria

## References
[1]  S.  Du,  X.  Zhai,  B.  Poczos  and  A.  Singh,  ”Gradient  Descent  Provably Optimizes Over-Parameterized Neural Networks”, Conference Paper at International  Conference  on  Learning  Representations  2019,  February 2019.
[2]  P.  Bartlett,  N.  Harvey,  C.  Liaw,  and  A.  Mehrabian,  ”Nearly-tight  VC-dimension  and  pseudodimension  bounds  for  piecewise  Linear  NNs”, Conference  Paper  at  Conference  on  Learning  Theory  2017,  October 2017.
[3] 

