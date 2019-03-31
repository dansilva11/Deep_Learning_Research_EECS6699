

# Measuring Generalization Impact of Depth and Width on Over-Parameterized Neural Networks maintaining VC Dimension

## Abstract
Randomly initialized first-order methods achieve zero training loss despite having a non-convex and non-smooth objective function. One explanation of this phenomena is that the neural network is over-parameterized, with previous instances using 100x parameters than the number of training data. Extending this architectural approach, one theory speculates that over-parametrization of a Neural Network relative to the data set enables the network to fit all training data, though this does not guarantee zero testing loss. In this paper, we investigate the trade-offs between depth and width on test data loss when controlling for network capacity (VC dimension).

## 1. Introduction

A. Success of Randomly Initialized Gradient Descent

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

