# Estimating the Covariance Structure of a High-Dimensional Random Field like a Loss Function from Supervised ML

## Abstract

Estimating the Covariance structure of a Random Field is nothing new. But
many of these approaches stem from geostastics, where the dimension of the
Random Field is generally quite small (e.g. d=2). If we want
to view the loss function from a supervised machine learning problem as a 
random field (to apply random field informed gradient methods) we have to
estimate the covariance structure of a random field with very high input
dimension. As the input dimension is the number of parameters, it can easily
cross a million or more in this setting. This high dimensionality causes its
own set of problems, but also enables new approaches. We are going to address
some of these problems and make a first attempt to tackle this problem.