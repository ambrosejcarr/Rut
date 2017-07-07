# RUT
Sampling framework to enable comparison of means or medians of two samples whose characteristics break IID assumptions required for existing methods to generate valid results. Currently implements several such tests:

1. Mann-Whitney U
2. Wilcoxon Behrens Fisher
3. Welch's T
4. KruskalWallis ANOVA

In addition, the same method can be used to generate a distance matrix between clusters whose sampling rates differ. This package implements a method of calculating a distance matrix which can be used with the Phenograph clustering method.  
 
# Installation

```
cd Rut
python3 setup.py install
```

If sklearn installation causes problems, first install sklearn from git:

`pip3 install git+git://github.com/scikit-learn/scikit-learn.git@master`

# Usage and Documentation

```
> ipython  # use ipython environment for interactive documentation
> import rut
> ?rut.differential_expression.MannWhitneyU
> ?rut.differential_expression.KruskalWallis
> ?rut.differential_expression.WilcoxonBF
> ?rut.differential_expression.WelchsT
> ?rut.cluster
```
