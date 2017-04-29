# RUT
Resampled U-Test for non-parametric comparison of two or more distributions with unequal sampling and variance
 
# Installation

```
cd Rut
python3 setup.py install
```

If sklearn installation causes problems (they have odd naming conventions for their packages), first install sklearn from git:

`pip3 install git+git://github.com/scikit-learn/scikit-learn.git@master`

# usage

```
> ipython  # use ipython environment for interactive documentation
> import Rut
> ?rut.mannwhitneyu
> ?rut.kruskalwallis
> ?rut.cluster
```

# memory requirements
In almost all cases the default parameters should work on a machine with 32gb RAM. I am in the process of reducing the RAM usage by leveraging shared memory modules. The memory usage (and run time) scales with the both the number of cells and genes measured, since I designed this test to process all genes at once in each sample. The former (cells) is by default restricted to be 500 or less by the max_obs_per_sample parameter. I don't suggest increasing this.  

pre-running the kruskal-wallis ANOVA across all your groups is a good way to reduce the memory overhead of the following mann-whitney tests (I typically throw out any gene with q > 0.1). 

In addition to the DE tests and the clustering method, there is also an in-development module `score_features` that will return gene set scores (expression sums) across normalized samples from two or more sets of cells. 
