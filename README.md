# Smooth isotonic, convex spline regressor.

This program shows how you can perform an arbitray isotonic, convex, smooth spline regression.
![Demo of isotonic concave regression](./demo.png?raw=true "Title")
or just a smooth spline regression 
![Demo of regression](./covid_demo.png?raw=true "Title")
where the best smoothing penalty is found by Bayesian Information Criterion.

I used python 3.6.7 with cvxopt as well as standard numpy, scipy and matplotlib packages to perform the analysis. cvxopt is pretty awesome!
You can install cvxopt as 
```
conda install -y -c conda-forge cvxopt
```

And you can see an example of the code in smooth_isotonic_regressor.ipynb
