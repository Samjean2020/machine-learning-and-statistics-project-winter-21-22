# machine-learning-and-statistics-project-winter-21-22
![scipyshiny_small](https://user-images.githubusercontent.com/60227439/147887960-d90daf7c-c330-4674-a8f7-7bb5e1939e6d.png)

![scikit-learn-logo-small](https://user-images.githubusercontent.com/60227439/147887966-a7226630-a2ad-4a01-b938-5e4ac3e524aa.png)

## Project Title

### Machine Learning and Statistics Assignment

For the purpose of HDip. in Science in Computing (Data Analytics) 2021-2022 at the Atlantic Technological University, I am required to submit a project on Machine Learning and Statistics as per the Python Programming Language.

Machine Learning makes the computer learn from studying data and statistics. It is a step towards artificial intelligence (AI).  It entails a program that analyses and learns to predict the outcome (w3schools.com).

https://www.w3schools.com/python/python_ml_getting_started.asp
 
## Description
This repository contains Jupyter notebooks and other relevant files demonstrating my work on the Python packages ‘scikit-learn’ and ‘sciPy-stats’ for the modules Machine Learning and Statistics.

#### The Scikit-learn Jupyter Notebook

The Scikit-learn Jupyter Notebook includes a file called scikit-learn.ipynb that contains:

	An overview of scikit an overview of the scikit-learn Python library including classification, regression, and clustering functionalities.

	Scikit-learn algorithms on Classification include K-Nearest Neighbours Classifiers, Cross-validation, and Logistic Regression Path. 
	Plots and other visualisations related to the classification functionality are also included in this section because of practical reasons.

	Scikit-learn algorithms on Regression include Linear regression, Ridge regression, and Lasso regression.
	Scikit-learn algorithms on Clustering encompass, The Mean-shift, Spectral clustering, K-Means, Hierarchical clustering, and Agglomerative clustering.
 
#### The Scipy Stats Jupyter Notebook

The Scipy Stats Jupyter Notebook includes a Jupyter notebook called scipy-stats.ipynb that encompasses the following:

•	An overview of the scipy.stats Python library focusing on probability distributions, including  continuous distributions, multivariate, and discrete distributions”.
•	An example hypothesis test using ANOVA  calibrated on the Breast Cancer Wisconsin (Diagnostics) Data Set available at https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29 , and perform the following computational operations, including Assumption1: Dependent and independent variables, Assumption 4: Outliers, Assumption 5: Normality using sns.displot( ) function, Shapiro-wilk test as ss.shaprio() function. Assumption 6: Homogeneity of Variances using scipy.stats.Levene() function and Unequal variances using Welch_anova_np () function.
•	Plots and other visualisations including chiefly those related to seaborn.boxplot() function that executes the following operations: Draw a vertical boxplot grouped by a categorical variable, Draw a boxplot with nested grouping by two categorical variables, Draw a boxplot with nested grouping when some bins are empty, Control box order by passing an explicit order, Use hue without changing box position or width, Use swarmplot() to show the datapoints on top of the boxes and Use catplot() to combine a boxplot() and a FacetGrid.

## Install
                                                 
Steps to install scikit-learn.ipynb and scipy-stats.ipynb programs written on Jupyter Notebooks.
1.	Set up a good Python environment by downloading the Python Programming Language and its various libraries from Anaconda 3. 
2.	Get more code, and text editors such as the Cmder (Console Emulator), VSC ( Visual Studio Code), Notepad++, etc.
3.	Run commands.

## Run
Here is how to run the project notebooks:
1.	Open terminal.
2.	Run command.
3.	Go to browser.

## Explore

Have a look at the 2 notebooks in this repository in Jupyter. 
Some interesting aspects:
The notebook scipy-learn.ipynb has various plot types as examples. You can edit the parameters of the plots to see different effects.

Find a script for Ridge Regression below:

Set up (this is a comment and # must be placed in front it).
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

X is the 10x10 Hilbert matrix (this is a comment and # must be placed in front it).
X = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

Compute paths (this is a comment and # must be placed in front it).
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

Display results (this is a comment and # must be placed in front it).

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")

Reverse axis (this is a comment and # must be placed in front it).

ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization")
plt.axis("tight")
plt.show(

When you run the above code an output of Ridge Regression plot will be displayed. Ridge Regression is used as the estimator in the above code output. Each color represents a distinctive feature of the coefficient vector and this is shown as a function of the regularisation characteristic (scikit-learn.org). 

## Credits
                                              
I used extensive online sources to build-up this repository. References including in each Jupyter Notebook are useful and can be consulted if there is a need to go further on Machine Learning and Statistics, particularly with Scikit-Learn and SciPy-Stats. I have also used Laerd Statistics (SPSS) for Assumptions on scipy-stats.ipynb program, especially on an example hypothesis test using ANOVA part. Their link address: https://statistics.laerd.com/spss-tutorials/one-way-anova-using-spss-statistics.php. You can also access https://statistics.laerd.com/ for SPSS Statistics Tutorials and Statistical Guides.




