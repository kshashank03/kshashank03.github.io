---
layout: post
title: Basic Structure Machine Learning/Deep Learning
---

## Define the Problem

First define the number of output variables (univariate, bivariate, multivariate), the machine learning paradigm of the problem, then the type of problem, then the algorithm(s) to be used:

**Supervised**

- Regression - Predicting a value
- Classification - Predict a class

**Unsupervised**

- Clustering - Group ungrouped data
- Association Rule Mining - Find associations between two or more classes

**Reinforcement**

## Data Exploration

### Numerical Exploration

Take measures of central tendency

Check for missing values

### Data Visualization

Histograms

Scatter Matrices

- These plot every numerical variable against every other numerical variable to help you visually seek out correlations

## Data Preprocessing

You will spend most of your time here

Basic Steps for Data Preprocessing:

- Split X and y
- Data Imputation
- Encode Categorical Variables
    - Label Encoder
    - ColumnTransformer + One Hot Encoding
- Train-test split
- Standard Scaler

Build out `Pipelines` in most cases to make the process easy to reimplement if necessary. A pipeline is just a series of steps that you can call at any time on data to apply the desired transformations to.

## Model Setup, Implementation, and Reimplementation

### Train/Fit Model

We can use the train-test split we had from the data preprocessing phase or switch to **cross-validation**

### Parameter Optimization

For larger datasets we can use Gradient or Stochastic Gradient Descent to tune our ML algorithm's parameters efficiently to minimize our cost function

### Hyperparameter Tuning

Hyperparameters are the parameters that decide the basic structure of your model (number of trees in a forest, learning rates for logistic regression)

Hyperparameter tuning seeks to improve the performance of an algorithm by tuning the hyperparameters to their optimal values

Manual Search

- Hyperparameters are manually input

Grid Search

- Very simplistic hyperparameter optimization where we search every single combination of reasonable hyperparameters

Random Search

- Randomly tries different hyperparameters. Isn't as thorough as a grid search but can be much faster for a high dimensional dataset

## Model Evaluation

Pick a metric to measure the accuracy of your model

Regression

- Root Mean Squared Error or Mean Squared Error are standard

Classification

- Accuracy Score
- Confusion matrix
- Precision/Recall and Harmonic/F1 Score
- Receiver Operating Characteristics (ROC) Curve