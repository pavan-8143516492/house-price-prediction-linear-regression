# House Price Prediction — Linear Regression from Scratch

A simple linear regression model implemented from scratch using NumPy 
to predict house prices based on size, as part of my learning journey 
through Andrew Ng's Machine Learning Specialization (Course 1).

## Project Overview

This project implements the core building blocks of linear regression:
- Cost function (Mean Squared Error)
- Gradient computation
- Gradient descent optimization

The model is trained on a small dataset of house sizes (in 100 sq ft) 
and corresponding prices (in lakhs), and learns to predict prices for 
new house sizes.

## What I Implemented

- `compute_cost()` — calculates the mean squared error between predictions and actual prices
- `compute_gradient()` — computes the gradients of the cost function with respect to w and b
- `gradient_descent()` — iteratively updates w and b to minimize cost

## Results

- The cost decreased steadily from **23.77** at iteration 0 to **0.28** 
  by iteration 900, confirming convergence
- Final learned parameters: **w = 1.793**, **b = 10.374**
- The model fits a line through the training data that captures the 
  relationship between house size and price
- Example prediction: a **1300 sq ft** house is predicted at 
  approximately **33.68 lakhs**

## Tools Used

- Python
- NumPy
- Matplotlib
- Google Colab

## Key Learning

This project helped me understand how gradient descent works under the 
hood — including the impact of learning rate on convergence, and why 
the gradient calculation needs to be averaged correctly across all 
training examples. A subtle bug in averaging the gradient inside the 
loop instead of outside taught me how sensitive gradient descent is to 
even small implementation errors.

## Next Steps

- Extend to multiple features (bedrooms, location, age of house)
- Apply feature scaling for multivariate gradient descent
- Compare results with scikit-learn's `LinearRegression`
- Apply the model to a real-world dataset (e.g., Bangalore house prices from Kaggle)
