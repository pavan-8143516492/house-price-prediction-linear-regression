# Multi-Feature Linear Regression — House Price Prediction

A linear regression model implemented from scratch using NumPy to predict
house prices based on multiple features (size, number of bedrooms, number
of floors, and age of the home). This extends my earlier single-feature
project, as part of Andrew Ng's Machine Learning Specialization (Course 1).

## Project Overview

This notebook builds on the single-feature version by introducing:
- Multiple input features instead of one
- Vectorized operations using NumPy (np.dot) instead of manual loops
- Z-score normalization to handle features on different scales
  (e.g. size in the thousands vs. number of floors in single digits)

## Dataset

A small synthetic dataset representing Indian housing data with the
following features:

Size: Size of the house in sq ft
Bedrooms: Number of bedrooms
Floors: Number of floors
Age: Age of the home in years

Target variable: house price (in lakhs INR)

## Implementation

Prediction function: f(x) = w·x + b, computed using np.dot
Cost function: Mean Squared Error, vectorized
Gradient computation: Vectorized gradients for all features
simultaneously using np.dot(X.T, error)
Gradient descent: Iteratively updates weights and bias to
minimize cost
Feature scaling: Z-score normalization applied before training

## Tech stack

Python, NumPy, Matplotlib
