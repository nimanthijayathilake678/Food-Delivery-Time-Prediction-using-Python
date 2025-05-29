# Food Delivery Time Prediction using Python

This project predicts food delivery times based on key factors such as the delivery partner's age, ratings, and the distance to be covered. It combines data analysis, visualization, and machine learning techniques to provide insights and predictions for optimizing delivery efficiency.

## Overview
The project uses a dataset containing information about food deliveries, including delivery partner details, ratings, and distances. It explores the relationships between these factors and delivery times through visualizations and trains a Long Short-Term Memory (LSTM) neural network to predict delivery times.

## Features
- **Data Preprocessing**: Cleaned and prepared the dataset, including calculating distances using the Haversine formula.
- **Data Visualization**: Created interactive visualizations to explore relationships between:
  - Delivery time and distance.
  - Delivery partner's age and delivery time.
  - Ratings and delivery time.
  - Type of order and vehicle used.
- **Machine Learning**: Built and trained an LSTM neural network to predict delivery times.
- **User Interaction**: Allows users to input delivery partner details and distance to predict the expected delivery time.

## Technologies Used
- **Python**: Programming language for data analysis and machine learning.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical computations.
- **Plotly Express**: For creating interactive visualizations.
- **Keras**: For building and training the LSTM neural network.
- **Scikit-learn**: For splitting the dataset into training and testing sets.
