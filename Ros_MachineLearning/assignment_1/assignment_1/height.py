#!/usr/bin/env python3
"""
This module implements a ROS2 node for linear regression on height prediction based on weight using scikit-learn.
It subscribes to topics for training and prediction, publishes predictions, logs regression metrics, and saves a plot.

Classes:
    LinearRegressionWithMetricsAndPlot(Node):
        A ROS2 node that:
            - Trains a linear regression model from a CSV file containing 'Weight' and 'Height' columns.
            - Evaluates and logs model metrics (MAE, MSE, R²).
            - Publishes predictions for new weight inputs.
            - Saves and displays a plot comparing actual vs predicted heights.

Functions:
    main(args=None):
        Initializes and spins the ROS2 node.
"""
import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os

class HeightWeightPredictor(Node):
    def __init__(self):
        super().__init__('height_weight_predictor')
        self.get_logger().info('Height Weight Predictor Node Started')
        
        # Step 1: Load the dataset and get features/labels
        self.load_dataset()
        
        # Step 2: Split the data into train and test sets
        self.split_data()
        
        # Step 3: Fit the model
        self.train_model()
        
        # Step 4: Predict and evaluate
        self.predict_and_evaluate()
        
        # Visualize results
        self.visualize_results()
        
    def load_dataset(self):
        # Load the dataset (in a real ROS2 node, you'd load from a proper path)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        data_path = '/mnt/c/Users/nanin/Downloads/new_height_weight.csv'
        
        try:
            self.df = pd.read_csv(data_path)
            self.get_logger().info('Dataset loaded successfully')
            
            # Extract features (Height) and labels (Weight)
            self.X = self.df[['Height']].values  # Features
            self.y = self.df['Weight'].values    # Labels
            
            self.get_logger().info(f'Dataset shape: {self.df.shape}')
            self.get_logger().info(f'Features shape: {self.X.shape}')
            self.get_logger().info(f'Labels shape: {self.y.shape}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load dataset: {str(e)}')
            raise
    
    def split_data(self):
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        
        self.get_logger().info(f'Training set size: {len(self.X_train)}')
        self.get_logger().info(f'Testing set size: {len(self.X_test)}')
    
    def train_model(self):
        # Create and train the linear regression model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        self.get_logger().info('Model trained successfully')
        self.get_logger().info(f'Coefficient: {self.model.coef_[0]}')
        self.get_logger().info(f'Intercept: {self.model.intercept_}')
    
    def predict_and_evaluate(self):
        # Make predictions on the test set
        self.y_pred = self.model.predict(self.X_test)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        self.get_logger().info(f'Mean Absolute Error: {mae:.2f}')
        self.get_logger().info(f'R-squared Score: {r2:.2f}')
    
    def visualize_results(self):
        try:
            # Create figure directory if it doesn't exist
            os.makedirs('figures', exist_ok=True)
            
            # Plot 1: Training data with regression line
            plt.figure(figsize=(12, 6))
            plt.scatter(self.X_train, self.y_train, color='blue', label='Training Data')
            plt.plot(self.X_train, self.model.predict(self.X_train), color='red', label='Regression Line')
            plt.title('Height vs Weight (Training Set)')
            plt.xlabel('Height (inches)')
            plt.ylabel('Weight (pounds)')
            plt.legend()
            plt.grid(True)
            plt.savefig('figures/training_data.png')
            
            # Plot 2: Test data with predictions
            plt.figure(figsize=(12, 6))
            plt.scatter(self.X_test, self.y_test, color='green', label='Actual Test Data')
            plt.scatter(self.X_test, self.y_pred, color='orange', label='Predicted Values')
            plt.title('Height vs Weight (Test Set)')
            plt.xlabel('Height (inches)')
            plt.ylabel('Weight (pounds)')
            plt.legend()
            plt.grid(True)
            plt.savefig('figures/test_predictions.png')
            
            # Plot 3: Residual plot
            residuals = self.y_test - self.y_pred
            plt.figure(figsize=(12, 6))
            plt.scatter(self.y_pred, residuals, color='purple')
            plt.axhline(y=0, color='black', linestyle='--')
            plt.title('Residual Plot')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.grid(True)
            plt.savefig('figures/residual_plot.png')
            
            self.get_logger().info('Visualizations saved to figures directory')
            
        except Exception as e:
            self.get_logger().error(f'Failed to create visualizations: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = HeightWeightPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()