#!/usr/bin/env python3
"""
This module implements a ROS 2 node for performing linear regression on a dataset of human brain weights and head sizes.
It provides the following functionalities:

- Subscribes to the `/train_from_csv` topic (std_msgs/String) to receive a trigger for training the model from a CSV file.
- Subscribes to the `/predict_input` topic (std_msgs/Float32) to receive input values for prediction using the trained model.
- Publishes predictions to the `/prediction` topic (std_msgs/Float32).
- Trains a linear regression model using scikit-learn on the provided dataset.
- Calculates and logs regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
- Generates and saves a plot comparing actual vs. predicted values for the test set.
- Handles errors related to file access, CSV format, and prediction requests before training.

Classes:
    LinearRegressionWithMetricsAndPlot(Node): 
        ROS 2 node that encapsulates the linear regression model, training, prediction, and plotting logic.

Functions:
    main(args=None): 
        Initializes and spins the ROS 2 node.
"""

import rclpy
from rclpy.node import Node
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from std_msgs.msg import String, Float32
import matplotlib.pyplot as plt

class LinearRegressionWithMetricsAndPlot(Node):
    def __init__(self):
        super().__init__('linear_regression_metrics_plot_node')

        self.model = LinearRegression()
        self.trained = False

        # Subscribers
        self.create_subscription(String, "/train_from_csv", self.train_from_csv_callback, 10)
        self.create_subscription(Float32, "/predict_input", self.predict_callback, 10)

        # Publisher
        self.pred_pub = self.create_publisher(Float32, "/prediction", 10)

        self.get_logger().info("Linear Regression Node with Metrics and Plot Initialized.")

    def train_from_csv_callback(self, msg):
        home_dir = os.path.expanduser("~")
        #dataset_path = os.path.join(home_dir, "Downloads", "HumanBrain_WeightandHead_size.csv")
        dataset_path = "/mnt/c/Users/nanin/Downloads/HumanBrain_WeightandHead_size.csv"

        if not os.path.exists(dataset_path):
            self.get_logger().error(f"Dataset file does not exist at {dataset_path}")
            return
        try:
            df = pd.read_csv(dataset_path)

            if 'Head Size(cm^3)' not in df.columns or 'Brain Weight(grams)' not in df.columns:
                self.get_logger().error("CSV must contain 'Head Size(cm^3)' and 'Brain Weight(grams)' columns.")
                return

            X = df[['Head Size(cm^3)']]  # Make sure X is 2D
            y = df['Brain Weight(grams)']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.model.fit(X_train, y_train)
            self.trained = True

            y_pred = self.model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.get_logger().info(f"Model trained.")
            self.get_logger().info(f"MAE: {mae:.4f}")
            self.get_logger().info(f"MSE: {mse:.4f}")
            self.get_logger().info(f"R² Score: {r2:.4f}")

            self.plot_results(X_test, y_test, y_pred)

        except Exception as e:
            self.get_logger().error(f"Error during training: {e}")

    def plot_results(self, X_test, y_test, y_pred):
        plt.figure(figsize=(8, 5))
        plt.scatter(X_test, y_test, color='blue', label='Actual')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
        plt.title('Linear Regression: Actual vs Predicted')
        plt.xlabel('Head Size (cm^3)')
        plt.ylabel('Brain Weight (grams)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('/tmp/linear_regression_plot.png')
        self.get_logger().info("Plot saved to /tmp/linear_regression_plot.png")
        plt.show()
        plt.close()

    def predict_callback(self, msg):
        if not self.trained:
            self.get_logger().warn("Model not trained yet. Please send a CSV path to /train_from_csv.")
            return

        x_input = np.array([[msg.data]])
        y_pred = self.model.predict(x_input)
        msg_out = Float32()
        msg_out.data = float(y_pred[0])
        self.pred_pub.publish(msg_out)
        self.get_logger().info(f"Predicted {y_pred[0]:.2f} for input {msg.data:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = LinearRegressionWithMetricsAndPlot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
