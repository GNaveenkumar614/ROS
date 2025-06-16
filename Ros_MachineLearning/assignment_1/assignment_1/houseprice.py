#!/usr/bin/env python3
"""
houseprice.py

A ROS2 node for binary classification of Boston housing prices using logistic regression.
The node loads a dataset, performs feature engineering, visualizes data, splits into train/test sets,
trains a logistic regression model, evaluates its performance, and logs results.

Classes:
    HousePriceNode(Node):
        A ROS2 node that:
            - Loads and preprocesses the Boston housing dataset.
            - Performs binary classification: 'high' price if MEDV > 25, else 'low'.
            - Standardizes features and splits data into training and testing sets.
            - Trains a logistic regression model.
            - Evaluates the model using accuracy, classification report, confusion matrix, and cross-validation.
            - Visualizes data correlations and feature relationships, saving plots to /tmp.

Functions:
    main(args=None):
        Initializes and spins the HousePriceNode.

Usage:
    Run as a ROS2 node to perform housing price classification and analysis.
"""

import rclpy
from rclpy.node import Node

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class HousePriceNode(Node):
    def __init__(self):
        super().__init__('house_price_node')
        self.get_logger().info("🏡 House Price Node Initialized")

        # Load the dataset
        file_path = '/mnt/c/Users/nanin/Downloads/boston_housing.csv'
        df = pd.read_csv(file_path)

        # Feature Engineering
        self.get_logger().info("🔧 Performing feature engineering...")
        df.dropna(inplace=True)

        # Binary classification: MEDV > 25 is 'high' price (1), else low (0)
        df['PriceCategory'] = (df['MEDV'] > 25).astype(int)

        # Separate features and label
        X_raw = df.drop(columns=['MEDV', 'PriceCategory'])
        y = df['PriceCategory']

        # Standardize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        self.get_logger().info("📊 Visualizing data...")
        self.visualize_data(df)

        # Split the dataset
        self.get_logger().info("✂️ Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        self.get_logger().info("🧠 Training Logistic Regression model...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate the model
        self.get_logger().info("📈 Evaluating model...")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        self.get_logger().info(f"✅ Accuracy: {acc:.2f}")
        self.get_logger().info("📋 Classification Report:\n" + classification_report(y_test, y_pred))
        self.get_logger().info("🧮 Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

        # Optional: Cross-validation
        scores = cross_val_score(model, X, y, cv=5)
        self.get_logger().info(f"📊 Cross-validation accuracy (5-fold): {scores.mean():.2f}")

    def visualize_data(self, df):
        # Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig('/tmp/house_price_corr_heatmap.png')
        plt.close()

        # Scatter: RM vs MEDV
        plt.figure()
        sns.scatterplot(x='RM', y='MEDV', data=df, hue='PriceCategory', palette='Set1')
        plt.title("Rooms vs Price")
        plt.xlabel("Average Number of Rooms (RM)")
        plt.ylabel("Median House Price (MEDV)")
        plt.tight_layout()
        plt.savefig('/tmp/house_price_scatter.png')
        plt.close()

        self.get_logger().info("🖼️ Plots saved to /tmp:\n"
                               " - /tmp/house_price_corr_heatmap.png\n"
                               " - /tmp/house_price_scatter.png")


def main(args=None):
    rclpy.init(args=args)
    node = HousePriceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
