#!/usr/bin/env python3
"""
A ROS2 node for classifying fruits (apples and oranges) using machine learning.
This node implements a fruit classification system using logistic regression. It processes
a dataset containing fruit characteristics (weight, sphericity, and color) and trains
a model to distinguish between apples and oranges.
The node performs the following operations:
1. Loads and preprocesses the fruit dataset
2. Performs feature engineering
3. Creates data visualizations
4. Trains a logistic regression model
5. Evaluates the model performance
Classes:
    FruitClassifier: Main node class for fruit classification
Methods:
    load_data: Loads the fruit dataset from CSV file
    prepare_data: Processes and prepares data for model training
    visualize_data: Creates visualization plots of the dataset
    split_data: Splits data into training and testing sets
    train_and_evaluate: Trains the model and evaluates its performance
Dependencies:
    - ROS2 (rclpy)
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn
Output files:
    - fruit_visualization.png: Data visualization plots
    - fruit_confusion_matrix.png: Model performance confusion matrix
"""

import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class FruitClassifier(Node):
    def __init__(self):
        super().__init__('/mnt/c/Users/nanin/Downloads/fruits_weight_sphercity.csv')
        self.get_logger().info("Fruit Classifier Node Started")
        
        # Load dataset
        self.load_data()
        
        # Feature engineering and data preparation
        self.prepare_data()
        
        # Data visualization
        self.visualize_data()
        
        # Data splitting
        self.split_data()
        
        # Model training and evaluation
        self.train_and_evaluate()

    def load_data(self):
        # Load fruit dataset
        self.df = pd.read_csv('/mnt/c/Users/nanin/Downloads/fruits_weight_sphercity.csv')
        self.get_logger().info(f"Dataset loaded with {len(self.df)} samples")
        
        # Show initial data info
        self.get_logger().info("\nInitial data info:")
        self.get_logger().info(str(self.df.info()))
        self.get_logger().info("\nSample data:\n" + str(self.df.head()))

    def prepare_data(self):
        # Encode color to numerical values (custom mapping)
        color_mapping = {
            'Red': 80,
            'Orange': 40,
            'Greenish yellow': 30,
            'Green': 20,
            'Reddish yellow': 60
        }
        self.df['color_encoded'] = self.df['Color'].map(color_mapping)
        
        # Encode labels (apple=0, orange=1)
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['labels'])
        
        # Select features and target
        self.features = self.df[['Weight', 'Sphericity', 'color_encoded']]
        self.target = self.df['label_encoded']
        
        # Standardize features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)
        
        self.get_logger().info("\nFeature engineering completed")
        self.get_logger().info("Features used: Weight, Sphericity, color_encoded")

    def visualize_data(self):
        # Set style for plots
        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Weight vs Sphericity
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='Weight', y='Sphericity', hue='labels', data=self.df)
        plt.title('Weight vs Sphericity by Fruit Type')
        
        # Plot 2: Color distribution
        plt.subplot(1, 2, 2)
        sns.boxplot(x='labels', y='Weight', data=self.df)
        plt.title('Weight Distribution by Fruit Type')
        
        plt.tight_layout()
        plt.savefig('fruit_visualization.png')
        self.get_logger().info("Data visualization saved to fruit_visualization.png")
        plt.close()

    def split_data(self):
        # Split data into training (80%) and testing (20%) sets
        (self.X_train, self.X_test, 
         self.y_train, self.y_test) = train_test_split(
            self.scaled_features, 
            self.target, 
            test_size=0.2, 
            random_state=42
        )
        self.get_logger().info(f"\nData split into training ({len(self.X_train)} samples) and testing ({len(self.X_test)} samples) sets")

    def train_and_evaluate(self):
        # Initialize and train logistic regression model
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        self.get_logger().info("\nModel training completed")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        self.get_logger().info(f"Model Accuracy: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(
            self.y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_
        )
        self.get_logger().info("\nClassification Report:\n" + report)
        
        # Plot confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_, 
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('fruit_confusion_matrix.png')
        self.get_logger().info("Confusion matrix saved to fruit_confusion_matrix.png")
        plt.close()

def main(args=None):
    rclpy.init(args=args)
    node = FruitClassifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()