#!/usr/bin/env python3
"""
Iris Classifier ROS2 Node
This module defines the `IrisClassifier` class, a ROS2 node for loading, visualizing, and classifying the Iris dataset using multiple machine learning models. The node performs the following steps:
1. Loads the Iris dataset from a CSV file.
2. Performs feature engineering by adding PetalArea and SepalArea features.
3. Encodes categorical labels and standardizes features.
4. Visualizes the data using scatter plots and box plots, saving the figures to disk.
5. Splits the data into training and testing sets.
6. Trains and evaluates three classifiers: K-Nearest Neighbors, Decision Tree, and Random Forest.
7. Logs evaluation metrics (accuracy, precision, recall, F1 score) and classification reports.
8. Plots and saves confusion matrices for each model.
Classes:
    IrisClassifier(Node): 
        A ROS2 node that encapsulates the entire workflow for Iris dataset classification and visualization.
Functions:
    main(args=None): 
        Initializes and spins the ROS2 node.
Usage:
    Run this script as a ROS2 node to perform data analysis and classification on the Iris dataset. Visualization images and confusion matrices will be saved to the working directory.
"""

import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class IrisClassifier(Node):
    def __init__(self):
        super().__init__('iris_classifier')
        self.get_logger().info("Iris Classifier Node Started")
        
        # Load dataset
        self.load_data()
        
        # Feature engineering and label extraction
        self.prepare_data()
        
        # Data visualization
        self.visualize_data()
        
        # Data splitting
        self.split_data()
        
        # Train and evaluate models
        self.evaluate_models()

    def load_data(self):
        self.df = pd.read_csv('/mnt/c/Users/nanin/Downloads/Iris.csv')
        self.get_logger().info(f"Dataset loaded with {len(self.df)} samples")

    def prepare_data(self):
        # Feature engineering
        self.df = self.df.drop('Id', axis=1)
        self.df['PetalArea'] = self.df['PetalLengthCm'] * self.df['PetalWidthCm']
        self.df['SepalArea'] = self.df['SepalLengthCm'] * self.df['SepalWidthCm']
        
        # Extract features and labels
        self.features = self.df.drop('Species', axis=1)
        self.labels = self.df['Species']
        
        # Encode labels to numerical values
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        # Standardize features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)
        
        self.get_logger().info("Feature engineering and label extraction completed")

    def visualize_data(self):
        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=self.df)
        plt.title('Petal Length vs Width by Species')
        
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='PetalArea', y='SepalArea', hue='Species', data=self.df)
        plt.title('Petal Area vs Sepal Area by Species')
        
        plt.subplot(2, 2, 3)
        sns.boxplot(x='Species', y='PetalLengthCm', data=self.df)
        plt.title('Petal Length Distribution by Species')
        
        plt.tight_layout()
        plt.savefig('iris_visualization.png')
        self.get_logger().info("Data visualization saved to iris_visualization.png")
        plt.close()

    def split_data(self):
        (self.X_train, self.X_test, 
         self.y_train, self.y_test) = train_test_split(
            self.scaled_features, 
            self.encoded_labels, 
            test_size=0.2, 
            random_state=42
        )
        self.get_logger().info(f"Data split into training ({len(self.X_train)} samples) and testing ({len(self.X_test)} samples) sets")

    def evaluate_models(self):
        # Initialize models
        models = {
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            self.get_logger().info(f"\nEvaluating {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Log metrics
            self.get_logger().info(f"{name} Accuracy: {accuracy:.4f}")
            self.get_logger().info(f"{name} Precision: {precision:.4f}")
            self.get_logger().info(f"{name} Recall: {recall:.4f}")
            self.get_logger().info(f"{name} F1 Score: {f1:.4f}")
            
            # Classification report
            report = classification_report(
                self.y_test, 
                y_pred, 
                target_names=self.label_encoder.classes_
            )
            self.get_logger().info(f"\n{name} Classification Report:\n{report}")
            
            # Plot confusion matrix
            self.plot_confusion_matrix(y_pred, name)

    def plot_confusion_matrix(self, y_pred, model_name):
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_, 
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        filename = f'iris_confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
        plt.savefig(filename)
        self.get_logger().info(f"Confusion matrix saved to {filename}")
        plt.close()

def main(args=None):
    rclpy.init(args=args)
    node = IrisClassifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()