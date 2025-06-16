#!/usr/bin/env python3
"""
Breast Cancer Classifier ROS2 Node
This module defines a ROS2 node for classifying breast cancer diagnoses using a Support Vector Machine (SVM) classifier.
It loads and preprocesses a breast cancer dataset, visualizes the data, trains an SVM model, evaluates its performance,
and visualizes the decision boundary of the classifier.
Classes:
    BreastCancerClassifier(Node): 
        A ROS2 node that encapsulates the workflow for breast cancer classification, including:
            - Loading and preprocessing the dataset.
            - Visualizing feature distributions.
            - Training and evaluating an SVM classifier.
            - Visualizing the SVM decision boundary.
Functions:
    main(args=None):
        Initializes and spins the ROS2 node.
Usage:
    Run this script as a ROS2 node. The node will log progress and save visualizations as PNG files.
Dataset:
    Expects a CSV file named 'Breast_Cancer.csv' with columns including 'id', 'diagnosis', and various feature columns.
    The 'diagnosis' column should contain 'M' (malignant) or 'B' (benign) values.
Visualization:
    - Saves a scatter plot of two features ('radius_mean' vs 'texture_mean') as 'data_distribution.png'.
    - Saves a plot of the SVM decision boundary using two features ('perimeter_mean' vs 'area_mean') as 'svm_decision_boundary.png'.
Note:
    This script is intended for demonstration and educational purposes within a ROS2 environment.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

class BreastCancerClassifier(Node):
    def __init__(self):
        super().__init__('breast_cancer_classifier')
        self.get_logger().info('Breast Cancer Classifier Node Started')
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Visualize data
        self.visualize_data()
        
        # Train and evaluate SVM
        self.train_and_evaluate_svm()
        
        # Visualize predictions
        self.visualize_predictions()
    
    def load_and_preprocess_data(self):
        # Load the dataset (in a real ROS2 app, this would come from a topic or service)
        try:
            data = pd.read_csv('/mnt/c/Users/nanin/Downloads/Breast_Cancer.csv')
            self.get_logger().info('Successfully loaded breast cancer dataset')
        except Exception as e:
            self.get_logger().error(f'Failed to load dataset: {str(e)}')
            return
        
        # Drop the ID column as it's not useful for classification
        data.drop(['id'], axis=1, inplace=True)
        
        # Convert diagnosis to binary values (M=1, B=0)
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        
        # Separate features and target
        self.X = data.drop(['diagnosis'], axis=1)
        self.y = data['diagnosis']
        
        # Standardize the features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.3, random_state=42)
        
        self.get_logger().info(f'Dataset shapes - X_train: {self.X_train.shape}, X_test: {self.X_test.shape}')
    
    def visualize_data(self):
        # Select two features to visualize (mean radius and mean texture)
        feature1 = 'radius_mean'
        feature2 = 'texture_mean'
        
        plt.figure(figsize=(10, 6))
        
        # Plot benign cases
        benign = self.X[self.y == 0]
        plt.scatter(benign[feature1], benign[feature2], color='blue', label='Benign', alpha=0.6)
        
        # Plot malignant cases
        malignant = self.X[self.y == 1]
        plt.scatter(malignant[feature1], malignant[feature2], color='red', label='Malignant', alpha=0.6)
        
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title('Breast Cancer Diagnosis - Radius Mean vs Texture Mean')
        plt.legend()
        plt.grid(True)
        plt.savefig('data_distribution.png')  # Save instead of show for ROS2
        self.get_logger().info('Saved data visualization to data_distribution.png')
    
    def train_and_evaluate_svm(self):
        # Create and train SVM classifier
        self.svm_classifier = SVC(kernel='linear', random_state=42)
        self.svm_classifier.fit(self.X_train, self.y_train)
        
        # Predict on test set
        y_pred = self.svm_classifier.predict(self.X_test)
        
        # Evaluate performance
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred)
        
        self.get_logger().info(f'\nSVM Classifier Results:')
        self.get_logger().info(f'Accuracy: {accuracy:.4f}')
        self.get_logger().info(f'Confusion Matrix:\n{conf_matrix}')
        self.get_logger().info(f'Classification Report:\n{class_report}')
    
    def visualize_predictions(self):
    # Select indices of the two features to visualize (e.g., perimeter_mean and area_mean)
        feature1_idx = 2  # perimeter_mean
        feature2_idx = 3  # area_mean
        
        # Extract only these two features for training a new SVM
        X_train_2d = self.X_train[:, [feature1_idx, feature2_idx]]
        
        # Train a new SVM on just these 2 features
        svm_2d = SVC(kernel='linear', random_state=42)
        svm_2d.fit(X_train_2d, self.y_train)
        
        # Create mesh grid for decision boundary
        h = 0.02  # step size in the mesh
        x_min, x_max = self.X_train[:, feature1_idx].min() - 1, self.X_train[:, feature1_idx].max() + 1
        y_min, y_max = self.X_train[:, feature2_idx].min() - 1, self.X_train[:, feature2_idx].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predict on mesh grid
        Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 6))
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
        
        # Plot test points (using the same 2 features)
        X_test_2d = self.X_test[:, [feature1_idx, feature2_idx]]
        plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], 
                c=self.y_test, cmap=plt.cm.Paired, edgecolors='k')
        
        plt.xlabel('Perimeter Mean (standardized)')
        plt.ylabel('Area Mean (standardized)')
        plt.title('SVM Decision Boundary (2-Feature Subset)')
        plt.savefig('svm_decision_boundary.png')
        self.get_logger().info('Saved SVM decision boundary visualization to svm_decision_boundary.png')
        

def main(args=None):
    rclpy.init(args=args)
    node = BreastCancerClassifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()