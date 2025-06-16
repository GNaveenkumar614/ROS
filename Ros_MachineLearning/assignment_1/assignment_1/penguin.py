#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MultiDatasetClassifier(Node):
    def __init__(self):
        super().__init__('multi_dataset_classifier')
        self.get_logger().info("Multi-Dataset Classifier Node Started")
        
        # Process Penguin dataset
        self.get_logger().info("\n==== Processing Penguin Dataset ====")
        self.process_penguin_dataset()
        
        # Process Fruit dataset
        self.get_logger().info("\n==== Processing Fruit Dataset ====")
        self.process_fruit_dataset()
        
        # Compare results
        self.compare_results()

    def process_penguin_dataset(self):
        # Load dataset
        df = pd.read_csv('/mnt/c/Users/nanin/Downloads/Penguins.csv')
        self.get_logger().info(f"\nPenguin dataset loaded with {len(df)} samples")
        
        # Data cleaning
        df = df.dropna()
        self.get_logger().info(f"After cleaning: {len(df)} samples remaining")
        
        # Feature engineering
        df['culmen_area'] = df['culmen_length_mm'] * df['culmen_depth_mm']
        df['flipper_to_mass_ratio'] = df['flipper_length_mm'] / df['body_mass_g']
        
        # Encode categorical features
        le = LabelEncoder()
        df['species_encoded'] = le.fit_transform(df['species'])
        df['sex_encoded'] = df['sex'].apply(lambda x: 1 if x == 'MALE' else 0)
        df = pd.get_dummies(df, columns=['island'], prefix='island')
        
        # Select features and target
        features = df[['culmen_length_mm', 'culmen_depth_mm', 
                      'flipper_length_mm', 'body_mass_g',
                      'culmen_area', 'flipper_to_mass_ratio',
                      'sex_encoded', 'island_Biscoe', 
                      'island_Dream', 'island_Torgersen']]
        target = df['species_encoded']
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, target, test_size=0.2, random_state=42)
        
        # Visualize
        self.visualize_penguin_data(df)
        
        # Train SVM
        svm = SVC(kernel='linear', random_state=42)
        svm.fit(X_train, y_train)
        
        # Evaluate
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.penguin_results = {
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred, target_names=le.classes_),
            'model': svm
        }
        
        self.get_logger().info(f"\nPenguin SVM Accuracy: {accuracy:.4f}")
        self.get_logger().info("\nClassification Report:\n" + self.penguin_results['report'])

    def process_fruit_dataset(self):
        # Load dataset
        df = pd.read_csv('fruits_weight_sphercity.csv')
        self.get_logger().info(f"\nFruit dataset loaded with {len(df)} samples")
        
        # Feature engineering
        color_mapping = {
            'Red': 80,
            'Orange': 40,
            'Greenish yellow': 30,
            'Green': 20,
            'Reddish yellow': 60
        }
        df['color_encoded'] = df['Color'].map(color_mapping)
        
        # Encode labels
        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df['labels'])
        
        # Select features and target
        features = df[['Weight', 'Sphericity', 'color_encoded']]
        target = df['label_encoded']
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, target, test_size=0.2, random_state=42)
        
        # Visualize
        self.visualize_fruit_data(df)
        
        # Train SVM
        svm = SVC(kernel='linear', random_state=42)
        svm.fit(X_train, y_train)
        
        # Evaluate
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.fruit_results = {
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred, target_names=le.classes_),
            'model': svm
        }
        
        self.get_logger().info(f"\nFruit SVM Accuracy: {accuracy:.4f}")
        self.get_logger().info("\nClassification Report:\n" + self.fruit_results['report'])

    def visualize_penguin_data(self, df):
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Culmen Length vs Depth by Species
        plt.subplot(2, 2, 1)
        sns.scatterplot(x='culmen_length_mm', y='culmen_depth_mm', 
                       hue='species', data=df)
        plt.title('Culmen Length vs Depth by Species')
        
        # Plot 2: Flipper Length Distribution
        plt.subplot(2, 2, 2)
        sns.boxplot(x='species', y='flipper_length_mm', data=df)
        plt.title('Flipper Length by Species')
        
        # Plot 3: Body Mass Distribution
        plt.subplot(2, 2, 3)
        sns.violinplot(x='species', y='body_mass_g', hue='sex', 
                      data=df, split=True)
        plt.title('Body Mass Distribution by Species and Sex')
        
        plt.tight_layout()
        plt.savefig('penguin_visualization.png')
        self.get_logger().info("Penguin visualizations saved to penguin_visualization.png")
        plt.close()

    def visualize_fruit_data(self, df):
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Weight vs Sphericity
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='Weight', y='Sphericity', hue='labels', data=df)
        plt.title('Weight vs Sphericity by Fruit Type')
        
        # Plot 2: Color distribution
        plt.subplot(1, 2, 2)
        sns.boxplot(x='labels', y='color_encoded', data=df)
        plt.title('Color Encoding Distribution by Fruit Type')
        
        plt.tight_layout()
        plt.savefig('fruit_visualization.png')
        self.get_logger().info("Fruit visualizations saved to fruit_visualization.png")
        plt.close()

    def compare_results(self):
        plt.figure(figsize=(10, 5))
        
        # Create comparison bar plot
        results = pd.DataFrame({
            'Dataset': ['Penguin', 'Fruit'],
            'Accuracy': [self.penguin_results['accuracy'], self.fruit_results['accuracy']]
        })
        
        sns.barplot(x='Dataset', y='Accuracy', data=results)
        plt.title('SVM Classifier Accuracy Comparison')
        plt.ylim(0, 1.1)
        
        # Add accuracy values on top of bars
        for index, row in results.iterrows():
            plt.text(index, row.Accuracy + 0.02, 
                    f"{row.Accuracy:.3f}", 
                    color='black', ha="center")
        
        plt.savefig('accuracy_comparison.png')
        self.get_logger().info("\nAccuracy comparison saved to accuracy_comparison.png")
        plt.close()
        
        self.get_logger().info("\n==== Final Comparison ====")
        self.get_logger().info(f"Penguin Accuracy: {self.penguin_results['accuracy']:.4f}")
        self.get_logger().info(f"Fruit Accuracy: {self.fruit_results['accuracy']:.4f}")

def main(args=None):
    rclpy.init(args=args)
    node = MultiDatasetClassifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()