import json

# Create the complete notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add all cells
cells_data = [
    # Title
    ("markdown", "# DEEP NEURAL NETWORKS - ASSIGNMENT 2: CNN FOR IMAGE CLASSIFICATION\n## Convolutional Neural Networks: Custom Implementation vs Transfer Learning"),
    
    # Student Info
    ("markdown", "## STUDENT INFORMATION (REQUIRED - DO NOT DELETE)\n\n**BITS ID:** 2025AA05036  \n**Name:** JOHN DOE  \n**Email:** john.doe@wilp.bits-pilani.ac.in  \n**Date:** 02-05-2026\n\n---\n\n**IMPORTANT:** Replace the above information with your actual details before submission!"),
    
    # Imports
    ("markdown", "## Import Required Libraries"),
    ("code", """# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")"""),
    
    # Part 1
    ("markdown", "## PART 1: DATASET LOADING AND EXPLORATION"),
    ("code", """# 1.1 Dataset Selection and Loading
# Using Cats vs Dogs dataset from TensorFlow Datasets

print("Loading Cats vs Dogs dataset...")

# Load dataset
(ds_train, ds_test), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:85%]', 'train[85%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Dataset metadata
dataset_name = "Cats vs Dogs"
dataset_source = "TensorFlow Datasets (Microsoft)"
n_samples = ds_info.splits['train'].num_examples
n_classes = 2
samples_per_class = f"min: {n_samples//2}, max: {n_samples//2}, avg: {n_samples//2}"
image_shape = [224, 224, 3]
problem_type = "binary_classification"
train_test_ratio = "85/15"
train_samples = int(n_samples * 0.85)
test_samples = int(n_samples * 0.15)

# Primary metric selection
primary_metric = "accuracy"
metric_justification = "Accuracy is chosen as the primary metric because the Cats vs Dogs dataset is balanced with approximately equal samples per class, making accuracy a reliable performance indicator."

print("\\n" + "="*70)
print("DATASET INFORMATION")
print("="*70)
print(f"Dataset: {dataset_name}")
print(f"Source: {dataset_source}")
print(f"Total Samples: {n_samples}")
print(f"Number of Classes: {n_classes}")
print(f"Samples per Class: {samples_per_class}")
print(f"Image Shape: {image_shape}")
print(f"Primary Metric: {primary_metric}")
print(f"Metric Justification: {metric_justification}")
print(f"\\nTrain/Test Split: {train_test_ratio}")
print(f"Training Samples: {train_samples}")
print(f"Test Samples: {test_samples}")"""),
    
    ("code", """# 1.2 Data Preprocessing Functions

def preprocess_image(image, label):
    \"\"\"Resize and normalize images\"\"\"
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment_image(image, label):
    \"\"\"Apply data augmentation\"\"\"
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    return image, label

# Configure datasets
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Prepare training dataset with augmentation
ds_train = ds_train.map(preprocess_image, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.map(augment_image, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(1000)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Prepare test dataset
ds_test = ds_test.map(preprocess_image, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(AUTOTUNE)

print("Dataset preprocessing complete!")"""),
    
    ("code", """# 1.3 Data Visualization

# Display sample images
plt.figure(figsize=(12, 8))
class_names = ['Cat', 'Dog']

for images, labels in ds_test.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(f"{class_names[labels[i].numpy()]}")
        plt.axis('off')

plt.suptitle('Sample Images from Cats vs Dogs Dataset', fontsize=16)
plt.tight_layout()
plt.show()

# Class distribution
plt.figure(figsize=(8, 6))
class_counts = [n_samples//2, n_samples//2]
plt.bar(class_names, class_counts, color=['orange', 'skyblue'])
plt.title('Class Distribution', fontsize=14)
plt.ylabel('Number of Images')
plt.xlabel('Class')
for i, v in enumerate(class_counts):
    plt.text(i, v + 100, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.show()"""),
]

# Add cells to notebook
for cell_type, content in cells_data:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": content.split('\n') if '\n' in content else [content]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    notebook["cells"].append(cell)

# Save notebook
with open('2025AA05036_cnn_assignment.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook part 1 created successfully!")
