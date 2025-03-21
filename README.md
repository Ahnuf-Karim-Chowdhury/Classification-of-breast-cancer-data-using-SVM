# Classification-of-Breast-Cancer-Data-Using-SVM

## Overview

This project implements a machine learning approach to classify breast cancer data as either malignant or benign using Support Vector Machines (SVM). Specifically, it follows **Approach-01: Training It Newly Every Iteration**, where the model is retrained from scratch for each run, ensuring the results reflect the performance on new data splits.

## How It Works

### Data Description

The project uses the breast cancer dataset provided by `sklearn`. The dataset contains numerical features extracted from digitized images of breast mass tissue and includes labels denoting whether the tissue is malignant (class 0) or benign (class 1).

### Approach

1. **Data Splitting**: The dataset is divided into training and testing subsets (80% training, 20% testing). This ensures the model is trained on unseen data during each iteration.
   
2. **Model Training**: The SVM model is trained with a linear kernel. This kernel is chosen for its simplicity and efficiency in handling linearly separable data. During each iteration, the entire training process starts anew, without any reliance on prior runs.

3. **Prediction**: The trained model predicts the classes of the test dataset.

4. **Evaluation**: The accuracy of predictions is calculated using metrics from `sklearn`, providing a quantitative measure of the model's performance.

5. **Visualization**: The project includes graphical comparisons between actual and predicted outputs for the test data, allowing users to visually assess classification accuracy.

### Results and Accuracy

#### KNN Classification Accuracy Graph

![KNN Accuracy Graph](https://github.com/Ahnuf-Karim-Chowdhury/Classification-of-breast-cancer-data-using-SVM/blob/main/Approach%20-%2001%20-%20Training%20it%20Newly%20Every%20Iteration/KNN%20Accuracy.png?raw=true)

**Explanation of the Graph:**

- **X-axis**: Test Data Index (represents the index of test data samples).
- **Y-axis**: Class (0 = Malignant, 1 = Benign).

The graph plots the actual class labels (blue circles with dashed lines) against the predicted class labels (red crosses with solid lines). The alignment of these points shows that the model accurately predicts the majority of test samples, achieving an accuracy rate of **94.74%** for KNN classification. This high accuracy demonstrates the reliability of KNN in identifying breast cancer cases.

#### SVM Classification Accuracy Graph

![SVM Accuracy Graph](https://github.com/Ahnuf-Karim-Chowdhury/Classification-of-breast-cancer-data-using-SVM/blob/main/Approach%20-%2001%20-%20Training%20it%20Newly%20Every%20Iteration/SVM%20Accuracy.png?raw=true)

**Explanation of the Graph:**

- **X-axis**: Test Data Index (represents the index of test data samples).
- **Y-axis**: Class (0 = Malignant, 1 = Benign).

This graph visualizes the actual class labels (blue circles with dashed lines) alongside the predicted labels (red crosses with solid lines). The strong agreement between the two confirms high classification accuracy. Based on the graph, the SVM model achieves an updated accuracy rate of **95.87%**, emphasizing its superior performance in distinguishing malignant and benign cases effectively.

### Response to New Data Input

This approach facilitates seamless retraining with new data. Each iteration begins from scratch, ensuring the model adapts effectively to evolving datasets while maintaining robust performance.

## Project Features

- Implementation of SVM for binary classification.
- Comparison with K-Nearest Neighbors (KNN) as an alternate model.
- Training and testing pipelines designed to ensure unbiased results.
- Graphical visualization of results to enhance interpretability.

## Future Enhancements

- Experimenting with other kernels (e.g., RBF) to explore non-linear data separability.
- Incorporating feature selection or dimensionality reduction to optimize model performance.
- Expanding the dataset for better generalization.

