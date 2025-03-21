# Classification-of-Breast-Cancer-Data-Using-SVM

## Overview

This project implements a machine learning approach to classify breast cancer data as either malignant or benign using Support Vector Machines (SVM). 

## How It Works

### Data Description

The project uses the breast cancer dataset provided by `sklearn`. The dataset contains numerical features extracted from digitized images of breast mass tissue and includes labels denoting whether the tissue is malignant (class 0) or benign (class 1).

### Approach - 01 Training It Newly Every Iteration

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

This graph visualizes the actual class labels (blue circles with dashed lines) alongside the predicted labels (red crosses with solid lines). The strong agreement between the two confirms high classification accuracy. Based on the graph, the SVM model achieves an updated accuracy rate of **96.49%**, emphasizing its superior performance in distinguishing malignant and benign cases effectively.

### Response to New Data Input

This approach facilitates seamless retraining with new data. Each iteration begins from scratch, ensuring the model adapts effectively to evolving datasets while maintaining robust performance.

## Project Features

- Implementation of SVM for binary classification.
- Comparison with K-Nearest Neighbors (KNN) as an alternate model.
- Training and testing pipelines designed to ensure unbiased results.
- Graphical visualization of results to enhance interpretability.

## Future Enhancements Reqired Here

- Experimenting with other kernels (e.g., RBF) to explore non-linear data separability.
- Incorporating feature selection or dimensionality reduction to optimize model performance.
- Expanding the dataset for better generalization.

---

## Approach - 02 Saving the Most Accurate Data

This approach focuses on optimizing the SVM model to achieve the highest possible accuracy and then saving the best-performing model for future use.

### Methodology

1.  **Data Preparation:**
    * The breast cancer dataset is loaded from scikit-learn.
    * The dataset is split into training and testing sets to evaluate the model's performance on unseen data.
2.  **Model Optimization:**
    * Two optimization techniques are employed to find the best SVM model parameters:
        * **Randomized Search:**
            * This method explores a wide range of hyperparameters by randomly sampling from defined distributions.
            * It efficiently searches for good parameter combinations.
            * Hyperparameters such as 'C' (regularization parameter), 'kernel' (linear, rbf, poly), and 'degree' (for polynomial kernel) are tuned.
        * **Grid Search:**
            * This method exhaustively searches through a predefined grid of hyperparameters.
            * It guarantees finding the optimal parameters within the specified grid.
            * Hyperparameters such as 'C', 'kernel', and 'degree' are tuned.
    * Both methods utilize cross-validation to assess the model's performance for each parameter combination, ensuring robustness and generalization.
3.  **Model Training and Evaluation:**
    * The best SVM model, as determined by the optimization process, is trained on the training data.
    * The trained model is then used to predict the class labels for the test data.
    * The accuracy of the model is calculated by comparing the predicted labels with the actual labels.
4.  **Model Persistence:**
    * The best-performing SVM model is saved to a file using `pickle`. This allows the model to be loaded and used for future predictions without retraining.
5.  **Visualization:**
    * A plot is generated to visualize the actual and predicted class labels for the test data, providing a clear comparison of the model's performance.
    * The plot displays the "Actual" and "Predicted" values, and the accuracy of the model is displayed in the title of the plot.
6.  **New Data Input:**
    * The saved model can be loaded and used to predict the class of new, unseen data points.
    * The new data must be formatted in the same way as the training data.
    * The loaded model's `.predict()` method is used to generate the predictions.

### Accuracy and Performance

* The optimization processes (Randomized and Grid Search) aim to maximize the accuracy of the SVM model.
* **Based on the graph below, the accuracy achieved by the optimized SVM model is 97.37%.**
* The final accuracy of the model will depend on the random state of the data split, and the parameters found by the search.
* The output to the console will display the accuracy of the best model found.
* The plot will also show the accuracy in the title.
* The model responds to new data inputs by loading the saved model, and using the loaded model to predict the class of the new data.

### Visualization

The following graph shows the performance of the optimized SVM model:

![SVM Classification Accuracy](https://github.com/Ahnuf-Karim-Chowdhury/Classification-of-breast-cancer-data-using-SVM/blob/main/Approach%20-%2002%20-%20Saving%20the%20Most%20Accurate%20Data/Optimized%20Polynomial%20Graphical%20Representation.png?raw=true)

**Explanation of the Graph:**

* **X-axis (Test Data Index):** Represents the index of each data point in the test dataset.
* **Y-axis (Class):** Represents the class label, where 0 indicates "Malignant" and 1 indicates "Benign".
* **Blue line with circles:** Shows the actual class labels from the test dataset.
* **Red line with crosses:** Shows the predicted class labels by the SVM model.

The graph clearly demonstrates the close match between the actual and predicted labels, highlighting the high accuracy of the optimized SVM model.

### Usage

To use the saved model for new data:

1.  Load the saved model using `pickle.load()`.
2.  Prepare the new data in the same format as the training data.
3.  Use the loaded model's `predict()` method to generate predictions.

